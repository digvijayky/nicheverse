"""Hierarchical VQ-VAE: paired cell and neighborhood codebooks with cross-attention.

The core model. Encoders live in :mod:`nicheverse.models.encoders`, quantizers in
:mod:`nicheverse.models.quantizers`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .encoders import _ENCODERS, _largest_divisor, _mlp, build_encoder
from .quantizers import _QUANTIZERS, build_quantizer

logger = logging.getLogger(__name__)

# Cell-branch reconstruction modes.
#   "nb" (NEW DEFAULT): NB-NLL on the RAW counts (softmax-proportion decoder scaled
#       by the OBSERVED total count = the scVI library), PLUS a Bernoulli/BCE
#       detection hurdle when detection_weight>0 (positive by default). No MSE on
#       the cell branch, so there is no log-space-vs-count-space scale mixing; only
#       NB vs BCE need balancing (both are reduced to per-gene means). Needs raw
#       INTEGER counts as the target (captured before normalize/log1p); the log1p
#       expression is the encoder input.
#   "mse": pure MSE-on-log1p (with niche_recon="mse" and detection_weight=0 this is
#       the exact old released MSE-only path).
#   "poisson": pure Poisson count NLL on the raw counts (no dispersion).
#   "both": MSE-on-log1p + NB, summed (the two terms are each reduced so neither
#       dominates); detection added when detection_weight>0.
#   "default": defer to ``recon`` (mse/nb/poisson on the encoder input directly).
CELL_RECON_MODES = ("default", "mse", "nb", "poisson", "both")
# Modes that need the raw-count reconstruction target (and hence a cell_log_theta
# for the NB variants). Membership gates the raw-count plumbing in the trainer.
CELL_RECON_COUNT_MODES = ("nb", "poisson", "both")
# Modes that allocate a learned per-gene NB dispersion.
CELL_RECON_NB_MODES = ("nb", "both")
# Niche-branch reconstruction modes.
#   "mse_dirmult" (NEW DEFAULT): balanced sum of composition MSE + a
#       Dirichlet-multinomial NLL on the aggregated-neighbor composition.
#   "mse": the released composition MSE (pure).
#   "dirichlet_multinomial": pure Dirichlet-multinomial NLL (self half stays MSE).
NICHE_RECON_MODES = ("mse", "dirichlet_multinomial", "mse_dirmult")
# Niche modes that use the Dirichlet-multinomial term (allocate niche_log_alpha).
NICHE_DIRMULT_MODES = ("dirichlet_multinomial", "mse_dirmult")


@dataclass
class ModelConfig:
    """Configuration for :class:`HierarchicalVQVAE`.

    The default field values reproduce the published RCC/BrM production model:
    a cell codebook (256 entries, 64 dims) and a neighborhood codebook
    (32 entries, 256 dims), coupled by multi-head cross-attention.

    Parameters
    ----------
    input_dim
        Number of genes in the input panel.
    hidden_dims
        Width of each hidden layer of the encoder MLP. The decoder uses the
        reversed sequence. Must have length >= 1.
    cell_embedding_dim
        Dimensionality of the cell latent space and cell codebook entries.
    cell_num_embeddings
        Number of cell codebook entries (cell states).
    neighborhood_embedding_dim
        Dimensionality of the neighborhood latent space and codebook entries.
    neighborhood_num_embeddings
        Number of neighborhood codebook entries (niches).
    commitment_cost
        Weight of the VQ commitment loss term (van den Oord 2017, beta).
    use_cross_attention
        If True, condition the cell representation on its niche representation
        via a multi-head cross-attention block before reconstruction.
    cross_attention_weight
        Residual mixing weight for cross-attention output (default 0.5).
    cross_attention_heads
        Number of attention heads (default 4). Automatically reduced to the
        largest divisor of ``cell_embedding_dim`` that is ``<=`` this value when
        the two are incompatible, so any embedding dimension is accepted.
    vq_distance
        Codebook assignment metric: ``"l2"`` (default, squared Euclidean) or
        ``"cosine"`` (assign to the maximum-cosine-similarity code).
    quantizer_type, quantizer_kwargs
        Codebook family (``"vq"`` default; also ``fsq`` / ``soft`` / ``rot`` /
        ``qinco`` / ``pq``) and its extra keyword arguments. See
        :func:`~nicheverse.models.build_quantizer`.
    encoder_type, encoder_kwargs
        Encoder backbone (``"mlp_deep"`` default; also ``mlp`` / ``mlp_plr`` /
        ``residual_mlp`` / ``transformer``) and its extra keyword arguments.
        The default is ``mlp_deep`` (a SwiGLU pre-norm residual MLP, no per-gene
        numerical embedding): it had the best raw codebook health in the RCC
        Xenium sweep, and per-gene numerical embeddings (``mlp_plr`` / PLE)
        degenerate on the very sparse Xenium counts that are the default sweep
        data. See :func:`~nicheverse.models.build_encoder`.
    recon
        Reconstruction likelihood: ``"mse"`` (default, Gaussian), ``"nb"``
        (negative binomial, raw counts), or ``"poisson"`` (raw counts).
    cell_recon
        Cell-branch reconstruction mode. **The default is now ``"nb"``**, a
        negative-binomial count likelihood on the RAW integer counts PLUS a
        Bernoulli detection hurdle:

        ``cell_loss = NB_NLL(raw counts) + detection_weight * BCE(detection)``

        The NB NLL uses a softmax-proportion decoder scaled by the OBSERVED total
        count per cell (the standard scVI library) and allocates a learned per-gene
        dispersion ``cell_log_theta``; BCE is a hurdle on the 0-vs-nonzero mask of
        the raw counts (the decoder output is reused as the detection logits). There
        is NO MSE term on the cell branch, so there is no log-space-vs-count-space
        scale mixing; only the NB and BCE terms need balancing, and both are reduced
        to per-gene means (divided by ``input_dim``) so they are comparable, then
        BCE is weighted by ``detection_weight``. The default needs raw INTEGER counts
        as the NB / detection target, captured before normalize/log1p; the default
        training pipeline provides them via ``TrainConfig(normalize=True,
        log1p=True)`` (log1p expression is the encoder input, raw counts stashed to a
        layer).

        The remaining modes stay selectable so the old behavior is fully
        recoverable: ``"mse"`` = pure MSE-on-log1p (with ``niche_recon="mse"`` and
        ``detection_weight=0`` this is the exact released MSE-only path);
        ``"poisson"`` = a pure Poisson count NLL on the raw counts; ``"both"`` =
        MSE-on-log1p + NB (each term reduced so neither dominates), balanced by
        ``w_mse`` / ``w_nb``; ``"default"`` = defer to ``recon`` on the encoder input
        directly (byte-identical to the pre-composite path). For every count mode
        (``nb`` / ``poisson`` / ``both``), the detection hurdle is added whenever
        ``detection_weight > 0``.
    detection_weight
        Weight of the Bernoulli/BCE detection-hurdle term (per-gene-mean BCE on the
        0-vs-nonzero mask of the raw counts, decoder output reused as the detection
        logits). **Default ``0.5``** (on, so the default ``"nb"`` mode is NB + BCE).
        Applied for any count mode (``nb`` / ``poisson`` / ``both``); IGNORED for
        ``"mse"`` / ``"default"`` (no raw-count target). Set to ``0`` to drop it.
    w_mse, w_nb
        Balancing weights for the MSE-on-log1p and NB-NLL terms of the ``"both"``
        cell mode (unused by the default ``"nb"`` mode, which has no MSE term). Both
        default to ``1.0``. The NB term in ``"both"`` is reduced to a per-gene mean
        so it is comparable to the per-element MSE at unit weight.
    niche_recon
        Neighborhood-branch reconstruction mode. **The default is now
        ``"mse_dirmult"``**, a balanced sum of the composition MSE on the log1p niche
        vector and a Dirichlet-multinomial NLL whose target is the COUNT-SCALE
        aggregated-neighbor composition (the SAME weighted-mean neighbor aggregation
        applied to the RAW counts, so its row sum is a real transcript total, NOT a
        log1p mean): ``niche_loss = w_niche_mse * MSE(log1p niche) + w_dirmult *
        DirMult_NLL(count-scale comp)``, with the DirMult reduced to a per-feature
        mean so it is comparable to the MSE. ``"mse"`` (pure released composition MSE)
        and ``"dirichlet_multinomial"`` (pure count-scale DirMult on the neighbor half,
        log1p MSE on the self half) stay selectable. The DM niche modes require raw
        integer counts (so the count-scale neighbor composition can be built), i.e. a
        count ``cell_recon``; otherwise the trainer raises a clear error.
    w_niche_mse, w_dirmult
        Balancing weights for the two terms of the composite niche loss
        (``"mse_dirmult"``). Both default to ``1.0``; the DirMult term is a
        per-feature mean so it is comparable to the composition MSE at unit weight.
    gene_names
        Tuple of gene names matching ``input_dim``. Recorded in the checkpoint
        so that :func:`nicheverse.predict_codes` can verify gene
        panel compatibility at inference time.

    Raises
    ------
    ValueError
        If ``hidden_dims`` is empty, any dimension is non-positive, or
        ``vq_distance`` is not one of ``{"l2", "cosine"}``.
    """

    input_dim: int
    hidden_dims: tuple[int, ...] = (256, 128)
    cell_embedding_dim: int = 64
    cell_num_embeddings: int = 256
    neighborhood_embedding_dim: int = 256
    neighborhood_num_embeddings: int = 32
    commitment_cost: float = 0.25
    use_cross_attention: bool = True
    cross_attention_weight: float = 0.5
    cross_attention_heads: int = 4
    vq_distance: str = "l2"
    quantizer_type: str = "vq"
    quantizer_kwargs: dict[str, Any] = field(default_factory=dict)
    encoder_type: str = "mlp_deep"
    encoder_kwargs: dict[str, Any] = field(default_factory=dict)
    recon: str = "mse"
    cell_recon: str = "nb"
    detection_weight: float = 0.5
    w_mse: float = 1.0
    w_nb: float = 1.0
    niche_recon: str = "mse_dirmult"
    w_niche_mse: float = 1.0
    w_dirmult: float = 1.0
    gene_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if len(self.hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer")
        if any(h <= 0 for h in self.hidden_dims):
            raise ValueError(f"hidden_dims must be positive, got {self.hidden_dims}")
        for name in (
            "cell_embedding_dim",
            "cell_num_embeddings",
            "neighborhood_embedding_dim",
            "neighborhood_num_embeddings",
        ):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        if self.vq_distance not in ("l2", "cosine"):
            raise ValueError(f"vq_distance must be 'l2' or 'cosine', got {self.vq_distance!r}")
        if not self.commitment_cost >= 0:
            raise ValueError(f"commitment_cost must be finite and >= 0, got {self.commitment_cost}")
        if self.cross_attention_heads < 1:
            raise ValueError(f"cross_attention_heads must be >= 1, got {self.cross_attention_heads}")
        if self.recon not in ("mse", "nb", "poisson"):
            raise ValueError(f"recon must be one of (mse, nb, poisson), got {self.recon!r}")
        if self.cell_recon not in CELL_RECON_MODES:
            raise ValueError(
                f"cell_recon must be one of {CELL_RECON_MODES}, got {self.cell_recon!r}"
            )
        if self.cell_recon in CELL_RECON_COUNT_MODES and self.recon != "mse":
            # The count-mode cell_recon carries its own likelihood on the raw
            # counts (encoder input stays log1p). Combining it with recon in
            # {nb,poisson} (which forces raw-count encoder input) is contradictory.
            raise ValueError(
                f"cell_recon={self.cell_recon!r} expects recon='mse' (log1p encoder input); "
                f"got recon={self.recon!r}."
            )
        if self.niche_recon not in NICHE_RECON_MODES:
            raise ValueError(
                f"niche_recon must be one of {NICHE_RECON_MODES}, got {self.niche_recon!r}"
            )
        if self.detection_weight < 0:
            raise ValueError(f"detection_weight must be >= 0, got {self.detection_weight}")
        if self.detection_weight > 0 and self.cell_recon not in CELL_RECON_COUNT_MODES:
            # The detection hurdle needs the raw counts as its target, which only the
            # count modes carry. Rather than error (which would force every pure-MSE
            # recovery to also zero detection_weight), silently IGNORE it for
            # non-count modes and warn once, so cell_recon="mse" recovers the exact
            # old MSE-only path even at the default detection_weight.
            logger.warning(
                "detection_weight=%.3g is ignored for cell_recon=%r (no raw-count target); "
                "the detection hurdle only applies to count modes %s.",
                self.detection_weight,
                self.cell_recon,
                CELL_RECON_COUNT_MODES,
            )
        for wname in ("w_mse", "w_nb", "w_niche_mse", "w_dirmult"):
            wv = getattr(self, wname)
            if wv < 0:
                raise ValueError(f"{wname} must be >= 0, got {wv}")
        if self.encoder_type not in _ENCODERS:
            raise ValueError(
                f"encoder_type must be one of {sorted(_ENCODERS)}, got {self.encoder_type!r}"
            )
        if self.quantizer_type not in _QUANTIZERS:
            raise ValueError(
                f"quantizer_type must be one of {sorted(_QUANTIZERS)}, got {self.quantizer_type!r}"
            )
        if self.gene_names and len(self.gene_names) != self.input_dim:
            raise ValueError(
                f"len(gene_names)={len(self.gene_names)} != input_dim={self.input_dim}. "
                "Either pass an empty tuple or a tuple aligned with the gene panel."
            )
        if self.gene_names and len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError("gene_names must be unique (no duplicate gene names).")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (tuples become lists)."""
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        d["gene_names"] = list(self.gene_names)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Build a :class:`ModelConfig` from a JSON-loaded dict.

        Tolerates bytes in ``gene_names`` (which can come out of h5 attribute
        load paths) by decoding them to str.
        """
        d = dict(d)
        # Backward-compat: checkpoints saved before the loss refactor lack these keys.
        # Fall back to the OLD (MSE-only) behavior so their state_dicts (which have no
        # cell_log_theta / niche_log_alpha) still load strictly, instead of the new
        # nb / mse_dirmult dataclass defaults which would allocate those params.
        d.setdefault("cell_recon", "default")
        d.setdefault("niche_recon", "mse")
        d.setdefault("detection_weight", 0.0)
        d["hidden_dims"] = tuple(int(x) for x in d.get("hidden_dims", (256, 128)))
        gn = d.get("gene_names", ())
        gn_clean: list[str] = []
        for g in gn:
            if isinstance(g, bytes):
                gn_clean.append(g.decode("utf-8"))
            else:
                gn_clean.append(str(g))
        d["gene_names"] = tuple(gn_clean)
        # Drop unknown keys gracefully so old configs still load.
        known = {f for f in cls.__dataclass_fields__}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)


class HierarchicalVQVAE(nn.Module):
    """Two-codebook hierarchical VQ-VAE for spatial transcriptomics.

    Parameters
    ----------
    config
        :class:`ModelConfig` instance with input dimensionality, codebook sizes,
        and cross-attention flag.

    Notes
    -----
    The cross-attention block (when enabled) gives the cell representation
    access to the niche representation via a single multi-head attention call:

    ``q_cell_final = q_cell + cross_attention_weight * Attention(q_cell, proj(q_neigh), proj(q_neigh))``

    The residual weight is exposed as ``config.cross_attention_weight`` (default
    0.5) so it can be ablated or tuned without code changes.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hd = list(config.hidden_dims)
        self.cell_encoder = build_encoder(
            config.encoder_type,
            in_dim=config.input_dim,
            out_dim=config.cell_embedding_dim,
            hidden=hd,
            **config.encoder_kwargs,
        )
        self.neighborhood_encoder = build_encoder(
            config.encoder_type,
            in_dim=config.input_dim * 2,
            out_dim=config.neighborhood_embedding_dim,
            hidden=hd,
            **config.encoder_kwargs,
        )
        self.cell_vq = build_quantizer(
            config.quantizer_type,
            num_embeddings=config.cell_num_embeddings,
            embedding_dim=config.cell_embedding_dim,
            commitment_cost=config.commitment_cost,
            distance_metric=config.vq_distance,
            **config.quantizer_kwargs,
        )
        self.neighborhood_vq = build_quantizer(
            config.quantizer_type,
            num_embeddings=config.neighborhood_num_embeddings,
            embedding_dim=config.neighborhood_embedding_dim,
            commitment_cost=config.commitment_cost,
            distance_metric=config.vq_distance,
            **config.quantizer_kwargs,
        )
        self.use_cross_attention = config.use_cross_attention
        self.cross_attention_weight = config.cross_attention_weight
        if self.use_cross_attention:
            heads = _largest_divisor(config.cell_embedding_dim, config.cross_attention_heads)
            if heads != config.cross_attention_heads:
                logger.warning(
                    "cell_embedding_dim=%d is not divisible by cross_attention_heads=%d; "
                    "using %d head(s) instead.",
                    config.cell_embedding_dim,
                    config.cross_attention_heads,
                    heads,
                )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.cell_embedding_dim,
                num_heads=heads,
                dropout=0.1,
                batch_first=True,
            )
            self.neighborhood_projection = nn.Linear(
                config.neighborhood_embedding_dim, config.cell_embedding_dim
            )
        self.cell_decoder = _mlp(config.cell_embedding_dim, list(reversed(hd)), config.input_dim)
        self.neighborhood_decoder = _mlp(
            config.neighborhood_embedding_dim, list(reversed(hd)), config.input_dim * 2
        )
        if config.recon == "nb" or config.cell_recon in CELL_RECON_NB_MODES:
            self.cell_log_theta = nn.Parameter(torch.zeros(config.input_dim))
        if config.niche_recon in NICHE_DIRMULT_MODES:
            # Per-feature log-concentration for the Dirichlet-multinomial niche
            # likelihood (the niche decoder emits (B, 2*input_dim); the DirMult is
            # applied to the aggregated-neighbor half, input_dim features).
            self.niche_log_alpha = nn.Parameter(torch.zeros(config.input_dim))

    def forward(
        self,
        cell_features: torch.Tensor,
        neighborhood_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the full forward pass.

        Parameters
        ----------
        cell_features
            ``(B, input_dim)`` normalized expression vector per cell.
        neighborhood_features
            ``(B, 2 * input_dim)`` concatenation of self and aggregated
            neighbor expression.

        Returns
        -------
        cell_recon, neigh_recon
            Reconstructions of the inputs.
        cell_vq_loss, neigh_vq_loss
            Scalar VQ losses (commitment plus diversity, plus codebook MSE if
            EMA is disabled).
        cell_idx, neigh_idx
            ``(B, 1)`` hard code assignments.
        cell_perp, neigh_perp
            Batch perplexities of the two codebooks.
        """
        z_cell = self.cell_encoder(cell_features).unsqueeze(2)
        cell_vq_loss, q_cell, cell_perp, cell_idx = self.cell_vq(z_cell)
        q_cell = q_cell.squeeze(2)
        z_neigh = self.neighborhood_encoder(neighborhood_features).unsqueeze(2)
        neigh_vq_loss, q_neigh, neigh_perp, neigh_idx = self.neighborhood_vq(z_neigh)
        q_neigh = q_neigh.squeeze(2)
        if self.use_cross_attention:
            proj = self.neighborhood_projection(q_neigh)
            attn, _ = self.cross_attention(
                q_cell.unsqueeze(1), proj.unsqueeze(1), proj.unsqueeze(1)
            )
            q_cell_final = q_cell + self.cross_attention_weight * attn.squeeze(1)
        else:
            q_cell_final = q_cell
        cell_recon = self.cell_decoder(q_cell_final)
        neigh_recon = self.neighborhood_decoder(q_neigh)
        return (
            cell_recon,
            neigh_recon,
            cell_vq_loss,
            neigh_vq_loss,
            cell_idx,
            neigh_idx,
            cell_perp,
            neigh_perp,
        )

    @torch.inference_mode()
    def encode(
        self, cell_features: torch.Tensor, neighborhood_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cell_idx, neighborhood_idx)`` code assignments without reconstruction.

        A lightweight inference helper for annotating cells: it skips the
        decoders and returns the two hard code-index vectors of shape ``(B,)``.
        """
        was_training = self.training
        self.eval()
        _, _, _, _, cell_idx, neigh_idx, _, _ = self.forward(cell_features, neighborhood_features)
        if was_training:
            self.train()
        return cell_idx.reshape(-1), neigh_idx.reshape(-1)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> HierarchicalVQVAE:
        """Load a trained model from a checkpoint written by :func:`save_checkpoint`."""
        return load_checkpoint(path, device=device)


def save_checkpoint(model: HierarchicalVQVAE, path: str | Path) -> Path:
    """Save model state dict and embedded :class:`ModelConfig` to ``path``.

    Writes ``path`` (a ``.pt`` file) and a sibling ``.json`` of the config for
    human inspection.

    Returns
    -------
    Path
        The checkpoint path actually written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config.to_dict()},
        path,
    )
    cfg_json = path.with_suffix(".json")
    cfg_json.write_text(json.dumps(model.config.to_dict(), indent=2))
    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    config: ModelConfig | None = None,
) -> HierarchicalVQVAE:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Parameters
    ----------
    path
        Path to the ``.pt`` file.
    device
        Target device for the loaded model.
    config
        Required only for legacy bare-state-dict checkpoints with no embedded
        config. New checkpoints carry the config and ignore this argument.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the checkpoint is a legacy bare state dict and ``config`` is None.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:  # legacy checkpoints / objects not on the safe allowlist
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt and "state_dict" in ckpt:
        cfg = ModelConfig.from_dict(ckpt["config"])
        state_dict = ckpt["state_dict"]
    else:
        if config is None:
            raise ValueError(
                f"Checkpoint at {path} has no embedded config (legacy bare state_dict). "
                "Pass `config=ModelConfig(...)` explicitly, matching the training-time "
                "input_dim, codebook sizes, and gene_names."
            )
        cfg = config
        state_dict = ckpt
    model = HierarchicalVQVAE(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
