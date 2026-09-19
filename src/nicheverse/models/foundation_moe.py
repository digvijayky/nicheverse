"""MoE variant of the foundation VQ-VAE: encoder MoE, decoder MoE, or both."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..losses import masked_bernoulli_detection_bce, masked_nb_nll
from .encoders import _largest_divisor, build_encoder
from .foundation import FoundationConfig, _GeneHead, load_foundation
from .moe import (
    DeterministicRouter, MoEDecoder, MoENicheDecoder,
    TopKRouter, build_platform_to_expert, moe_mlp,
)
from .quantizers import build_quantizer
from .sparse import SparseBag, SparseBagEncoder

__all__ = ["MoEConfig", "MoEFoundationVQVAE", "save_moe_foundation", "load_moe_foundation"]


@dataclass
class MoEConfig(FoundationConfig):
    num_experts: int = 8
    top_k: int = 2
    moe_mode: str = "topk"
    moe_scope: str = "decoder_full"
    load_balance_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    moe_noise_std: float = 0.1


class MoEFoundationVQVAE(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        hd = list(config.hidden_dims)
        enc_kw = dict(
            vocab_size=config.vocab_size,
            embed_dim=config.gene_embed_dim,
            use_context=config.use_context,
        )
        enc_kw.update(config.encoder_kwargs)
        self.cell_encoder = build_encoder(
            config.encoder_type, in_dim=config.vocab_size,
            out_dim=config.cell_embedding_dim, hidden=hd,
            dropout=config.dropout, n_bags=1, **enc_kw,
        )
        self.neighborhood_encoder = build_encoder(
            config.encoder_type, in_dim=config.vocab_size * 2,
            out_dim=config.neighborhood_embedding_dim, hidden=hd,
            dropout=config.dropout, n_bags=2, **enc_kw,
        )
        self.cell_vq = build_quantizer(
            config.quantizer_type, num_embeddings=config.cell_num_embeddings,
            embedding_dim=config.cell_embedding_dim,
            commitment_cost=config.commitment_cost, distance_metric=config.vq_distance,
            **config.quantizer_kwargs,
        )
        self.neighborhood_vq = build_quantizer(
            config.quantizer_type, num_embeddings=config.neighborhood_num_embeddings,
            embedding_dim=config.neighborhood_embedding_dim,
            commitment_cost=config.commitment_cost, distance_metric=config.vq_distance,
            **config.quantizer_kwargs,
        )
        self.use_cross_attention = config.use_cross_attention
        self.cross_attention_weight = config.cross_attention_weight
        if self.use_cross_attention:
            heads = _largest_divisor(config.cell_embedding_dim, config.cross_attention_heads)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.cell_embedding_dim, num_heads=heads,
                dropout=0.1, batch_first=True,
            )
            self.neighborhood_projection = nn.Linear(
                config.neighborhood_embedding_dim, config.cell_embedding_dim,
            )
        rev = list(reversed(hd))
        self._has_decoder_moe = config.moe_scope in ("decoder_full", "decoder_head")

        if self._has_decoder_moe:
            share_trunk = config.moe_scope == "decoder_head"
            dec_cell_router = self._build_router(config.cell_embedding_dim, config)
            dec_niche_router = self._build_router(config.neighborhood_embedding_dim, config)
            self.cell_decoder = MoEDecoder(
                num_experts=config.num_experts,
                make_trunk_fn=lambda: moe_mlp(config.cell_embedding_dim, rev, config.decoder_hidden, config.dropout),
                make_head_fn=lambda: _GeneHead(config.decoder_hidden, config.vocab_size),
                router=dec_cell_router, share_trunk=share_trunk,
            )
            self.niche_decoder = MoENicheDecoder(
                num_experts=config.num_experts,
                make_trunk_fn=lambda: moe_mlp(config.neighborhood_embedding_dim, rev, config.decoder_hidden, config.dropout),
                make_self_head_fn=lambda: _GeneHead(config.decoder_hidden, config.vocab_size),
                make_nbr_head_fn=lambda: _GeneHead(config.decoder_hidden, config.vocab_size),
                router=dec_niche_router, share_trunk=share_trunk,
            )
        else:
            self.cell_trunk = moe_mlp(config.cell_embedding_dim, rev, config.decoder_hidden, config.dropout)
            self.niche_trunk = moe_mlp(config.neighborhood_embedding_dim, rev, config.decoder_hidden, config.dropout)
            self.cell_head = _GeneHead(config.decoder_hidden, config.vocab_size)
            self.niche_self_head = _GeneHead(config.decoder_hidden, config.vocab_size)
            self.niche_nbr_head = _GeneHead(config.decoder_hidden, config.vocab_size)

        self.cell_log_theta = nn.Parameter(torch.zeros(config.vocab_size))
        self.niche_log_theta = nn.Parameter(torch.zeros(config.vocab_size))

    @staticmethod
    def _build_router(input_dim: int, config: MoEConfig):
        if config.moe_mode == "deterministic":
            return DeterministicRouter(config.num_experts)
        use_platform = config.moe_mode == "hybrid"
        return TopKRouter(
            input_dim=input_dim, num_experts=config.num_experts,
            top_k=config.top_k, noise_std=config.moe_noise_std,
            use_platform=use_platform, n_platforms=config.n_platforms,
        )

    def forward(
        self,
        cell_bag: SparseBag,
        nbr_bag: SparseBag,
        measured: torch.Tensor,
        cell_context: SparseBag | None = None,
        nbr_context: SparseBag | None = None,
        has_context: torch.Tensor | None = None,
        platform_id: torch.Tensor | None = None,
        species_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_cell = self.cell_encoder([cell_bag], [cell_context], has_context)
        z_niche = self.neighborhood_encoder(
            [cell_bag, nbr_bag], [cell_context, nbr_context], has_context)

        with torch.autocast(device_type=z_cell.device.type, enabled=False):
            cell_vq_loss, q_cell, cell_perp, cell_idx = self.cell_vq(z_cell.float().unsqueeze(2))
            q_cell = q_cell.squeeze(2)
            niche_vq_loss, q_niche, niche_perp, niche_idx = self.neighborhood_vq(
                z_niche.float().unsqueeze(2))
            q_niche = q_niche.squeeze(2)
        if self.use_cross_attention:
            proj = self.neighborhood_projection(q_niche)
            attn, _ = self.cross_attention(
                q_cell.unsqueeze(1), proj.unsqueeze(1), proj.unsqueeze(1))
            q_cell_final = q_cell + self.cross_attention_weight * attn.squeeze(1)
        else:
            q_cell_final = q_cell

        dec_aux = torch.zeros((), device=measured.device)
        if self._has_decoder_moe:
            cell_logits, cell_dec_aux = self.cell_decoder(q_cell_final, measured, platform_id)
            niche_self, niche_nbr, niche_dec_aux = self.niche_decoder(q_niche, measured, platform_id)
            dec_aux = cell_dec_aux + niche_dec_aux
        else:
            h_cell = self.cell_trunk(q_cell_final)
            h_niche = self.niche_trunk(q_niche)
            cell_logits = self.cell_head(h_cell, measured)
            niche_self = self.niche_self_head(h_niche, measured)
            niche_nbr = self.niche_nbr_head(h_niche, measured)

        return dict(
            cell_logits=cell_logits,
            niche_self_logits=niche_self,
            niche_nbr_logits=niche_nbr,
            cell_idx=cell_idx.reshape(-1),
            niche_idx=niche_idx.reshape(-1),
            cell_vq_loss=cell_vq_loss,
            niche_vq_loss=niche_vq_loss,
            cell_perplexity=cell_perp,
            niche_perplexity=niche_perp,
            z_cell=z_cell,
            z_niche=z_niche,
            moe_aux_loss=dec_aux,
        )

    def compute_loss(
        self,
        out: dict[str, torch.Tensor],
        cell_target: torch.Tensor,
        nbr_target: torch.Tensor,
        measured: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        m = float(measured.numel())
        lib_c = cell_target.sum(1, keepdim=True)
        lib_n = nbr_target.sum(1, keepdim=True)
        cell_nb = masked_nb_nll(
            cell_target, out["cell_logits"], self.cell_log_theta, measured, library=lib_c
        ) / m
        cell = cell_nb
        dw = float(self.config.detection_weight)
        det = torch.zeros((), device=cell.device)
        if dw > 0:
            det = masked_bernoulli_detection_bce(cell_target, out["cell_logits"], measured) / m
            cell = cell + dw * det
        niche_self = masked_nb_nll(
            cell_target, out["niche_self_logits"], self.niche_log_theta, measured, library=lib_c
        ) / m
        niche_nbr = masked_nb_nll(
            nbr_target, out["niche_nbr_logits"], self.niche_log_theta, measured, library=lib_n
        ) / m
        niche = 0.5 * (niche_self + niche_nbr)
        vq = out["cell_vq_loss"] + out["niche_vq_loss"]
        moe_aux = out.get("moe_aux_loss", torch.zeros((), device=cell.device))
        lb_w = float(self.config.load_balance_weight)
        loss = cell + self.config.niche_weight * niche + self.config.vq_weight * vq + lb_w * moe_aux
        parts = dict(
            loss=float(loss.detach()),
            cell_nb=float(cell_nb.detach()),
            detection=float(det.detach()),
            niche=float(niche.detach()),
            vq=float(vq.detach()),
            moe_aux=float(moe_aux.detach()) if isinstance(moe_aux, torch.Tensor) else float(moe_aux),
            cell_perplexity=float(out["cell_perplexity"].detach()),
            niche_perplexity=float(out["niche_perplexity"].detach()),
        )
        return loss, parts

    @torch.inference_mode()
    def encode(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        out = self.forward(*args, **kwargs)
        if was_training:
            self.train()
        return {k: out[k] for k in ("cell_idx", "niche_idx", "z_cell", "z_niche")}

    @classmethod
    def from_pretrained_base(
        cls, base_path: str | Path, moe_config: MoEConfig, device: str = "cpu"
    ) -> MoEFoundationVQVAE:
        base = load_foundation(base_path, device=device)
        model = cls(moe_config).to(device)
        model.cell_encoder.load_state_dict(base.cell_encoder.state_dict())
        model.neighborhood_encoder.load_state_dict(
            base.neighborhood_encoder.state_dict())
        model.cell_vq.load_state_dict(base.cell_vq.state_dict())
        model.neighborhood_vq.load_state_dict(base.neighborhood_vq.state_dict())
        if model.use_cross_attention and base.use_cross_attention:
            model.cross_attention.load_state_dict(base.cross_attention.state_dict())
            model.neighborhood_projection.load_state_dict(
                base.neighborhood_projection.state_dict())
        if model._has_decoder_moe:
            base_cell_trunk_sd = base.cell_trunk.state_dict()
            base_cell_head_sd = base.cell_head.state_dict()
            base_niche_trunk_sd = base.niche_trunk.state_dict()
            base_niche_self_head_sd = base.niche_self_head.state_dict()
            base_niche_nbr_head_sd = base.niche_nbr_head.state_dict()
            if moe_config.moe_scope == "decoder_head":
                model.cell_decoder.shared_trunk.load_state_dict(base_cell_trunk_sd, strict=False)
                for eh in model.cell_decoder.expert_heads:
                    eh.load_state_dict(base_cell_head_sd)
                model.niche_decoder.shared_trunk.load_state_dict(base_niche_trunk_sd, strict=False)
                for sh in model.niche_decoder.expert_self_heads:
                    sh.load_state_dict(base_niche_self_head_sd)
                for nh in model.niche_decoder.expert_nbr_heads:
                    nh.load_state_dict(base_niche_nbr_head_sd)
            else:
                for expert in model.cell_decoder.experts:
                    expert.trunk.load_state_dict(base_cell_trunk_sd, strict=False)
                    expert.head.load_state_dict(base_cell_head_sd)
                for i in range(moe_config.num_experts):
                    model.niche_decoder.expert_trunks[i].load_state_dict(base_niche_trunk_sd, strict=False)
                    model.niche_decoder.expert_self_heads[i].load_state_dict(base_niche_self_head_sd)
                    model.niche_decoder.expert_nbr_heads[i].load_state_dict(base_niche_nbr_head_sd)
        else:
            model.cell_trunk.load_state_dict(base.cell_trunk.state_dict(), strict=False)
            model.niche_trunk.load_state_dict(base.niche_trunk.state_dict(), strict=False)
            model.cell_head.load_state_dict(base.cell_head.state_dict())
            model.niche_self_head.load_state_dict(base.niche_self_head.state_dict())
            model.niche_nbr_head.load_state_dict(base.niche_nbr_head.state_dict())
        model.cell_log_theta.data.copy_(base.cell_log_theta.data)
        model.niche_log_theta.data.copy_(base.niche_log_theta.data)
        return model


def save_moe_foundation(model: MoEFoundationVQVAE, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()}, path)
    cfg = model.config.to_dict()
    cfg["vocabulary"] = f"<{len(model.config.vocabulary)} tokens>"
    path.with_suffix(".json").write_text(json.dumps(cfg, indent=2))
    return path


def load_moe_foundation(
    path: str | Path, device: str | torch.device = "cpu"
) -> MoEFoundationVQVAE:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MoEFoundationVQVAE(MoEConfig.from_dict(ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
