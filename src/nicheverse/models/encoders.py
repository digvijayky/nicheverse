"""Encoder building blocks for nicheverse models.

Every encoder is registered via ``@register_encoder(name)`` and shares the builder
contract ``(in_dim, out_dim, hidden, dropout, **kwargs) -> nn.Module`` mapping a
per-cell gene vector ``x:[B, in_dim]`` to a latent ``[B, out_dim]``. ``hidden[0]``
sets the model width; architecture-specific knobs are read from ``**kwargs``.

Registered encoders:
    mlp             released Linear/BatchNorm/ReLU/Dropout stack (frozen for reproduction)
    residual_mlp    pre-activation residual MLP
    transformer     gene-patch self-attention Transformer
    cnn             1D-CNN over a projected gene sequence
    fast_cnn        lightweight SiLU 1D-CNN
    deep_cnn        residual 1D-CNN with squeeze-and-excitation and multiscale fusion
    gnn             graph attention over an internal gene-patch token graph
    diffusion       U-Net denoiser encoder (middle representation as the latent)
    dit             diffusion Transformer (AdaLN conditioning) with CLS pooling
    set_transformer ISAB and PMA set encoder over gene-patch tokens
    perceiver_io    latent cross-attention (Perceiver IO) over gene-patch tokens
    soft_moe        soft mixture-of-experts routing over gene-patch tokens
    mlp_deep        deep, wide high-capacity pre-norm residual SwiGLU MLP
    mlp_plr         MLP with per-gene periodic numerical embeddings + SwiGLU trunk
    ft_transformer  feature-tokenizer transformer (gene-as-token self-attention)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    """Build a Linear -> BN -> ReLU -> Dropout stack ending in a Linear projection.

    ``hidden`` must be a non-empty sequence; the caller is expected to validate.
    """
    if len(hidden) < 1:
        raise ValueError("hidden must contain at least one layer width")
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


_ENCODERS: dict[str, Callable[..., nn.Module]] = {}


def register_encoder(name: str):
    """Register an encoder builder (callable ``(in_dim, out_dim, hidden, dropout) -> nn.Module``)."""

    def deco(fn: Callable) -> Callable:
        _ENCODERS[name] = fn
        return fn

    return deco


def build_encoder(
    name: str,
    *,
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int],
    dropout: float = 0.2,
    **kwargs: Any,
) -> nn.Module:
    """Build an encoder by registry name. ``"mlp"`` (default) is the released encoder.

    Extra keyword arguments are forwarded to the encoder builder (e.g.
    ``patch_size`` / ``num_heads`` / ``num_layers`` for ``"transformer"``).
    """
    if name not in _ENCODERS:
        raise ValueError(f"unknown encoder_type {name!r}; choose from {sorted(_ENCODERS)}")
    return _ENCODERS[name](in_dim, out_dim, hidden, dropout, **kwargs)


@register_encoder("mlp")
def _mlp_encoder(
    in_dim: int, out_dim: int, hidden: Sequence[int], dropout: float
) -> nn.Sequential:
    return _mlp(in_dim, hidden, out_dim, dropout)


@register_encoder("residual_mlp")
class ResidualMLP(nn.Module):
    """Pre-activation residual MLP encoder (LayerNorm, Linear, GELU, Dropout, Linear + skip).

    Parameters
    ----------
    in_dim, out_dim
        Input and output dimensionality.
    hidden
        Per-block intermediate widths; the residual stream width is ``hidden[0]``.
    dropout
        Dropout probability inside each residual block.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden, dropout: float = 0.2) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        d = hidden[0]
        self.proj_in = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, w),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(w, d),
            )
            for w in hidden
        )
        self.proj_out = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a gene vector to a latent through the residual stack.

        Shape
        -----
        - Input: ``(B, in_dim)`` per-cell (normalized) gene vector.
        - Output: ``(B, out_dim)`` latent embedding.
        """
        x = self.proj_in(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.proj_out(x)


def _largest_divisor(dim: int, target: int) -> int:
    """Largest value ``<= target`` that divides ``dim`` (>= 1); keeps head counts valid."""
    for h in range(min(target, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


@register_encoder("transformer")
class TransformerEncoder(nn.Module):
    """Gene-patch Transformer encoder: patchify the gene vector, self-attend, mean-pool to a latent.

    Models combinatorial (nonlinear) gene interactions an MLP only mixes
    additively. Operates on the existing per-cell gene vector, no extra inputs.

    Parameters
    ----------
    in_dim, out_dim
        Input gene count and output latent dimensionality.
    hidden
        The model width is ``hidden[0]``.
    dropout
        Attention and feed-forward dropout.
    patch_size
        Genes per patch token.
    num_heads
        Requested attention heads (reduced to the largest divisor of the width).
    num_layers
        Number of Transformer encoder layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.2,
        patch_size: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        d = hidden[0]
        self.patch_size = patch_size
        self.n_patches = (in_dim + patch_size - 1) // patch_size
        self.pad = self.n_patches * patch_size - in_dim
        self.patch_embed = nn.Linear(patch_size, d)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=_largest_divisor(d, num_heads),
            dim_feedforward=d * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.out = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify, self-attend over gene patches, mean-pool, and project to a latent.

        Shape
        -----
        - Input: ``(B, in_dim)`` gene vector (right-padded to a multiple of
          ``patch_size``).
        - Intermediate: ``(B, n_patches, hidden[0])`` patch tokens.
        - Output: ``(B, out_dim)`` latent embedding (mean over patch tokens).
        """
        if self.pad:
            x = F.pad(x, (0, self.pad))
        x = x.reshape(x.shape[0], self.n_patches, self.patch_size)
        x = self.patch_embed(x) + self.pos
        x = self.encoder(x)
        return self.out(x.mean(1))


# =============================================================================
# Ported encoder backbones. Everything below is purely additive: the mlp,
# residual_mlp and transformer paths above are frozen for reproduction.
# =============================================================================


# ---- 1D-CNN family (ported from spatial_vqvae encoders) ---------------------


@register_encoder("cnn")
class CNNEncoder(nn.Module):
    """1D-CNN gene encoder: project the gene vector to a sequence, stack GELU conv
    blocks with max-pool downsampling, global-average-pool, and project to the latent.

    Registered under ``encoder_type="cnn"``. Ported from spatial_vqvae
    ``CNNEncoder``. Channel widths default to a growing progression derived from
    ``hidden[0]`` (``[h//4, h//2, h]``); override with ``channels`` /
    ``kernel_sizes`` via kwargs.

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The model width is ``hidden[0]``; it seeds the default channel schedule.
    dropout : float, default=0.2
        Dropout after each conv block. Range ``[0, 1)``.
    channels : Sequence[int] or None, default=None
        Explicit per-block output-channel widths. If ``None``, uses
        ``[max(1, h//4), max(1, h//2), h]`` from ``h = hidden[0]``.
    kernel_sizes : Sequence[int], default=(7, 5, 3)
        Convolution kernel sizes per block (reused for the last block if fewer
        entries than blocks).
    use_batch_norm : bool, default=True
        Insert a ``BatchNorm1d`` after each convolution.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.2,
        channels: Sequence[int] | None = None,
        kernel_sizes: Sequence[int] = (7, 5, 3),
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        h = hidden[0]
        chans = list(channels) if channels is not None else [max(1, h // 4), max(1, h // 2), h]
        self.input_proj = nn.Linear(in_dim, chans[0])
        layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(chans):
            k = kernel_sizes[min(i, len(kernel_sizes) - 1)]
            layers.append(nn.Conv1d(in_ch, out_ch, k, padding=k // 2))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers += [nn.GELU(), nn.Dropout(dropout)]
            if i < len(chans) - 1:
                layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(chans[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to a 1-channel sequence, convolve, global-average-pool, and project out.

        Shape
        -----
        - Input: ``(B, in_dim)`` gene vector.
        - Intermediate: ``(B, 1, channels[0])`` then ``(B, channels[-1], L)`` after
          the conv stack.
        - Output: ``(B, out_dim)`` latent embedding.
        """
        x = self.input_proj(x).unsqueeze(1)
        x = self.conv_layers(x)
        x = self.adaptive_pool(x).squeeze(-1)
        return self.output_proj(x)


@register_encoder("fast_cnn")
class FastCNNEncoder(nn.Module):
    """Lightweight 1D-CNN gene encoder using SiLU conv blocks (ported from
    spatial_vqvae ``FastCNNEncoder``).

    Registered under ``encoder_type="fast_cnn"``. Uses smaller channels than
    ``cnn``, defaulting to ``[h//8, h//4, h//2]`` from ``hidden[0]`` for a cheaper
    encoder.

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The model width is ``hidden[0]``; it seeds the default channel schedule.
    dropout : float, default=0.1
        Dropout before the output projection. Range ``[0, 1)``.
    channels : Sequence[int] or None, default=None
        Explicit per-block output-channel widths. If ``None``, uses
        ``[max(1, h//8), max(1, h//4), max(1, h//2)]``.
    kernel_sizes : Sequence[int], default=(5, 3, 3)
        Convolution kernel sizes per block.
    use_batch_norm : bool, default=True
        Insert ``BatchNorm1d`` after each convolution (else ``Identity`` and a
        conv bias).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.1,
        channels: Sequence[int] | None = None,
        kernel_sizes: Sequence[int] = (5, 3, 3),
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        h = hidden[0]
        chans = (
            list(channels)
            if channels is not None
            else [max(1, h // 8), max(1, h // 4), max(1, h // 2)]
        )
        self.input_proj = nn.Linear(in_dim, chans[0] * 2)
        self.conv_blocks = nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(chans):
            k = kernel_sizes[min(i, len(kernel_sizes) - 1)]
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=not use_batch_norm),
                    nn.BatchNorm1d(out_ch) if use_batch_norm else nn.Identity(),
                    nn.SiLU(),
                )
            )
            if i < len(chans) - 1:
                self.conv_blocks.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(chans[-1], out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve a projected 1-channel sequence with SiLU blocks and pool to a latent.

        Shape
        -----
        - Input: ``(B, in_dim)`` gene vector.
        - Output: ``(B, out_dim)`` latent embedding.
        """
        x = self.input_proj(x).unsqueeze(1)
        for block in self.conv_blocks:
            x = block(x)
        return self.output(x)


class ResidualBlock1D(nn.Module):
    """Pre-activation residual block for 1D convolutions (spatial_vqvae port)."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        p = kernel_size // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=p)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=p)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.bn1(x)))
        x = self.conv2(self.dropout(F.gelu(self.bn2(x))))
        return x + residual


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel gating for 1D convolutions (spatial_vqvae port)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        red = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, red),
            nn.ReLU(),
            nn.Linear(red, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1)
        return x * w


class DownsampleBlock(nn.Module):
    """Strided conv downsample followed by residual blocks (spatial_vqvae port)."""

    def __init__(
        self, in_ch: int, out_ch: int, n_res: int = 2, kernel: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=2, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(out_ch, kernel, dropout) for _ in range(n_res)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_blocks(F.gelu(self.bn(self.conv(x))))


@register_encoder("deep_cnn")
class DeepCNNEncoder(nn.Module):
    """Deep residual 1D-CNN encoder with squeeze-and-excitation gating and multiscale
    feature fusion (ported from spatial_vqvae ``DeepCNNEncoder``).

    Registered under ``encoder_type="deep_cnn"``. The gene vector is projected to
    a short sequence, passed through progressively downsampling residual stages
    (optionally squeeze-and-excitation gated), optionally fused across scales,
    globally pooled, and projected to the latent. Weights are initialized with
    Kaiming/Xavier schemes.

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The model width is ``hidden[0]`` (also the default ``base_channels``).
    dropout : float, default=0.1
        Dropout inside residual blocks and the output head. Range ``[0, 1)``.
    base_channels : int or None, default=None
        Channel width of the first stage; defaults to ``hidden[0]``.
    depth_multiplier : float, default=1.5
        Per-stage channel growth factor (channels capped at 512).
    n_stages : int, default=4
        Number of downsampling residual stages.
    res_blocks_per_stage : int, default=2
        Residual blocks per stage.
    kernel_sizes : Sequence[int], default=(7, 5, 3, 3)
        Convolution kernels for the stem and each stage.
    use_se : bool, default=True
        Apply squeeze-and-excitation channel gating after each stage.
    use_multiscale : bool, default=True
        Fuse all stage outputs (interpolated to a common length) before pooling.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.1,
        base_channels: int | None = None,
        depth_multiplier: float = 1.5,
        n_stages: int = 4,
        res_blocks_per_stage: int = 2,
        kernel_sizes: Sequence[int] = (7, 5, 3, 3),
        use_se: bool = True,
        use_multiscale: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        base_channels = base_channels or hidden[0]
        self.n_stages = n_stages
        self.use_multiscale = use_multiscale
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, base_channels * 2),
            nn.GELU(),
            nn.Linear(base_channels * 2, base_channels * 4),
        )
        channels = [base_channels]
        for _ in range(n_stages):
            channels.append(int(channels[-1] * depth_multiplier))
        channels = [min(c, 512) for c in channels]
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_sizes[0], padding=kernel_sizes[0] // 2),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        self.se_blocks = nn.ModuleList() if use_se else None
        for i in range(n_stages):
            k = kernel_sizes[min(i, len(kernel_sizes) - 1)]
            self.stages.append(
                DownsampleBlock(channels[i], channels[i + 1], res_blocks_per_stage, k, dropout)
            )
            if self.se_blocks is not None:
                self.se_blocks.append(SEBlock1D(channels[i + 1]))
        if use_multiscale:
            self.stage_projs = nn.ModuleList(
                nn.Conv1d(c, channels[-1], 1) for c in channels[1:]
            )
            self.fusion = nn.Sequential(
                nn.Conv1d(channels[-1] * n_stages, channels[-1], 1),
                nn.BatchNorm1d(channels[-1]),
                nn.GELU(),
            )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem, downsampling stages, optional multiscale fusion, and pool.

        Shape
        -----
        - Input: ``(B, in_dim)`` gene vector.
        - Intermediate: ``(B, 1, base_channels*4)`` sequence, then per-stage
          ``(B, channels[i+1], L_i)`` feature maps.
        - Output: ``(B, out_dim)`` latent embedding.
        """
        b = x.size(0)
        x = self.input_proj(x).view(b, 1, -1)
        x = self.stem(x)
        stage_outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.se_blocks is not None:
                x = self.se_blocks[i](x)
            stage_outputs.append(x)
        if self.use_multiscale and len(stage_outputs) > 1:
            target = stage_outputs[-1].size(-1)
            fused = [
                proj(F.interpolate(feat, size=target, mode="linear", align_corners=False))
                for feat, proj in zip(stage_outputs, self.stage_projs, strict=False)
            ]
            x = self.fusion(torch.cat(fused, dim=1))
        x = self.pool(x).squeeze(-1)
        return self.output_head(x)


# ---- Graph attention over an internal gene-patch token graph ----------------


class GraphConvLayer(nn.Module):
    """Attention-aggregating graph convolution over a fully connected token graph.

    Ports the message / attention / update mechanism of the spatial_vqvae
    ``GraphConvLayer`` to a dense self-graph so no external edge list is needed:
    every gene-patch token is a node connected to every other token. For each node
    it concatenates itself with each neighbor, scores the pair with an attention
    MLP, aggregates messages by softmax weights, and updates via an MLP.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.GELU(), nn.Linear(out_dim, 1)
        )
        self.update_mlp = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = h.size(1)
        node = h.unsqueeze(2).expand(-1, -1, n, -1)
        neigh = h.unsqueeze(1).expand(-1, n, -1, -1)
        pair = torch.cat([node, neigh], dim=-1)
        messages = self.message_mlp(pair)
        weights = F.softmax(self.attn_mlp(pair), dim=2)
        aggregated = (messages * weights).sum(dim=2)
        return self.update_mlp(torch.cat([h, aggregated], dim=-1))


@register_encoder("gnn")
class GNNEncoder(nn.Module):
    """Graph-attention gene encoder over an internal token graph.

    Registered under ``encoder_type="gnn"``. Patchifies the gene vector into
    tokens (like ``transformer``), builds a fully connected token graph, and runs
    attention-aggregating message passing (ports spatial_vqvae ``GNNEncoder`` +
    ``GraphConvLayer`` with residual LayerNorm blocks), then mean-pools the tokens
    to the latent. The graph is over gene-patch tokens within a single cell, not
    over spatial neighbors, so it needs no external neighbor graph and is
    batch-independent (safe at ``B=1``).

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The node/token width is ``hidden[0]``.
    dropout : float, default=0.2
        Dropout applied after each message-passing block. Range ``[0, 1)``.
    patch_size : int, default=16
        Genes per patch token.
    num_layers : int, default=3
        Number of graph-convolution (message-passing) layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.2,
        patch_size: int = 16,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        h = hidden[0]
        self.patch_size = patch_size
        self.n_patches = (in_dim + patch_size - 1) // patch_size
        self.pad = self.n_patches * patch_size - in_dim
        self.patch_embed = nn.Linear(patch_size, h)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, h))
        self.layers = nn.ModuleList(GraphConvLayer(h, h) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(h, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify to tokens, run graph message passing, mean-pool to a latent.

        Shape
        -----
        - Input: ``(B, in_dim)`` gene vector (right-padded to a multiple of
          ``patch_size``).
        - Intermediate: ``(B, n_patches, hidden[0])`` token features.
        - Output: ``(B, out_dim)`` latent embedding.
        """
        if self.pad:
            x = F.pad(x, (0, self.pad))
        x = x.reshape(x.shape[0], self.n_patches, self.patch_size)
        x = self.patch_embed(x) + self.pos
        for mp, norm in zip(self.layers, self.norms, strict=False):
            residual = x
            x = mp(x)
            x = norm(x + residual)
            x = self.dropout(F.gelu(x))
        return self.out(x.mean(1))


# ---- Diffusion encoders (ported from spatial_vqvae) -------------------------


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of a scalar diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -freqs)
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffusionBlock(nn.Module):
    """Time-conditioned MLP block for the diffusion denoiser (spatial_vqvae port)."""

    def __init__(self, in_dim: int, out_dim: int, time_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        h = h + self.time_proj(t_emb)
        return self.norm(h)


@register_encoder("diffusion")
class DiffusionEncoder(nn.Module):
    """Diffusion (denoising) encoder ported from spatial_vqvae ``DiffusionEncoder``.

    Registered under ``encoder_type="diffusion"``. A U-Net style time-conditioned
    denoiser; as an encoder it is run at the clean timestep ``t=0`` and the
    bottleneck ("middle") representation is projected to the latent (the up-path
    and noise-prediction head exist for optional denoising pretraining and are not
    used on the encode path).

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The base width is ``hidden[0]``; down blocks widen it to ``[h, 2h, 4h]``.
    dropout : float, default=0.1
        Dropout inside each time-conditioned block. Range ``[0, 1)``.
    time_emb_dim : int, default=128
        Dimensionality of the sinusoidal timestep embedding.
    n_timesteps : int, default=1000
        Nominal diffusion horizon (retained for pretraining; encoding uses ``t=0``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.1,
        time_emb_dim: int = 128,
        n_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        h = hidden[0]
        self.n_timesteps = n_timesteps
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.input_proj = nn.Linear(in_dim, h)
        dims = [h, h * 2, h * 4]
        self.down_blocks = nn.ModuleList(
            DiffusionBlock(in_d, out_d, time_emb_dim, dropout)
            for in_d, out_d in zip([h, *dims[:-1]], dims, strict=False)
        )
        self.middle = DiffusionBlock(dims[-1], dims[-1], time_emb_dim, dropout)
        self.up_blocks = nn.ModuleList(
            DiffusionBlock(in_d * 2, out_d, time_emb_dim, dropout)
            for in_d, out_d in zip(dims[::-1], [*dims[-2::-1], h], strict=False)
        )
        self.output_proj = nn.Linear(dims[-1], out_dim)
        self.noise_pred = nn.Linear(h, in_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Embed the timestep, run the down path, and project the bottleneck to a latent.

        Parameters
        ----------
        x : torch.Tensor
            Gene vector, shape ``(B, in_dim)``.
        t : torch.Tensor or None, default=None
            Integer timestep per cell, shape ``(B,)``. ``None`` uses the clean
            timestep ``t=0`` (the encoder path).

        Shape
        -----
        - Input: ``(B, in_dim)``.
        - Output: ``(B, out_dim)`` latent embedding (the bottleneck representation).
        """
        b = x.size(0)
        if t is None:
            t = torch.zeros(b, device=x.device, dtype=torch.long)
        t_emb = self.time_embed(t)
        h = self.input_proj(x)
        for block in self.down_blocks:
            h = block(h, t_emb)
        h = self.middle(h, t_emb)
        return self.output_proj(h)


class AdaLN(nn.Module):
    """Adaptive LayerNorm: modulate a normalized token by a conditioning vector."""

    def __init__(self, d_model: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, d_model * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN conditioning (spatial_vqvae port)."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, cond_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.adaln1 = AdaLN(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.adaln2 = AdaLN(d_model, cond_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        return x + self.ff(self.adaln2(x, cond))


@register_encoder("dit")
class DiffusionTransformerEncoder(nn.Module):
    """DiT-style gene encoder ported from spatial_vqvae ``DiffusionTransformerEncoder``.

    Registered under ``encoder_type="dit"``. Patchifies the gene vector, prepends a
    learnable CLS token, and applies AdaLN-conditioned Transformer blocks
    (conditioning is the embedding of the clean timestep ``t=0``); the CLS token is
    projected to the latent.

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The model width is ``hidden[0]``.
    dropout : float, default=0.1
        Attention and feed-forward dropout. Range ``[0, 1)``.
    patch_size : int, default=16
        Genes per patch token.
    num_heads : int, default=8
        Requested attention heads (reduced to the largest divisor of the width).
    num_layers : int, default=4
        Number of DiT blocks.
    time_emb_dim : int, default=128
        Dimensionality of the sinusoidal timestep embedding.
    n_timesteps : int, default=1000
        Nominal diffusion horizon (encoding uses ``t=0``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.1,
        patch_size: int = 16,
        num_heads: int = 8,
        num_layers: int = 4,
        time_emb_dim: int = 128,
        n_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        d = hidden[0]
        nh = _largest_divisor(d, num_heads)
        self.n_timesteps = n_timesteps
        self.patch_size = patch_size
        self.n_patches = (in_dim + patch_size - 1) // patch_size
        self.pad = self.n_patches * patch_size - in_dim
        self.patch_embed = nn.Linear(patch_size, d)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.blocks = nn.ModuleList(
            DiTBlock(d, nh, d * 2, d, dropout) for _ in range(num_layers)
        )
        self.final_adaln = AdaLN(d, d)
        self.out = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Patchify with a CLS token, apply AdaLN-conditioned blocks, read out the CLS latent.

        Parameters
        ----------
        x : torch.Tensor
            Gene vector, shape ``(B, in_dim)`` (right-padded to a multiple of
            ``patch_size``).
        t : torch.Tensor or None, default=None
            Integer conditioning timestep per cell, shape ``(B,)``. ``None`` uses
            ``t=0``.

        Shape
        -----
        - Input: ``(B, in_dim)``.
        - Intermediate: ``(B, n_patches + 1, hidden[0])`` tokens (index 0 is CLS).
        - Output: ``(B, out_dim)`` latent embedding (the CLS token projection).
        """
        b = x.size(0)
        if t is None:
            t = torch.zeros(b, device=x.device, dtype=torch.long)
        t_emb = self.time_embed(t)
        if self.pad:
            x = F.pad(x, (0, self.pad))
        x = x.reshape(b, self.n_patches, self.patch_size)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.final_adaln(x, t_emb)
        return self.out(x[:, 0])


# ---- Set / attention encoders (ported from sota_encoders_vqvae) -------------
# Adapted to the flat contract: the gene vector is patchified into gene-patch
# tokens (pad, reshape, linear-embed) and the set/attention core runs over the
# tokens, then pools to a single latent.


class MAB(nn.Module):
    """Multihead Attention Block (Set Transformer building block; Lee+ ICML 2019)."""

    def __init__(self, dim_q: int, dim_k: int, dim_v: int, num_heads: int) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim_v, num_heads, batch_first=True)
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        self.ln0 = nn.LayerNorm(dim_v)
        self.ln1 = nn.LayerNorm(dim_v)
        self.fc = nn.Sequential(nn.Linear(dim_v, dim_v * 2), nn.GELU(), nn.Linear(dim_v * 2, dim_v))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        qp, kp, vp = self.fc_q(q), self.fc_k(k), self.fc_v(k)
        attn, _ = self.mha(qp, kp, vp)
        h = self.ln0(qp + attn)
        return self.ln1(h + self.fc(h))


class ISAB(nn.Module):
    """Induced Set Attention Block: O(n*m) set attention via learnable induced points."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int) -> None:
        super().__init__()
        self.inds = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.inds)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mab0(self.inds.repeat(x.size(0), 1, 1), x)
        return self.mab1(x, h)


class PMA(nn.Module):
    """Pooling by Multihead Attention: learnable seed queries pool a token set."""

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seeds)
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mab(self.seeds.repeat(x.size(0), 1, 1), x)


class _PatchTokenizer(nn.Module):
    """Pad, reshape and linear-embed a gene vector into gene-patch tokens with a
    learnable positional embedding (shared by the set/attention encoders)."""

    def __init__(self, in_dim: int, patch_size: int, width: int, dropout: float) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (in_dim + patch_size - 1) // patch_size
        self.pad = self.n_patches * patch_size - in_dim
        self.embed = nn.Linear(patch_size, width)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad:
            x = F.pad(x, (0, self.pad))
        x = x.reshape(x.shape[0], self.n_patches, self.patch_size)
        return self.drop(self.embed(x) + self.pos)


@register_encoder("set_transformer")
class SetTransformerEncoder(nn.Module):
    """Set Transformer gene encoder (Lee+ ICML 2019): ISAB blocks over gene-patch
    tokens followed by concat[max, mean, PMA] pooling to a single latent.

    Registered under ``encoder_type="set_transformer"``. Treats the gene patches
    as an unordered set and pools them by concatenating the set max, set mean, and
    a learnable PMA seed query, giving a permutation-equivariant alternative to the
    plain Transformer. PMA-alone pooling is collapse-prone (per the package
    benchmark), so the max/mean statistics are concatenated with the seed readout.

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The token width is ``hidden[0]``.
    dropout : float, default=0.2
        Patch-embedding dropout. Range ``[0, 1)``.
    patch_size : int, default=16
        Genes per patch token.
    num_heads : int, default=4
        Requested attention heads (reduced to the largest divisor of the width).
    num_inds : int, default=16
        Number of learnable induced points per ISAB (bounds attention cost).
    num_isab : int, default=2
        Number of stacked ISAB blocks.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.2,
        patch_size: int = 16,
        num_heads: int = 4,
        num_inds: int = 16,
        num_isab: int = 2,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        h = hidden[0]
        nh = _largest_divisor(h, num_heads)
        self.tok = _PatchTokenizer(in_dim, patch_size, h, dropout)
        self.isab_blocks = nn.ModuleList(ISAB(h, h, nh, num_inds) for _ in range(num_isab))
        self.pma = PMA(h, nh, num_seeds=1)
        # Pool with concat[max, mean, PMA]: PMA-alone collapses the pooled
        # embedding (documented in the package benchmark); max/mean over the set
        # dimension preserve token magnitude and spread so the quantizer sees
        # real structure. Dense set here (no mask), so plain max/mean.
        self.out = nn.Linear(h * 3, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize gene patches, apply ISAB blocks, pool with concat[max, mean, PMA].

        Shape
        -----
        - Input: ``(B, in_dim)`` gene vector.
        - Intermediate: ``(B, n_patches, hidden[0])`` set tokens.
        - Output: ``(B, out_dim)`` latent embedding (concat of set max, set mean,
          and the PMA seed readout).
        """
        h = self.tok(x)
        for block in self.isab_blocks:
            h = block(h)
        mx = h.max(dim=1).values
        mean = h.mean(dim=1)
        seed = self.pma(h).squeeze(1)
        return self.out(torch.cat([mx, mean, seed], dim=-1))


@register_encoder("perceiver_io")
class PerceiverIOEncoder(nn.Module):
    """Perceiver IO gene encoder (Jaegle+ ICLR 2022): a learnable latent array
    cross-attends the gene-patch tokens, refines with self-attention, and a single
    output query cross-attends the latents; the pooled latent concatenates the
    latent-array max, latent-array mean, and the output-query readout.

    Registered under ``encoder_type="perceiver_io"``. Decouples compute from the
    number of gene patches by attending through a fixed-size latent array. The
    output-query readout alone is collapse-prone (per the package benchmark), so
    the max/mean of the refined latent array are concatenated with it.

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The latent/token width is ``hidden[0]``.
    dropout : float, default=0.2
        Attention and feed-forward dropout. Range ``[0, 1)``.
    patch_size : int, default=16
        Genes per patch token.
    num_heads : int, default=4
        Requested attention heads (reduced to the largest divisor of the width).
    num_latents : int, default=64
        Size of the learnable latent array that cross-attends the tokens.
    num_self_layers : int, default=4
        Number of Transformer self-attention layers refining the latents.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.2,
        patch_size: int = 16,
        num_heads: int = 4,
        num_latents: int = 64,
        num_self_layers: int = 4,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        h = hidden[0]
        nh = _largest_divisor(h, num_heads)
        self.tok = _PatchTokenizer(in_dim, patch_size, h, dropout)
        self.latents = nn.Parameter(torch.empty(num_latents, h))
        nn.init.xavier_uniform_(self.latents)
        self.cross_attn = nn.MultiheadAttention(h, nh, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(h)
        self.ln_kv = nn.LayerNorm(h)
        self.self_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                h, nh, dim_feedforward=h * 4, dropout=dropout, batch_first=True, norm_first=True
            )
            for _ in range(num_self_layers)
        )
        self.output_q = nn.Parameter(torch.empty(1, 1, h))
        nn.init.xavier_uniform_(self.output_q)
        self.cross_out = nn.MultiheadAttention(h, nh, dropout=dropout, batch_first=True)
        # Pool with concat[max, mean, output-query readout]: the single learned
        # output query alone collapses the pooled embedding (documented in the
        # package benchmark); max/mean over the refined latent array preserve its
        # magnitude and spread. Dense latent array (no mask), so plain max/mean.
        self.out = nn.Linear(h * 3, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-attend a latent array into the tokens, self-attend, pool concat[max, mean, query].

        Shape
        -----
        - Input: ``(B, in_dim)`` gene vector.
        - Intermediate: ``(B, num_latents, hidden[0])`` latent array.
        - Output: ``(B, out_dim)`` latent embedding (concat of latent-array max,
          latent-array mean, and the output-query readout).
        """
        b = x.shape[0]
        kv = self.ln_kv(self.tok(x))
        q = self.ln_q(self.latents.unsqueeze(0).expand(b, -1, -1))
        h, _ = self.cross_attn(q, kv, kv)
        h = q + h
        for layer in self.self_layers:
            h = layer(h)
        outq = self.output_q.expand(b, -1, -1)
        z, _ = self.cross_out(outq, h, h)
        mx = h.max(dim=1).values
        mean = h.mean(dim=1)
        return self.out(torch.cat([mx, mean, z.squeeze(1)], dim=-1))


@register_encoder("soft_moe")
class SoftMoEEncoder(nn.Module):
    """Soft Mixture-of-Experts gene encoder (Puigcerver+ NeurIPS 2024): each expert
    holds slots that are soft (softmax) combinations of the gene-patch tokens; expert
    outputs are softly recombined per token and pooled by concat[max, mean, PMA] to
    the latent.

    Registered under ``encoder_type="soft_moe"``. Routes tokens to experts with a
    fully differentiable soft dispatch/combine (no hard top-k), so it trains
    stably at small batch sizes. PMA-alone pooling is collapse-prone (per the
    package benchmark), so the recombined-token max/mean are concatenated with the
    PMA seed readout.

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The token/expert width is ``hidden[0]``.
    dropout : float, default=0.2
        Patch-embedding dropout. Range ``[0, 1)``.
    patch_size : int, default=16
        Genes per patch token.
    num_heads : int, default=4
        Requested attention heads for the PMA pooling (reduced to the largest
        divisor of the width).
    num_experts : int, default=8
        Number of expert MLPs.
    slots_per_expert : int, default=2
        Soft slots each expert processes.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.2,
        patch_size: int = 16,
        num_heads: int = 4,
        num_experts: int = 8,
        slots_per_expert: int = 2,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        h = hidden[0]
        nh = _largest_divisor(h, num_heads)
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.tok = _PatchTokenizer(in_dim, patch_size, h, dropout)
        self.phi = nn.Parameter(torch.empty(h, num_experts * slots_per_expert))
        nn.init.xavier_uniform_(self.phi)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(h, h * 2), nn.GELU(), nn.Linear(h * 2, h))
            for _ in range(num_experts)
        )
        self.out_pool = PMA(h, nh, num_seeds=1)
        # Pool with concat[max, mean, PMA]: PMA-alone collapses the pooled
        # embedding (documented in the package benchmark); max/mean over the token
        # dimension preserve magnitude and spread. Dense set here (no mask).
        self.out = nn.Linear(h * 3, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft-dispatch tokens to expert slots, recombine, pool concat[max, mean, PMA].

        Shape
        -----
        - Input: ``(B, in_dim)`` gene vector.
        - Intermediate: ``(B, n_patches, hidden[0])`` tokens and
          ``(B, num_experts * slots_per_expert, hidden[0])`` expert-slot features.
        - Output: ``(B, out_dim)`` latent embedding (concat of token max, token
          mean, and the PMA seed readout).
        """
        tokens = self.tok(x)
        logits = tokens @ self.phi
        dispatch = F.softmax(logits, dim=1)
        x_slot = dispatch.transpose(1, 2) @ tokens
        b, _, width = x_slot.shape
        s = self.slots_per_expert
        x_slot = x_slot.view(b, self.num_experts, s, width)
        y_slot = torch.stack(
            [self.experts[i](x_slot[:, i]) for i in range(self.num_experts)], dim=1
        )
        y_slot = y_slot.view(b, self.num_experts * s, width)
        combine = F.softmax(logits, dim=2)
        y = combine @ y_slot
        mx = y.max(dim=1).values
        mean = y.mean(dim=1)
        seed = self.out_pool(y).squeeze(1)
        return self.out(torch.cat([mx, mean, seed], dim=-1))


# ---- Tabular-DL style encoders (numerical feature embeddings + GLU FFN) -------
# These treat the gene vector as an (unordered) feature set rather than a 1D
# signal or gene patches. Rationale: on tabular / omics feature vectors the
# decisive ingredient is a learned per-feature numerical embedding plus a strong
# feed-forward mixer (Gorishniy 2021/2022; TabM/TabReD 2024), not gene-patch
# attention. `mlp_plr` is the MLP-with-numerical-embeddings variant; `ft_transformer`
# is the feature-tokenizer transformer diagnostic (full gene-gene attention).


class _SwiGLU(nn.Module):
    """Pre-norm residual SwiGLU block: ``x + Wo(SiLU(Wg(LN x)) * Wv(LN x))``.

    GLU-variant feed-forward (Shazeer 2020); more expressive per parameter than a
    plain Linear-GELU-Linear block. Operates on the last dim, so it works for both
    ``(B, d)`` (MLP trunk) and ``(B, T, d)`` (transformer FFN).
    """

    def __init__(self, d: int, mult: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        h = max(1, int(d * mult))
        self.norm = nn.LayerNorm(d)
        self.wg = nn.Linear(d, h)
        self.wv = nn.Linear(d, h)
        self.wo = nn.Linear(h, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        return x + self.wo(self.drop(F.silu(self.wg(z)) * self.wv(z)))


@register_encoder("mlp_plr")
class MLPPLREncoder(nn.Module):
    """MLP with per-gene Periodic-Linear-ReLU numerical embeddings and a pre-norm
    residual SwiGLU trunk (registered under ``encoder_type="mlp_plr"``).

    Each scalar gene value is expanded into a learned embedding before the MLP:
    periodic (Fourier) features with learnable per-gene frequencies, a per-gene
    linear map, then ReLU (Gorishniy et al. 2022, "On Embeddings for Numerical
    Features"). This numerical-embedding step is the ingredient that makes MLPs
    match or beat attention-based models on tabular/omics feature vectors, while
    staying cheap and stable in front of a VQ bottleneck.

    The Gorishniy periodic map ``v = 2*pi * c * x`` with ``c ~ N(0, sigma)``
    calibrates its frequencies to standardized (zero-mean unit-variance) inputs;
    the reference RTDL pipeline quantile/standard-normalizes numeric features
    before the embedding. This model instead feeds raw log1p expression whose
    per-gene mean and scale differ by orders of magnitude across the panel, so a
    single global ``freq_init_sigma`` mistunes most genes: low-count genes (``x``
    near 0) sit near the flat part of every sinusoid and contribute almost
    nothing while high-count genes dominate. To decouple the periodic frequencies
    from the raw log1p scale without any precomputed dataset statistics, a
    learnable per-gene input scale ``s_g`` (shape ``(in_dim,)``, initialized to
    1.0 so the map is identity at init) rescales each gene before the periodic
    map: ``v = 2*pi * (x * s_g) * c``. The model then learns the effective
    per-gene frequency scale, letting low-expression genes contribute. ``s_g`` is
    a bare no-decay parameter (auto-excluded from weight decay by the optimizer's
    bare-parameter rule).

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        The residual-stream width is ``hidden[0]``.
    dropout : float
        Dropout inside the SwiGLU blocks.
    n_freq : int, default 16
        Number of learnable periodic frequencies per gene.
    num_emb_dim : int, default 8
        Per-gene embedding width after the linear+ReLU.
    n_blocks : int or None, default None
        SwiGLU residual blocks (defaults to ``max(2, len(hidden))``).
    freq_init_sigma : float, default 0.1
        Std of the periodic-frequency initialization.
    ffn_mult : float, default 2.0
        SwiGLU hidden expansion factor.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.1,
        n_freq: int = 16,
        num_emb_dim: int = 8,
        n_blocks: int | None = None,
        freq_init_sigma: float = 0.1,
        ffn_mult: float = 2.0,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        d = hidden[0]
        self.in_dim = in_dim
        self.n_freq = n_freq
        # Learnable per-gene input scale (identity at init) so the periodic map is
        # decoupled from the raw per-gene log1p scale; bare no-decay parameter.
        self.in_scale = nn.Parameter(torch.ones(in_dim))
        self.freq = nn.Parameter(torch.randn(in_dim, n_freq) * freq_init_sigma)
        self.emb_w = nn.Parameter(
            torch.randn(in_dim, 2 * n_freq, num_emb_dim) / math.sqrt(2 * n_freq)
        )
        self.emb_b = nn.Parameter(torch.zeros(in_dim, num_emb_dim))
        self.proj_in = nn.Linear(in_dim * num_emb_dim, d)
        nb = n_blocks if n_blocks is not None else max(2, len(hidden))
        self.blocks = nn.ModuleList(_SwiGLU(d, ffn_mult, dropout) for _ in range(nb))
        self.norm_out = nn.LayerNorm(d)
        self.proj_out = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed each gene value periodically, mix with a SwiGLU MLP, project to latent.

        Shape
        -----
        - Input: ``(B, in_dim)``.
        - Output: ``(B, out_dim)``.
        """
        xs = x * self.in_scale.unsqueeze(0)  # per-gene learnable rescale (B, F)
        v = (2.0 * math.pi) * xs.unsqueeze(-1) * self.freq.unsqueeze(0)  # (B, F, n_freq)
        per = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B, F, 2*n_freq)
        emb = torch.einsum("bfk,fkd->bfd", per, self.emb_w) + self.emb_b  # (B, F, num_emb_dim)
        z = self.proj_in(F.relu(emb).reshape(x.size(0), -1))
        for blk in self.blocks:
            z = blk(z)
        return self.proj_out(self.norm_out(z))


@register_encoder("ft_transformer")
class FTTransformerEncoder(nn.Module):
    """Feature-Tokenizer Transformer gene encoder (Gorishniy et al. 2021), registered
    under ``encoder_type="ft_transformer"``.

    Each gene becomes one token (a learned per-gene embedding scaled by the gene's
    value, plus a per-gene bias); a CLS token is prepended; pre-norm multi-head
    self-attention with a SwiGLU feed-forward mixes the gene tokens, and the CLS
    projection is the latent. There is no positional encoding because genes are an
    unordered set. Full gene-gene attention makes this the interpretable
    "co-expression" diagnostic; attention uses the fused/memory-efficient kernel
    (``need_weights=False``) so the ~n_genes token sequence is affordable.

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        Token/model width is ``hidden[0]``.
    dropout : float
        Attention and feed-forward dropout.
    num_heads : int, default 8
        Requested heads (reduced to the largest divisor of the width).
    num_layers : int, default 3
        Number of transformer blocks.
    ffn_mult : float, default 2.0
        SwiGLU hidden expansion factor.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.1,
        num_heads: int = 8,
        num_layers: int = 3,
        ffn_mult: float = 2.0,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        d = hidden[0]
        nh = _largest_divisor(d, num_heads)
        self.tok_w = nn.Parameter(torch.randn(in_dim, d) / math.sqrt(d))
        self.tok_b = nn.Parameter(torch.zeros(in_dim, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "n1": nn.LayerNorm(d),
                    "attn": nn.MultiheadAttention(d, nh, dropout=dropout, batch_first=True),
                    "ffn": _SwiGLU(d, ffn_mult, dropout),
                }
            )
            for _ in range(num_layers)
        )
        self.norm_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize genes, prepend CLS, apply pre-norm attention + SwiGLU, read CLS.

        Shape
        -----
        - Input: ``(B, in_dim)``.
        - Intermediate: ``(B, in_dim + 1, hidden[0])`` tokens (index 0 is CLS).
        - Output: ``(B, out_dim)``.
        """
        b = x.size(0)
        tokens = x.unsqueeze(-1) * self.tok_w.unsqueeze(0) + self.tok_b.unsqueeze(0)  # (B, F, d)
        h = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1)  # (B, F+1, d)
        for layer in self.layers:
            z = layer["n1"](h)
            a, _ = layer["attn"](z, z, z, need_weights=False)
            h = layer["ffn"](h + a)
        return self.out(self.norm_out(h[:, 0]))


@register_encoder("mlp_deep")
class DeepMLPEncoder(nn.Module):
    """Deep, wide, high-capacity MLP (registered under ``encoder_type="mlp_deep"``).

    A linear input projection followed by a deep stack of pre-norm residual SwiGLU
    blocks. Higher capacity than the released ``mlp`` / ``residual_mlp`` (deeper,
    gated feed-forward, pre-norm residual stream), but without the per-gene
    numerical embeddings of ``mlp_plr`` -- so comparing ``mlp_deep`` vs ``mlp_plr``
    isolates the benefit of raw depth/width capacity from the benefit of the
    numerical embedding. Run it wide (e.g. ``hidden=[512]``) and deep (e.g.
    ``n_blocks=8``) for the high-capacity configuration.

    Parameters
    ----------
    in_dim, out_dim : int
        Input gene count and output latent dimensionality.
    hidden : Sequence[int]
        Residual-stream width is ``hidden[0]``.
    dropout : float
        Dropout inside the SwiGLU blocks.
    n_blocks : int or None, default None
        Number of residual SwiGLU blocks (defaults to ``max(4, len(hidden))``).
    ffn_mult : float, default 2.0
        SwiGLU hidden expansion factor.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.1,
        n_blocks: int | None = None,
        ffn_mult: float = 2.0,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")
        d = hidden[0]
        self.proj_in = nn.Linear(in_dim, d)
        nb = n_blocks if n_blocks is not None else max(4, len(hidden))
        self.blocks = nn.ModuleList(_SwiGLU(d, ffn_mult, dropout) for _ in range(nb))
        self.norm_out = nn.LayerNorm(d)
        self.proj_out = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project, apply the deep SwiGLU residual stack, project to the latent.

        Shape
        -----
        - Input: ``(B, in_dim)``.
        - Output: ``(B, out_dim)``.
        """
        z = self.proj_in(x)
        for blk in self.blocks:
            z = blk(z)
        return self.proj_out(self.norm_out(z))
