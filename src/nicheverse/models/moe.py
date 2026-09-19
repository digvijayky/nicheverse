"""Mixture of Experts layers for the foundation VQ-VAE encoder and decoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse import SparseBag, SparseBagEncoder

__all__ = [
    "TopKRouter", "DeterministicRouter", "MoEDecoder", "MoENicheDecoder",
    "MoESparseBagEncoder", "build_platform_to_expert", "build_tissue_to_expert",
    "moe_mlp", "PLATFORM_GROUPS", "TISSUE_SYSTEMS",
]

PLATFORM_GROUPS = {
    "Xenium": [
        "Xenium", "Xenium (280 gene biomarker panel)", "Xenium (gene and protein)",
        "Xenium Prime 5K", "Xenium WTx",
    ],
    "CosMx": ["CosMx", "CosMx 5642-plex", "CosMx 6K", "CosMx 6K (multiomic)", "CosMx WTx"],
    "MERFISH": ["MERFISH", "MERFISH (Allen ABC)", "MERFISH WTx"],
    "Visium": ["Visium"],
    "STARmap": ["STARmap", "STARmap PLUS"],
    "seqFISH": ["seqFISH", "seqFISH+"],
    "osmFISH": ["osmFISH"],
    "other": ["BARISTAseq", "EEL FISH", "RAEFISH", "RIBOmap"],
}

TISSUE_SYSTEMS = {
    "Neural": ["Brain", "Cortex", "Hypothalamus", "Spinal cord", "Whole brain", "Retina"],
    "Immune": ["Bone marrow", "Lymph node", "Tonsil"],
    "Thoracic": ["Lung", "Heart", "Breast"],
    "GI": ["Colon", "Liver", "Pancreas"],
    "Renal": ["Kidney"],
    "Reproductive": ["Ovary", "Cervix", "Prostate", "Placenta"],
    "Skin": ["Skin"],
    "Other": ["Bone", "Carotid artery", "Multi-tissue", "Whole pup", "Whole embryo"],
}


def moe_mlp(in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def build_platform_to_expert(platform_names: list[str]) -> dict[int, int]:
    name_to_group: dict[str, int] = {}
    for gid, (_, members) in enumerate(PLATFORM_GROUPS.items()):
        for m in members:
            name_to_group[m] = gid
    return {pid: name_to_group.get(name, len(PLATFORM_GROUPS) - 1)
            for pid, name in enumerate(platform_names)}


def build_tissue_to_expert(tissue_names: list[str]) -> dict[int, int]:
    name_to_sys: dict[str, int] = {}
    for sid, (_, members) in enumerate(TISSUE_SYSTEMS.items()):
        for m in members:
            name_to_sys[m] = sid
    return {tid: name_to_sys.get(name, len(TISSUE_SYSTEMS) - 1)
            for tid, name in enumerate(tissue_names)}


class TopKRouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        use_platform: bool = False,
        n_platforms: int = 1,
        platform_embed_dim: int = 16,
        z_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.z_loss_weight = z_loss_weight
        self.use_platform = use_platform
        gate_in = input_dim
        if use_platform:
            self.platform_embed = nn.Embedding(n_platforms, platform_embed_dim)
            nn.init.normal_(self.platform_embed.weight, std=0.02)
            gate_in += platform_embed_dim
        self.gate = nn.Linear(gate_in, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor, platform_id: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_platform and platform_id is not None:
            pe = self.platform_embed(platform_id.reshape(-1))
            gate_in = torch.cat([x, pe], dim=-1)
        else:
            gate_in = x
        logits = self.gate(gate_in)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_vals, dim=-1)
        probs = F.softmax(logits, dim=-1)
        tokens_per_expert = F.one_hot(top_idx[:, 0], self.num_experts).float().mean(0)
        balance_loss = self.num_experts * (tokens_per_expert * probs.mean(0)).sum()
        z_loss = logits.logsumexp(dim=-1).square().mean()
        aux = balance_loss + self.z_loss_weight * z_loss
        return weights, top_idx, aux


class DeterministicRouter(nn.Module):
    def __init__(self, num_experts: int, platform_to_expert: dict[int, int] | None = None) -> None:
        super().__init__()
        self.num_experts = num_experts
        mapping = torch.zeros(1024, dtype=torch.long)
        if platform_to_expert:
            for pid, eid in platform_to_expert.items():
                mapping[pid] = min(eid, num_experts - 1)
        self.register_buffer("_map", mapping)

    def forward(
        self, x: torch.Tensor, platform_id: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        dev = x.device
        if platform_id is not None:
            eid = self._map[platform_id.reshape(-1).clamp(0, self._map.shape[0] - 1)]
        else:
            eid = torch.zeros(b, dtype=torch.long, device=dev)
        weights = torch.ones(b, 1, device=dev, dtype=x.dtype)
        indices = eid.unsqueeze(1)
        return weights, indices, torch.zeros((), device=dev)


class MoESparseBagEncoder(nn.Module):
    """Encoder MoE: shared gene embeddings + pooling, per-expert norm+MLP."""

    def __init__(
        self,
        base_encoder: SparseBagEncoder,
        num_experts: int,
        router: TopKRouter | DeterministicRouter,
    ) -> None:
        super().__init__()
        self.base = base_encoder
        self.num_experts = num_experts
        self.router = router
        pooled_dim = base_encoder.norm.normalized_shape[0]
        hidden = [m.out_features for m in base_encoder.mlp if isinstance(m, nn.Linear)][:-1]
        out_dim = [m.out_features for m in base_encoder.mlp if isinstance(m, nn.Linear)][-1]
        dp = 0.0
        for m in base_encoder.mlp:
            if isinstance(m, nn.Dropout):
                dp = m.p
                break
        self.expert_norms = nn.ModuleList(
            [nn.LayerNorm(pooled_dim) for _ in range(num_experts)]
        )
        self.expert_mlps = nn.ModuleList(
            [moe_mlp(pooled_dim, hidden, out_dim, dp) for _ in range(num_experts)]
        )
        for i in range(num_experts):
            self.expert_norms[i].load_state_dict(base_encoder.norm.state_dict())
        self._last_aux: torch.Tensor | float = 0.0

    def forward(
        self,
        x: torch.Tensor | Sequence[SparseBag],
        context: Sequence[SparseBag | None] | None = None,
        has_context: torch.Tensor | None = None,
        platform_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.base._pool_all(x, context, has_context)
        pooled_normed = self.base.norm(pooled)
        weights, indices, aux = self.router(pooled_normed, platform_id)
        B = pooled.shape[0]
        out_dim = self.expert_mlps[0][-1].out_features
        out = torch.zeros(B, out_dim, device=pooled.device, dtype=pooled.dtype)
        top_k = indices.shape[1]
        for k in range(top_k):
            eid = indices[:, k]
            w = weights[:, k : k + 1]
            for e in range(self.num_experts):
                mask = eid == e
                if not mask.any():
                    continue
                h = self.expert_mlps[e](self.expert_norms[e](pooled[mask]))
                out[mask] = out[mask] + w[mask] * h
        self._last_aux = aux
        return out


class _Expert(nn.Module):
    def __init__(self, trunk: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x), measured)


class MoEDecoder(nn.Module):
    def __init__(
        self,
        num_experts: int,
        make_trunk_fn,
        make_head_fn,
        router: TopKRouter | DeterministicRouter,
        share_trunk: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = router
        self.share_trunk = share_trunk
        if share_trunk:
            self.shared_trunk = make_trunk_fn()
            self.expert_heads = nn.ModuleList([make_head_fn() for _ in range(num_experts)])
        else:
            self.experts = nn.ModuleList(
                [_Expert(make_trunk_fn(), make_head_fn()) for _ in range(num_experts)]
            )

    def forward(
        self, x: torch.Tensor, measured: torch.Tensor, platform_id: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, indices, aux_loss = self.router(x, platform_id)
        b, top_k = indices.shape
        out = torch.zeros(b, measured.shape[0], device=x.device, dtype=x.dtype)
        for k in range(top_k):
            eid = indices[:, k]
            w = weights[:, k].unsqueeze(1)
            for e in range(self.num_experts):
                mask = eid == e
                if not mask.any():
                    continue
                if self.share_trunk:
                    out[mask] += w[mask] * self.expert_heads[e](self.shared_trunk(x[mask]), measured)
                else:
                    out[mask] += w[mask] * self.experts[e](x[mask], measured)
        return out, aux_loss


class MoENicheDecoder(nn.Module):
    def __init__(
        self,
        num_experts: int,
        make_trunk_fn,
        make_self_head_fn,
        make_nbr_head_fn,
        router: TopKRouter | DeterministicRouter,
        share_trunk: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = router
        self.share_trunk = share_trunk
        if share_trunk:
            self.shared_trunk = make_trunk_fn()
            self.expert_self_heads = nn.ModuleList([make_self_head_fn() for _ in range(num_experts)])
            self.expert_nbr_heads = nn.ModuleList([make_nbr_head_fn() for _ in range(num_experts)])
        else:
            self.expert_trunks = nn.ModuleList([make_trunk_fn() for _ in range(num_experts)])
            self.expert_self_heads = nn.ModuleList([make_self_head_fn() for _ in range(num_experts)])
            self.expert_nbr_heads = nn.ModuleList([make_nbr_head_fn() for _ in range(num_experts)])

    def forward(
        self, x: torch.Tensor, measured: torch.Tensor, platform_id: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, indices, aux_loss = self.router(x, platform_id)
        b, top_k = indices.shape
        m = measured.shape[0]
        self_out = torch.zeros(b, m, device=x.device, dtype=x.dtype)
        nbr_out = torch.zeros(b, m, device=x.device, dtype=x.dtype)
        for k in range(top_k):
            eid = indices[:, k]
            w = weights[:, k].unsqueeze(1)
            for e in range(self.num_experts):
                mask = eid == e
                if not mask.any():
                    continue
                if self.share_trunk:
                    h = self.shared_trunk(x[mask])
                else:
                    h = self.expert_trunks[e](x[mask])
                self_out[mask] += w[mask] * self.expert_self_heads[e](h, measured)
                nbr_out[mask] += w[mask] * self.expert_nbr_heads[e](h, measured)
        return self_out, nbr_out, aux_loss
