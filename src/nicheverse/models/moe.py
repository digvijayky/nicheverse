"""Mixture of Experts layers for the foundation VQ-VAE encoder and decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TopKRouter", "DeterministicRouter", "MoEDecoder", "MoENicheDecoder",
    "build_platform_to_expert", "moe_mlp",
    "PLATFORM_GROUPS",
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


def moe_mlp(in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def build_platform_to_expert(platform_names: list[str]) -> dict[int, int]:
    name_to_group = {}
    for gid, (_, members) in enumerate(PLATFORM_GROUPS.items()):
        for m in members:
            name_to_group[m] = gid
    result = {}
    for pid, name in enumerate(platform_names):
        result[pid] = name_to_group.get(name, len(PLATFORM_GROUPS) - 1)
    return result


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
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
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
        # load-balancing loss (Switch Transformer)
        probs = F.softmax(logits, dim=-1)
        tokens_per_expert = F.one_hot(top_idx[:, 0], self.num_experts).float().mean(0)
        mean_prob = probs.mean(0)
        balance_loss = self.num_experts * (tokens_per_expert * mean_prob).sum()
        # router z-loss
        z_loss = logits.logsumexp(dim=-1).square().mean()
        aux = balance_loss + 0.1 * z_loss
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
        if self.share_trunk:
            h = self.shared_trunk(x)
            out = torch.zeros(b, measured.shape[0], device=x.device, dtype=x.dtype)
            for k in range(top_k):
                expert_ids = indices[:, k]
                w = weights[:, k].unsqueeze(1)
                for e in range(self.num_experts):
                    mask = expert_ids == e
                    if not mask.any():
                        continue
                    out[mask] += w[mask] * self.expert_heads[e](h[mask], measured)
        else:
            out = torch.zeros(b, measured.shape[0], device=x.device, dtype=x.dtype)
            for k in range(top_k):
                expert_ids = indices[:, k]
                w = weights[:, k].unsqueeze(1)
                for e in range(self.num_experts):
                    mask = expert_ids == e
                    if not mask.any():
                        continue
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
            expert_ids = indices[:, k]
            w = weights[:, k].unsqueeze(1)
            for e in range(self.num_experts):
                mask = expert_ids == e
                if not mask.any():
                    continue
                if self.share_trunk:
                    h = self.shared_trunk(x[mask])
                else:
                    h = self.expert_trunks[e](x[mask])
                self_out[mask] += w[mask] * self.expert_self_heads[e](h, measured)
                nbr_out[mask] += w[mask] * self.expert_nbr_heads[e](h, measured)
        return self_out, nbr_out, aux_loss
