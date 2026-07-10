"""Masked-gene self-supervised pretraining (MAE-style) for encoder initialization."""

from __future__ import annotations

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from ..models.encoders import _mlp, build_encoder
from ..utils import seed_everything


def mae_pretrain(
    adata,
    encoder_type: str = "mlp",
    hidden=(256, 128),
    embedding_dim: int = 64,
    mask_ratio: float = 0.5,
    num_epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    normalize: bool = True,
    log1p: bool = True,
    seed: int = 9,
    device: str | None = None,
):
    """Pretrain an encoder by masked-gene reconstruction; return the pretrained encoder.

    A fraction ``mask_ratio`` of genes is zeroed per cell; the encoder plus a light
    decoder reconstruct the full profile with the loss taken on masked positions
    only, biasing the encoder toward coexpression structure. The returned encoder
    has the same architecture as a model built with the matching
    ``encoder_type`` / ``hidden`` / ``embedding_dim`` and can initialize its
    ``cell_encoder`` via ``load_state_dict``.
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    seed_everything(seed)
    dev = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    a = adata.copy()
    from .trainer import _already_log_normalized

    if _already_log_normalized(a) and (normalize or log1p):
        normalize = log1p = False
    if normalize:
        sc.pp.normalize_total(a)
    if log1p:
        sc.pp.log1p(a)
    x = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    n, g = x.shape
    encoder = build_encoder(encoder_type, in_dim=g, out_dim=embedding_dim, hidden=hidden).to(dev)
    decoder = _mlp(embedding_dim, list(reversed(list(hidden))), g).to(dev)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    encoder.train()
    decoder.train()
    for _ep in range(num_epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            xb = x[perm[i : i + batch_size]].to(dev)
            if xb.shape[0] < 2:
                continue  # BatchNorm needs > 1 sample
            mask = (torch.rand(xb.shape, generator=gen) < mask_ratio).to(dev)
            recon = decoder(encoder(xb * (~mask)))
            loss = F.mse_loss(recon[mask], xb[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return encoder.cpu().eval()
