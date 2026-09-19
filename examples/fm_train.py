"""Train the nicheverse foundation model on every staged panel.

Supports single-GPU and multi-GPU (DDP) training. For multi-GPU, launch with
torchrun:

  Single GPU:
    python -u examples/fm_train.py --shards /path/to/shards --out /path/to/ckpt

  4-GPU DDP on one node:
    torchrun --nproc_per_node=4 examples/fm_train.py --shards ... --out ...

  Multi-node (2 nodes x 4 GPUs):
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=nvfm --rdzv_backend=c10d \\
      --rdzv_endpoint=HOST:PORT examples/fm_train.py --shards ... --out ...

Resumes automatically from <out>/last.pt, so the job can be submitted with --requeue.
The global batch size is kept constant across GPU counts: each rank processes
batch_size / world_size cells, and gradients plus the EMA codebook statistics are
all-reduced so the codebook is refreshed from the full global batch.
"""
import os, json, time, math, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from nicheverse.data.multipanel import MultiPanelSpatialDataset
from nicheverse.data.vocabulary import GeneVocabulary
from nicheverse.models.foundation import FoundationConfig, FoundationVQVAE, save_foundation
from nicheverse.models.sparse import SparseBag
from nicheverse._distributed import (
    ddp_env_requested, init_distributed, cleanup_distributed,
    is_main_process, get_rank, get_world_size,
    broadcast_module_, all_reduce_mean_,
)


def log(*a):
    if is_main_process():
        print(*a, flush=True)


def bags(b, dev):
    g = lambda k: b[k].to(dev, non_blocking=True)
    cell = SparseBag(g('cell_idx'), g('cell_val'), g('cell_off'))
    nbr = SparseBag(g('nbr_idx'), g('nbr_val'), g('nbr_off'))
    ctx = SparseBag(g('ctx_idx'), g('ctx_val'), g('ctx_off')) if 'ctx_idx' in b else None
    nctx = SparseBag(g('nbr_ctx_idx'), g('nbr_ctx_val'), g('nbr_ctx_off')) if 'nbr_ctx_idx' in b else None
    has = g('has_ctx') if 'has_ctx' in b else None
    return cell, nbr, ctx, nctx, has


def run_batch(fwd_model, loss_model, b, dev, amp_dtype):
    """Forward through fwd_model (DDP wrapped or raw), loss through loss_model (always raw)."""
    cell, nbr, ctx, nctx, has = bags(b, dev)
    measured = b['measured'].to(dev, non_blocking=True)
    with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
        out = fwd_model(cell, nbr, measured, cell_context=ctx, nbr_context=nctx, has_context=has,
                        platform_id=b['platform_id'].to(dev), species_id=b['species_id'].to(dev))
    loss, parts = loss_model.compute_loss(out, b['cell_target'].to(dev), b['nbr_target'].to(dev), measured)
    return loss, parts, out


def warm_start(model, ds, dev, amp_dtype, n_batches):
    if not hasattr(model.cell_vq, '_kmeans_init'):
        return
    per = {}
    zc, zn = [], []
    model.eval()
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=None, num_workers=2):
            d = b['dataset']
            if per.get(d, 0) >= 2:
                continue
            per[d] = per.get(d, 0) + 1
            cell, nbr, ctx, nctx, has = bags(b, dev)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
                z1 = model.cell_encoder([cell], [ctx], has)
                z2 = model.neighborhood_encoder([cell, nbr], [ctx, nctx], has)
            take = min(64, z1.shape[0])
            zc.append(z1[:take].float().cpu()); zn.append(z2[:take].float().cpu())
            if len(zc) >= n_batches:
                break
    model.train()
    if not zc:
        return
    model.cell_vq._kmeans_init(torch.cat(zc).to(dev))
    model.cell_vq._initialized.fill_(True)
    model.neighborhood_vq._kmeans_init(torch.cat(zn).to(dev))
    model.neighborhood_vq._initialized.fill_(True)
    log(f'  codebooks warm started from {len(zc)} batches over {len(per)} datasets')


def gini(c):
    c = np.sort(np.asarray(c, np.float64))
    n = len(c)
    if n == 0 or c.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(c) / (n * c.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--vocab', default='', help='path to vocab.json (default: <shards>/../meta/vocab.json)')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=4096, help='GLOBAL batch size (split across GPUs)')
    ap.add_argument('--max-target-elements', type=int, default=8_000_000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--cell-codes', type=int, default=1024)
    ap.add_argument('--niche-codes', type=int, default=64)
    ap.add_argument('--gene-embed-dim', type=int, default=256)
    ap.add_argument('--hidden', default='512,256')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--datasets', default='')
    ap.add_argument('--no-condition', action='store_true')
    ap.add_argument('--vq-distance', default='l2', choices=['l2', 'cosine'])
    ap.add_argument('--vq-weight', type=float, default=1.0)
    ap.add_argument('--commitment-cost', type=float, default=0.25)
    ap.add_argument('--keep-epoch-checkpoints', action='store_true')
    ap.add_argument('--amp', default='bf16', choices=['bf16', 'fp16', 'off'])
    ap.add_argument('--warm-batches', type=int, default=64)
    ap.add_argument('--log-every', type=int, default=200)
    ap.add_argument('--max-hours', type=float, default=0.0)
    a = ap.parse_args()

    rank, world_size, local_rank = 0, 1, 0
    if ddp_env_requested():
        rank, world_size, local_rank = init_distributed()
    dev = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'off': None}[a.amp]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(a.out, exist_ok=True)
    index = json.load(open(os.path.join(a.shards, 'index.json')))
    keep = [d for d in a.datasets.split(',') if d] or None
    vocab_path = a.vocab or os.path.join(os.path.dirname(a.shards.rstrip('/')), 'meta', 'vocab.json')
    voc = GeneVocabulary.load(vocab_path)

    per_rank_batch = max(2, a.batch_size // world_size)
    ds = MultiPanelSpatialDataset.from_staging(
        a.shards, keep, batch_size=per_rank_batch, shuffle=True, seed=17,
        max_target_elements=a.max_target_elements)
    n_cells = sum(index['datasets'][p.name]['n_cells'] for p in ds.panels)
    log(f'{len(ds.panels)} datasets, {n_cells} cells, vocabulary {len(voc)}, '
        f'{world_size} GPU(s), per-rank batch {per_rank_batch}')

    cfg = FoundationConfig(
        vocab_size=len(voc), hidden_dims=tuple(int(x) for x in a.hidden.split(',')),
        cell_num_embeddings=a.cell_codes, neighborhood_num_embeddings=a.niche_codes,
        gene_embed_dim=a.gene_embed_dim, decoder_hidden=a.gene_embed_dim,
        n_platforms=len(index['platforms']), n_species=len(index['species']),
        n_datasets=max(d['dataset_id'] for d in index['datasets'].values()) + 1,
        condition_decoders=not a.no_condition,
        vq_distance=a.vq_distance, vq_weight=a.vq_weight, commitment_cost=a.commitment_cost,
        use_cross_attention=True, vocabulary=tuple(voc.symbols))
    model = FoundationVQVAE(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    log(f'model {n_par / 1e6:.1f}M parameters; cell codes {a.cell_codes}, niche codes '
        f'{a.niche_codes}; vq_distance={a.vq_distance}')

    start_epoch = 0
    ck = os.path.join(a.out, 'last.pt')
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s['state_dict'])
        start_epoch = int(s['epoch']) + 1
        log(f'resumed from {ck} at epoch {start_epoch}')
    else:
        warm_start(model, ds, dev, amp_dtype, a.warm_batches)

    if world_size > 1:
        broadcast_module_(model, src=0)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if world_size > 1 else model

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    if os.path.exists(ck) and 'optimizer' in s:
        opt.load_state_dict(s['optimizer'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype is torch.float16)

    hist_path = os.path.join(a.out, 'history.json')
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    t_start = time.time()
    for ep in range(start_epoch, a.epochs):
        ds.set_epoch(ep)
        dl = DataLoader(ds, batch_size=None, num_workers=a.workers, pin_memory=True,
                        prefetch_factor=4 if a.workers else None, persistent_workers=False)
        model.train()
        agg, nb, seen = {}, 0, 0
        cell_use = np.zeros(a.cell_codes, np.int64)
        niche_use = np.zeros(a.niche_codes, np.int64)
        per_ds = {}
        t0 = time.time()
        for b in dl:
            loss, parts, out = run_batch(model, raw_model, b, dev, amp_dtype)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            ci = out['cell_idx'].detach().cpu().numpy()
            ni = out['niche_idx'].detach().cpu().numpy()
            cell_use += np.bincount(ci, minlength=a.cell_codes)
            niche_use += np.bincount(ni, minlength=a.niche_codes)
            u = per_ds.setdefault(b['dataset'], np.zeros(a.cell_codes, np.int64))
            u += np.bincount(ci, minlength=a.cell_codes)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1; seen += len(ci)
            if nb % a.log_every == 0:
                el = time.time() - t0
                log(f'  ep{ep} b{nb} cells={seen} {seen / el:.0f} cells/s '
                    f'loss={agg["loss"] / nb:.4f} nb={agg["cell_nb"] / nb:.4f} '
                    f'niche={agg["niche"] / nb:.4f} vq={agg["vq"] / nb:.4f} '
                    f'active={int((cell_use > 0).sum())}')
        el = time.time() - t0
        rec = dict(epoch=ep, batches=nb, cells=int(seen), seconds=round(el, 1),
                   cells_per_second=round(seen / max(el, 1e-9), 1),
                   gpus=world_size,
                   active_cell_codes=int((cell_use > 0).sum()),
                   active_niche_codes=int((niche_use > 0).sum()),
                   cell_gini=round(gini(cell_use), 4), niche_gini=round(gini(niche_use), 4),
                   **{k: round(v / max(nb, 1), 5) for k, v in agg.items()})
        log('EPOCH', json.dumps(rec))
        if is_main_process():
            hist.append(rec); json.dump(hist, open(hist_path, 'w'), indent=1)
            torch.save(dict(state_dict=raw_model.state_dict(), optimizer=opt.state_dict(),
                            epoch=ep, config=cfg.to_dict()), ck + '.tmp')
            os.replace(ck + '.tmp', ck)
            save_foundation(raw_model, os.path.join(a.out, 'model.pt'))
            if a.keep_epoch_checkpoints:
                save_foundation(raw_model, os.path.join(a.out, f'model_ep{ep:03d}.pt'))
            np.savez(os.path.join(a.out, f'usage_ep{ep:03d}.npz'), cell=cell_use, niche=niche_use,
                     datasets=np.array(sorted(per_ds)),
                     per_dataset=np.stack([per_ds[k] for k in sorted(per_ds)]) if per_ds else np.zeros((0, a.cell_codes)))
        if world_size > 1:
            torch.distributed.barrier()
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            log('time budget reached, stopping cleanly after epoch', ep); break
    log('TRAINING DONE')
    if world_size > 1:
        cleanup_distributed()


main()
