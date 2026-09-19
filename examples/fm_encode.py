"""Encode one or all staged datasets with the foundation checkpoint (no subsampling).

Writes <out>/<dataset>/codes.npz with, in the ORIGINAL adata row order:
  cell_code, niche_code, X_cell_embedding (float16), X_neighborhood_embedding (float16).

Usage:
  python -u examples/fm_encode.py --ckpt .../model.pt --shards .../shards --dataset mydata
  python -u examples/fm_encode.py --ckpt .../model.pt --shards .../shards --all
"""
import os, json, time, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from nicheverse.data.multipanel import MultiPanelSpatialDataset
from nicheverse.models.foundation import load_foundation
from nicheverse.models.sparse import SparseBag


def bags(b, dev):
    g = lambda k: b[k].to(dev, non_blocking=True)
    cell = SparseBag(g('cell_idx'), g('cell_val'), g('cell_off'))
    nbr = SparseBag(g('nbr_idx'), g('nbr_val'), g('nbr_off'))
    ctx = SparseBag(g('ctx_idx'), g('ctx_val'), g('ctx_off')) if 'ctx_idx' in b else None
    nctx = SparseBag(g('nbr_ctx_idx'), g('nbr_ctx_val'), g('nbr_ctx_off')) if 'nbr_ctx_idx' in b else None
    has = g('has_ctx') if 'has_ctx' in b else None
    return cell, nbr, ctx, nctx, has


def encode_one(shards, index, model, dev, ds_name, out_root, batch_size=8192, workers=4):
    out = os.path.join(out_root, ds_name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(out + '/codes.npz'):
        print('already encoded', ds_name, flush=True); return
    ds = MultiPanelSpatialDataset.from_staging(shards, [ds_name], batch_size=batch_size,
                                               shuffle=False, with_targets=False)
    n = index['datasets'][ds_name]['n_cells']
    cc = np.full(n, -1, np.int32); nc = np.full(n, -1, np.int32)
    zc = zn = None
    offsets, off = {}, 0
    for p in ds.panels[0].shards:
        offsets[p.name] = off
        off += int(np.load(p, mmap_mode='r')['counts_indptr'].shape[0] - 1)
    t0 = time.time()
    with torch.inference_mode():
        for b in DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True):
            cell, nbr, ctx, nctx, has = bags(b, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev != 'cpu'):
                o = model(cell, nbr, b['measured'].to(dev), cell_context=ctx, nbr_context=nctx,
                          has_context=has, platform_id=b['platform_id'].to(dev),
                          species_id=b['species_id'].to(dev))
            rows = b['row'].numpy() + offsets[b['shard']]
            if zc is None:
                zc = np.zeros((n, o['z_cell'].shape[1]), np.float16)
                zn = np.zeros((n, o['z_niche'].shape[1]), np.float16)
            cc[rows] = o['cell_idx'].cpu().numpy().astype(np.int32)
            nc[rows] = o['niche_idx'].cpu().numpy().astype(np.int32)
            zc[rows] = o['z_cell'].float().cpu().numpy().astype(np.float16)
            zn[rows] = o['z_niche'].float().cpu().numpy().astype(np.float16)
    assert (cc >= 0).all(), f'{int((cc < 0).sum())} cells were not encoded'
    order = np.load(os.path.join(shards, ds_name, 'order.npy'))
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    np.savez(out + '/codes.npz', cell_code=cc[inv], niche_code=nc[inv],
             X_cell_embedding=zc[inv], X_neighborhood_embedding=zn[inv])
    el = time.time() - t0
    json.dump(dict(dataset=ds_name, n_cells=int(n), seconds=round(el, 1),
                   cells_per_second=round(n / max(el, 1e-9), 1),
                   active_cell_codes=int(len(np.unique(cc))),
                   active_niche_codes=int(len(np.unique(nc)))),
              open(out + '/info.json', 'w'), indent=1)
    print(f'ENCODED {ds_name} {n} cells in {el / 60:.1f} min, '
          f'{len(np.unique(cc))} cell codes, {len(np.unique(nc))} niche codes', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default='')
    ap.add_argument('--idx', type=int, default=-1)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.shards, 'index.json')))
    names = sorted(index['datasets'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_foundation(a.ckpt, dev).eval()
    todo = names if a.all else [a.dataset or names[a.idx]]
    for ds_name in todo:
        try:
            encode_one(a.shards, index, model, dev, ds_name, a.out,
                       batch_size=a.batch_size, workers=a.workers)
        except Exception as ex:
            import traceback
            print('FAILED', ds_name, type(ex).__name__, str(ex)[:200], flush=True)
            traceback.print_exc()
    print('ALL DONE', flush=True)


main()
