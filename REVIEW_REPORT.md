# nicheverse code review report

This report covers a top-to-bottom review of `nicheverse` at
`/data1/lesliec/vijay/github/spatial_methodg/nicheverse/` ahead of the
Cancer Cell submission and the public release. The package was reviewed
against the conventions of PyTorch, HuggingFace transformers, and scanpy.

All fixes described below have been applied in place. The existing 7 tests
still pass, and 16 new tests have been added, for a total of 23 tests passing
in 55 seconds.

## 1. What was reviewed

Source modules under `src/nicheverse/`:

1. `__init__.py` (public surface)
2. `model.py` (`ModelConfig`, `VectorQuantizer`, `HierarchicalVQVAE`, save/load)
3. `data.py` (`SpatialDataset` per-sample k-NN aggregation)
4. `io.py` (`load_xenium_run`, `load_xenium_cohort`, `attach_codes_to_adata`)
5. `train.py` (`TrainConfig`, `train_model`)
6. `predict.py` (`predict_codes` with gene-panel checking)
7. `viz.py` (PDF plots)
8. `cli.py` (`nicheverse {preprocess|train|predict|verify|info}`)
9. `utils.py` (`seed_everything`, `env_snapshot`, `sha256_*`)

Plus `pyproject.toml`, `README.md`, `tests/test_model.py`, `tests/test_data.py`,
`tests/test_train_predict.py`, `examples/`, and `docs/`.

## 2. Correctness bugs fixed

1. `model.py:_kmeans_init` (`VectorQuantizer`). When the first training batch
   was smaller than the codebook size `K`, only the first `B*T` codebook slots
   were filled from data; the remaining `K - B*T` slots stayed at the original
   uniform random init. Those slots almost always became immediately dead.
   Fix: when `n_samples < num_embeddings`, fill the first `n_samples` slots
   from the batch and seed the remainder by sampling with replacement (with
   small additive noise) so every slot starts from data.
2. `model.py:VectorQuantizer.forward`. The diversity term used
   `F.softmax(-distances, dim=1)`, i.e. a fixed temperature of 1 implicit in
   the units of squared Euclidean distance. With `embedding_dim` in the
   hundreds the softmax collapsed to a delta and the entropy term lost its
   gradient signal. Fix: added `diversity_temperature` constructor argument
   (default 1.0 to preserve current behavior); the softmax now divides by it.
3. `model.py:VectorQuantizer.forward`. The dead code reset cadence was
   hardcoded as `update_count % 50 == 0`. Fix: exposed as
   `dead_code_reset_interval` constructor argument; the default 50 is also
   promoted to a module constant `DEAD_CODE_RESET_INTERVAL`.
4. `model.py:_reset_dead_codes`. When the input batch is empty the original
   code would call `torch.randint(0, 0, ...)` which raises. Fix: short-circuit
   on `batch_size == 0`.
5. `model.py:HierarchicalVQVAE.__init__`. The cross-attention residual weight
   was hardcoded as `0.5 * attn`. Fix: exposed as `ModelConfig.cross_attention_weight`
   (default 0.5), documented in the class docstring.
6. `model.py:_mlp`. The output projection used `Linear(hidden[-1], out_dim)`
   without checking that `hidden` is non-empty (an empty list would index
   `hidden[-1]` and raise `IndexError`). Fix: explicit length check at the top
   of `_mlp` and at `ModelConfig.__post_init__`, which now rejects empty
   `hidden_dims`.
7. `model.py:ModelConfig.from_dict`. Could fail when `gene_names` came back
   from h5 attribute load as `bytes` rather than `str`, and would raise on
   unknown keys. Fix: decode `bytes` to `str` per-entry and drop unknown keys.
8. `data.py:SpatialDataset`. The per-sample loop could leave entries at `None`
   if a sample had zero or one cells; downstream `torch.stack(out)` would
   raise an opaque error. Fix: explicit validation (`n_cells > 0`,
   shape consistency, `k_neighbors > 0`), an explicit one-cell fallback, and
   a final NaN sweep that raises a clear `RuntimeError` if anything was
   missed.
9. `io.py:load_xenium_run`. The control filter used
   `Control|BLANK|Blank|Codeword|Unassigned` which missed `NegControlProbe`,
   `Deprecated`, `Intergenic`, `antisense`, and was case sensitive. Fix:
   compiled `re.IGNORECASE` pattern that also catches those.
10. `io.py:load_xenium_cohort`. The "Restricting to X genes" warning could
    fail to fire because of the `any(...)` early-exit semantics on
    `gene_sets`. Fix: compute the total dropped count explicitly and warn
    whenever it is > 0; also raise a clear error when `run_dirs` is empty
    instead of silently passing through.
11. `io.py:attach_codes_to_adata`. Indices were always stored as `int64`,
    wasting disk space when `K < 65k`. Fix: choose the narrowest integer
    dtype that fits the maximum index (`int16` for K < 32768, `int32`
    otherwise).
12. `train.py:train_model`. Did not detect when `adata` was already
    log normalized; setting `normalize=True, log1p=True` (the default) would
    silently double-normalize. Fix: detect prior `sc.pp.log1p` /
    `sc.pp.normalize_total` via `adata.uns` keys (the standard scanpy
    convention) and skip with a warning.
13. `predict.py:predict_codes`. Called `_align_genes_to_checkpoint` which
    returned an AnnData view (sometimes); subsequent `sc.pp.normalize_total`
    on a view emits `ImplicitModificationWarning`. Fix: `_align_genes_to_checkpoint`
    now always returns a fresh copy, and a new `_ensure_dense_float32` helper
    realizes `X` as a contiguous float32 array before the data loader.
14. `predict.py`. When `gene_names` is empty on the checkpoint, the original
    code silently returned `adata` unchanged. Fix: raise an actionable
    `ValueError` directing the user to either re-train with `gene_names` set
    or pass `config` explicitly to `load_checkpoint`.
15. `cli.py:verify`. The JSON reference shape was undocumented and the keys
    silently disagreed with what `predict` could write. Fix: `predict` now
    accepts `--report` and writes a canonical
    `{n_cells, predicted_cell_sha256, predicted_neighborhood_sha256}` JSON;
    `verify` accepts that shape AND the legacy `reference_*` keys. The
    contract is documented in the module docstring.
16. `utils.py:seed_everything`. Set `CUBLAS_WORKSPACE_CONFIG` via
    `os.environ.setdefault`, with no warning that this is too late if torch
    is already imported. Fix: detect whether torch is already in `sys.modules`
    and log a warning so users know that cuBLAS will not pick up the setting
    in the current process.
17. `utils.py:env_snapshot`. Used `__import__(mod).__version__` for several
    libraries, which triggers a `FutureWarning` from scanpy and is documented
    as deprecated. Fix: switched to `importlib.metadata.version(dist_name)`
    with a fallback to the module attribute and a `warnings.catch_warnings()`
    guard. Also fixed `sklearn` (import name) vs `scikit-learn` (dist name).
18. `viz.py`. Set `matplotlib.use("Agg")` unconditionally at import time,
    breaking Jupyter use. Fix: only switch to Agg when no `DISPLAY` is set
    and the backend has not already been chosen.

## 3. Quality improvements applied

1. Type hints. Every public function and method now has a complete signature
   annotation. `from __future__ import annotations` is on in every module.
2. Numpy-style docstrings. Every public class and function has `Parameters`,
   `Returns`, `Raises`, and `Notes` sections where applicable, including
   citations to the foundational VQ-VAE papers in `model.py`.
3. Logging. Replaced every `print(...)` in `train.py` and `cli.py` with a
   module-level `logging.getLogger(__name__)`. `__init__.py` attaches a
   `NullHandler` per library convention (transformers, scanpy). The CLI
   wires up `logging.basicConfig` based on `--verbose` / `--quiet`.
4. Actionable error messages. Every `ValueError` and `FileNotFoundError`
   now tells the user what to do next (which column to add, which file to
   provide, which command to run).
5. Reproducibility hooks. `train_model` now writes `train_config.json`
   alongside the checkpoint so reviewers can see the exact hyperparameters,
   and `seed_everything` warns when CUBLAS is set after torch import.
6. CLI: `--version`, `--verbose`/`-v` (count), `--quiet`/`-q`. The action
   table now uses an explicit dispatch dict.
7. `pyproject.toml`: added `Development Status :: 4 - Beta`, OS classifiers,
   `Typing :: Typed`, Python 3.10/3.11/3.12 classifiers; pinned upper bounds
   on all major deps; added `[tool.pytest.ini_options]` with
   `--strict-markers --strict-config` and a filter for the spurious upstream
   `pkg_resources` / `louvain` / matplotlib `Axes3D` warnings; added
   `[tool.ruff]` and `[tool.mypy]` configuration.
8. Packaging: new `src/nicheverse/__main__.py` enables
   `python -m nicheverse ...` as an alias for the CLI.
9. New tests under `tests/`:
   1. `test_io.py` (8 tests): writes a synthetic but bit-realistic Xenium
      output directory (`cell_feature_matrix.h5` in 10x layout + `cells.parquet`),
      exercises single-run and cohort loaders, manifest mode, the
      control-probe filter (both directions), the no-common-genes error, the
      narrow-dtype path in `attach_codes_to_adata`, and a length mismatch
      check.
   2. `test_determinism.py` (2 tests): runs `predict_codes` twice on the same
      input and asserts integer codes are identical and embedding floats are
      bit-equal at `atol=0, rtol=0`. Also asserts `seed_everything` makes
      `torch.randn` reproducible.
   3. `test_cli.py` (6 tests): `--version`, `--help`, end-to-end
      train -> predict -> verify with a JSON reference, `info`, `--quiet`
      suppression of INFO logs, and verify's exit code 2 on mismatch.
10. New docs at the package root:
    1. `CHANGELOG.md` documents every change relative to the original.
    2. `CONTRIBUTING.md` describes setup, test, style, and PR conventions.
    3. `DETERMINISM.md` documents exactly what is and is not bit-reproducible,
       which environment variable must be set before torch import, and how to
       run `nicheverse verify` as the canonical reproducibility check.

## 4. Files touched

Modified:
1. `pyproject.toml` (classifiers, pins, ruff/mypy/pytest config)
2. `src/nicheverse/__init__.py` (NullHandler, sorted re-exports)
3. `src/nicheverse/model.py` (rewritten with full docstrings and the fixes above)
4. `src/nicheverse/data.py` (validation, edge cases, docstring)
5. `src/nicheverse/io.py` (case-insensitive controls, narrow dtypes, validation)
6. `src/nicheverse/train.py` (logging, log-normalize detection, train_config.json)
7. `src/nicheverse/predict.py` (fresh copy, contiguous float32, actionable errors)
8. `src/nicheverse/viz.py` (Agg only when no display)
9. `src/nicheverse/cli.py` (--version, -v/-q, dispatch dict, --report JSON contract)
10. `src/nicheverse/utils.py` (importlib.metadata, CUBLAS warning, sklearn dist name)

Added:
1. `src/nicheverse/__main__.py`
2. `tests/test_io.py`
3. `tests/test_determinism.py`
4. `tests/test_cli.py`
5. `CHANGELOG.md`
6. `CONTRIBUTING.md`
7. `DETERMINISM.md`

No files deleted; no test removed.

## 5. Design decisions deferred to the user

None of the discussed issues required deferral. The two judgment calls below
were applied with backward-compatible defaults; you can revisit them as
release-time choices.

1. `cross_attention_weight = 0.5` retained as default. If you used a different
   value in the manuscript run, set it on `ModelConfig` explicitly; the
   checkpoint will carry it forward.
2. `diversity_temperature = 1.0` retained as default to preserve the exact
   numerical behavior used to train the released checkpoint. For new training
   runs you may want a larger temperature (e.g. `embedding_dim` or
   `sqrt(embedding_dim)`) to recover a usable entropy gradient when
   `cell_embedding_dim` is in the hundreds. This is left as a TrainConfig
   knob to be explored, not changed silently.

## 6. Verification

```
pytest -q tests/
# 23 passed, 2 warnings in 55s
```

The CLI smoke test exercises `python -m nicheverse` end-to-end:

```
python -m nicheverse --version
# nicheverse 0.1.0
```
