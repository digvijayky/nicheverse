# Choosing hyperparameters

A short field guide. Defaults are tuned for a 300 to 500 gene Xenium panel on a multi sample human cohort.

## Cell codebook size K_c

K_c is the number of cell states the model can carry. Larger K_c gives finer resolution but harder annotation.

```
panel size      cells       K_c suggestion
100 to 200      < 500K      64 to 128
300 to 400      500K to 5M  128 to 256
500 plus        5M plus     256 to 512
```

If perplexity at the end of training is close to K_c, the model is using all of capacity. If it is below half of K_c, the dead code reset is keeping many codes alive but they are nearly redundant; consider lowering K_c. We aim for 70 to 90% of K_c utilized after 300 epochs.

## Cell embedding dim d_c

`d_c = 64` is plenty for panels up to 500 genes. Going higher rarely helps and slows training. d_c should not exceed the rank of cell type variation in your data; for a small targeted panel d_c = 32 is fine.

## Neighborhood codebook size K_n

Tissue niches are coarser than cell states. We use K_n = 32 for tumor cohorts (RCC, BrM) where a handful of immune neighborhoods, stromal neighborhoods, and tumor neighborhoods cover the space. For developmental atlases or pan tissue work, K_n = 16 may be enough.

## Neighborhood embedding dim d_n

Larger d_n carries more spatial detail. We use d_n = 256 for K_n = 32 (8 dimensions per code). Going below d_n = 64 collapses niches.

## k_neighbors

The radius of "neighborhood". 20 is a good default for Xenium cell densities (one neighbor per 30 to 50 microns); a typical 20 nearest gives roughly an effective radius of 150 microns. For sparser tissues use 30 to 50. For dense epithelia use 10 to 15.

## Spatial graph

`knn_radius` (default) is a k-nearest-neighbor graph whose edges are additionally capped at `radius` microns (default 50), so it keeps a dense local neighborhood in normal tissue but never bridges long gaps in sparse metastasis. The graph is always built per sample: cells from different samples never link, which prevents cross-contamination when slides share a coordinate frame. `knn` (pure k-NN, no cap), `radius`, `delaunay`, and `alpha_complex` are the alternatives.

## Aggregation

`weighted_mean` (inverse distance) is the default and is biologically most defensible: closer cells contribute more to the niche feature. `mean` is simpler and works almost as well. `max` is brittle and we do not recommend it.

## Encoder

`mlp_deep` (default) is a SwiGLU pre-norm residual MLP with no per-gene numerical embedding. On sparse Xenium counts it gives the healthiest raw codebook. Per-gene numerical embeddings (`mlp_plr` / PLE) degenerate on sparse counts, so prefer the plain MLP family (`mlp`, `mlp_deep`) there. Attention and diffusion backbones (`dit`, `diffusion`) collapse the codebook unless they use permutation-invariant set pooling (`set_transformer`, `perceiver_io`).

## Reconstruction loss

The cell branch defaults to `cell_recon="nb"`: a negative-binomial NLL on the raw counts (scVI-style library from the observed total count) plus a Bernoulli detection hurdle weighted by `detection_weight` (default 0.5). The niche branch defaults to `niche_recon="mse_dirmult"`: composition MSE plus a Dirichlet-multinomial on the count-scale aggregated composition. These count-native defaults require raw INTEGER counts in `adata.X`; a normalized or log matrix raises. To recover the pure-MSE path set `cell_recon="mse"` (with `detection_weight=0`) and `niche_recon="mse"`.

## Batch size

2048 is a sweet spot for the EMA codebook update on a typical GPU (24 to 80 GB). With batch size below 256 the dead code reset triggers too often and codes become noisy; above 8192 you risk OOM on the cross attention. Halve the batch if you OOM. Setting `batch_size="auto"` resolves the batch from the panel size at train time and, by default, scales the learning rate with it.

## Learning rate

3e-4 with AdamW and the `ReduceLROnPlateau` scheduler is robust. AdamW uses decoupled selective weight decay (`weight_decay=0.01`, applied only to Linear and Conv weights; biases, normalization parameters, and bare parameters are excluded). Higher rates (1e-3) train faster but more often collapse the codebook. Lower rates (1e-4) sometimes help on small cohorts (under 100K cells).

## Speed knobs

All are opt-in and accuracy-neutral. `TrainConfig.device_resident=True` moves the feature tensors onto the GPU once and gathers batches there, cutting the per-batch host-to-device copy for a roughly 3 to 15x speedup; it is memory-fit-gated and falls back cleanly to the CPU DataLoader path when the resident tensors would not fit. Per-run timing (total wall time, mean epoch seconds, cells per second, peak GPU memory) is written to `training_runtime.json`.

## Number of epochs

300 is a safe default. The cross attention layer typically converges by epoch 100; the codebook entropy stabilizes by epoch 200. Watch `training_losses.json`: if total loss has plateaued for 50 epochs the model is done. Less than 100 epochs almost always undertrains.

## Commitment cost

0.25 is the standard VQ-VAE value. Higher (0.5) snaps encoder outputs more tightly to the codebook at the cost of reconstruction quality. We have not needed to change this.

## Diversity weight

1.0 by default. If you see a few codes capturing 80% of cells, raise to 2.0. If usage is already uniform, you can drop to 0.5 to let the codebook differentiate freely.

## Cross attention on or off

On by default. Turn off via `--no-cross-attention` if you want a "pure" cell codebook that ignores spatial context for ablation comparisons. In our experience, off mode produces codes that disagree more with manual cell type labels because borderline cells (for example, low ploidy tumor adjacent to stroma) get assigned by transcripts alone with no smoothing.

## Random seed

Default 49. Different seeds produce different codebooks because k-means++ initialization is stochastic. To explore robustness, train three independent seeds and check that hierarchical clustering of the resulting codebooks recovers the same coarse structure.

## Quick recipe table

```
Cohort size      K_c   d_c   K_n   d_n   k_neigh   epochs   batch
< 100K           64    32    16    64    15        200      512
100K to 1M       128   64    16    128   20        300      1024
1M to 5M         256   64    32    256   20        300      2048
> 5M             256   64    32    256   20        300      2048
```
