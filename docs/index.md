---
sd_hide_title: true
html_theme.sidebar_secondary.remove: true
---

# NICHEVERSE

```{raw} html
<div class="nvh">
  <section class="nvh-hero">
    <div class="nvh-hero-copy">
      <span class="nvh-eyebrow">A world model for tissues</span>
      <h1 class="nvh-title">Read every tissue as a world&nbsp;model.</h1>
      <p class="nvh-lead">The released NICHEVERSE checkpoint, frozen and never retrained, learns paired discrete codebooks of recurrent cell states and multicellular spatial niches from imaging-based spatial transcriptomics, and reads any cohort reproducibly.</p>
      <div class="nvh-cta">
        <a class="nvh-btn" href="guides/gallery.html">Explore the atlas</a>
        <a class="nvh-btn nvh-btn--ghost" href="guides/quickstart.html">Get started</a>
      </div>
      <dl class="nv-homestats" id="nv-homestats" data-src="_static/gallery_data.json" aria-label="Atlas totals">
        <div class="nv-hs"><dd id="nv-hs-samples">381</dd><dt>samples</dt></div>
        <div class="nv-hs"><dd id="nv-hs-datasets">102</dd><dt>datasets</dt></div>
        <div class="nv-hs"><dd id="nv-hs-cells">39M</dd><dt>cells mapped</dt></div>
        <div class="nv-hs"><dd id="nv-hs-sites">20</dd><dt>tissues</dt></div>
        <div class="nv-hs"><dd id="nv-hs-plat">6</dd><dt>platforms</dt></div>
      </dl>
    </div>
    <figure class="nvh-hero-shot">
      <img src="_static/gallery/sweep/cosmx_wtx_brain__cosmx_wtx_brain.png" alt="A whole brain section read into neuronal and glial cell-state lineages">
    </figure>
  </section>

  <section class="nvh-explore">
    <div class="nvh-circles" aria-hidden="true">
      <img src="_static/gallery/sweep/rcc_brm_xenium__0069489_metastases_core3.png" loading="lazy" alt="">
      <img src="_static/gallery/sweep/cosmx_pancreas__cosmx_pancreas.png" loading="lazy" alt="">
      <img src="_static/gallery/sweep/pub_tonsil__pub_tonsil.png" loading="lazy" alt="">
      <img src="_static/gallery/sweep/cosmx_multiomic_breast_rna__cosmx_multiomic_breast_rna.png" loading="lazy" alt="">
      <img src="_static/gallery/sweep/pub_ovary_ff__pub_ovary_ff.png" loading="lazy" alt="">
      <img src="_static/gallery/sweep/xenium_dev_kidney_w16__xenium_dev_kidney_w16.png" loading="lazy" alt="">
      <img src="_static/gallery/sweep/pub_colon_cancer_base__pub_colon_cancer_base.png" loading="lazy" alt="">
      <img src="_static/gallery/sweep/merfish_hcc_1003__merfish_hcc_1003.png" loading="lazy" alt="">
      <img src="_static/gallery/sweep/pub_lung_5k__pub_lung_5k.png" loading="lazy" alt="">
    </div>
    <div class="nvh-explore-copy">
      <h2 class="nvh-h2">Explore the atlas</h2>
      <p class="nvh-p">One frozen checkpoint, read across <b>381 independent samples</b> from 102 datasets and every accessible platform, Xenium, CosMx, MERFISH, seqFISH, RIBOmap, EEL-FISH. Every cell is painted by the lineage of the cell-state code the model assigns it.</p>
      <a class="nvh-btn" href="guides/gallery.html">Browse all samples</a>
    </div>
  </section>
</div>
```

:::::{div} nvh-body

::::{grid} 1 2 2 3
:gutter: 3
:class-container: nv-features reveal

:::{grid-item-card} Hierarchical codebooks
Paired cell-state and spatial-niche codebooks coupled by cross-attention, so cell identity is read in tissue context.
:::
:::{grid-item-card} Swappable components
Encoder registry: `mlp_deep` (default) / `mlp` / `mlp_plr` / `residual_mlp` / `transformer` / `cnn` / `fast_cnn` / `deep_cnn` / `gnn` / `diffusion` / `dit` / `set_transformer` / `perceiver_io` / `soft_moe` / `ft_transformer`. Quantizer registry: `vq` (default) / `rvq` / `grvq` / `pq` / `qinco` / `rot` / `soft` / `bsq` / `lfq` / `fsq` / `residual_fsq`.
:::
:::{grid-item-card} Spatial-aware
Per-sample graphs (`knn`, `knn_radius` default at 50 microns, `radius`, `delaunay`, `alpha_complex`, `gabriel`, `rng`), inverse-distance aggregation, and opt-in spatial-coherence losses.
:::
:::{grid-item-card} Count-native
Default negative-binomial cell likelihood with a detection hurdle and a Dirichlet-multinomial niche composition term on raw counts, plus transcript-level subcellular context.
:::
:::{grid-item-card} Reproducible
Byte-exact reproduction of the published renal cell carcinoma and brain-metastasis model, guarded by a regression test.
:::
:::{grid-item-card} PyTorch-native
A `Trainer`, checkpoints, mixed precision, warmup-cosine scheduling, and MAE pretraining.
:::

::::

{.reveal}
## Install

```bash
pip install nicheverse       # or: conda install -c conda-forge nicheverse
```

{.reveal}
## Quickstart

```python
import nicheverse as nv

adata = nv.read_xenium_cohort(["./run_A", "./run_B"])
mc = nv.ModelConfig(input_dim=adata.n_vars, gene_names=tuple(adata.var_names))
model, adata = nv.Trainer(nv.TrainConfig(num_epochs=300)).fit(adata, "./ckpt", model_config=mc)

annotated = nv.predict_codes(
    nv.read_xenium_cohort(["./run_C"]), "./ckpt/hierarchical_vqvae_checkpoint.pt"
)
```

:::::

```{toctree}
:hidden:
:maxdepth: 1
:caption: Getting started

guides/installation
guides/gallery
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: API

api
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Guides

guides/quickstart
guides/method
guides/io
guides/hyperparameters
guides/annotation
guides/mcp
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reproducibility

reproducibility/bundle
reproducibility/reviewer_recipe
reproducibility/determinism
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project

changelog
contributing
```
