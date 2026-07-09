# Gallery

```{raw} html
<div class="bb-marquee reveal" aria-hidden="true">
  <div class="bb-marquee-track">
    <img src="../_static/gallery/0069490_primary1_core15_cell.png" alt="">
    <img src="../_static/gallery/0069490_primary1_core15_niche.png" alt="">
    <img src="../_static/gallery/0069489_metastases_core14_cell.png" alt="">
    <img src="../_static/gallery/0069489_metastases_core14_niche.png" alt="">
    <img src="../_static/gallery/S20_15120_untreated_cell.png" alt="">
    <img src="../_static/gallery/S20_15120_untreated_niche.png" alt="">
    <img src="../_static/gallery/0069490_primary1_core15_cell.png" alt="">
    <img src="../_static/gallery/0069490_primary1_core15_niche.png" alt="">
    <img src="../_static/gallery/0069489_metastases_core14_cell.png" alt="">
    <img src="../_static/gallery/0069489_metastases_core14_niche.png" alt="">
    <img src="../_static/gallery/S20_15120_untreated_cell.png" alt="">
    <img src="../_static/gallery/S20_15120_untreated_niche.png" alt="">
  </div>
</div>
```

```{raw} html
<div class="bb-explore">

  <div class="bb-explore-header reveal">
    <h2>Explore the collection</h2>
    <p class="bb-explore-lead">Cell-state (<code>cell_codebook_idx</code>, up to 256 codes) and spatial-niche
    (<code>neighborhood_codebook_idx</code>, 32 codes) assignments from the published NICHEVERSE model,
    painted onto segmented nuclear boundaries across renal cell carcinoma contexts. Each nucleus is
    colored by its learned code; neighbors that share a color share a cell state or a spatial niche.</p>
  </div>

  <div class="bb-controls">
    <div class="bb-count" id="bb-count"><b>6</b> maps</div>

    <div class="bb-chips" id="bb-chips" role="group" aria-label="Filter maps">
      <button class="bb-chip is-active" data-filter="all" data-type="all">All</button>
      <button class="bb-chip" data-filter="cell" data-type="kind">Cell states</button>
      <button class="bb-chip" data-filter="niche" data-type="kind">Spatial niches</button>
      <button class="bb-chip" data-filter="primary" data-type="site">Primary</button>
      <button class="bb-chip" data-filter="metastasis" data-type="site">Metastasis</button>
      <button class="bb-chip" data-filter="brain" data-type="site">Brain met</button>
    </div>

    <div class="bb-viewtoggle" id="bb-viewtoggle" role="group" aria-label="Columns">
      <span class="bb-vlabel">View</span>
      <button class="bb-view" data-cols="2" aria-label="Two columns">2</button>
      <button class="bb-view is-active" data-cols="3" aria-label="Three columns">3</button>
      <button class="bb-view" data-cols="4" aria-label="Four columns">4</button>
    </div>
  </div>

  <div class="bb-grid" id="bb-grid" data-cols="3">

    <figure class="bb-tile" data-kind="cell" data-site="primary">
      <span class="bb-frame"><img src="../_static/gallery/0069490_primary1_core15_cell.png" alt="Primary RCC cell states" loading="lazy"></span>
      <figcaption class="bb-caption"><b>Primary RCC · cell states</b><span class="bb-sub">TMA core 15 · nuclei colored by cell-state code</span></figcaption>
    </figure>

    <figure class="bb-tile" data-kind="niche" data-site="primary">
      <span class="bb-frame"><img src="../_static/gallery/0069490_primary1_core15_niche.png" alt="Primary RCC spatial niches" loading="lazy"></span>
      <figcaption class="bb-caption"><b>Primary RCC · spatial niches</b><span class="bb-sub">TMA core 15 · nuclei colored by niche code</span></figcaption>
    </figure>

    <figure class="bb-tile" data-kind="cell" data-site="metastasis">
      <span class="bb-frame"><img src="../_static/gallery/0069489_metastases_core14_cell.png" alt="RCC metastasis cell states" loading="lazy"></span>
      <figcaption class="bb-caption"><b>Metastasis · cell states</b><span class="bb-sub">TMA core 14 · nuclei colored by cell-state code</span></figcaption>
    </figure>

    <figure class="bb-tile" data-kind="niche" data-site="metastasis">
      <span class="bb-frame"><img src="../_static/gallery/0069489_metastases_core14_niche.png" alt="RCC metastasis spatial niches" loading="lazy"></span>
      <figcaption class="bb-caption"><b>Metastasis · spatial niches</b><span class="bb-sub">TMA core 14 · nuclei colored by niche code</span></figcaption>
    </figure>

    <figure class="bb-tile" data-kind="cell" data-site="brain">
      <span class="bb-frame"><img src="../_static/gallery/S20_15120_untreated_cell.png" alt="Brain metastasis cell states" loading="lazy"></span>
      <figcaption class="bb-caption"><b>Brain metastasis · cell states</b><span class="bb-sub">Whole-section resection · nuclei colored by cell-state code</span></figcaption>
    </figure>

    <figure class="bb-tile" data-kind="niche" data-site="brain">
      <span class="bb-frame"><img src="../_static/gallery/S20_15120_untreated_niche.png" alt="Brain metastasis spatial niches" loading="lazy"></span>
      <figcaption class="bb-caption"><b>Brain metastasis · spatial niches</b><span class="bb-sub">Whole-section resection · nuclei colored by niche code</span></figcaption>
    </figure>

  </div>

  <p class="bb-explore-lead" style="margin-top:1.8rem">Every map is produced by loading the production checkpoint,
  running <code>nicheverse.predict_codes</code>, and painting the per-cell codes onto the Xenium
  <code>nucleus_boundaries.parquet</code> polygons for that sample.</p>

</div>
```
