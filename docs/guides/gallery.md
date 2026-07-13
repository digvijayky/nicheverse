---
sd_hide_title: true
html_theme.sidebar_secondary.remove: true
---

# Gallery

```{raw} html
<div class="gx">
 <div class="gx-inner">

  <header class="gx-head reveal">
    <h1 class="gx-serif">Explore the atlas</h1>
    <p class="gx-tagline" id="gx-tag">One frozen model, read across every tissue.</p>
  </header>

  <div class="gx-body">

    <aside class="gx-side">
      <div class="gx-side-inner">
        <input class="bb-search" id="bb-search" type="search" placeholder="Search a site, sample, or cell type&hellip;" aria-label="Search samples">
        <button class="gx-clear" id="gx-clear" type="button" style="display:none">Clear all filters</button>
        <div class="gx-fgroups" id="gx-fgroups"></div>
        <div class="gx-key" id="gx-key" aria-label="Cell-state lineage color key"></div>
      </div>
    </aside>

    <div class="gx-main">
      <div class="gx-resbar">
        <span class="bb-count" id="bb-count"><b>&hellip;</b> results</span>
        <div class="bb-viewtoggle" id="bb-viewtoggle" role="group" aria-label="Grid density">
          <button class="bb-view" data-cols="2" title="Large"><span></span><span></span></button>
          <button class="bb-view is-active" data-cols="3" title="Medium"><span></span><span></span><span></span></button>
          <button class="bb-view" data-cols="4" title="Small"><span></span><span></span><span></span><span></span></button>
        </div>
      </div>

      <div class="bb-grid" id="bb-grid" data-cols="3" data-gallery-src="../_static/gallery_data.json"></div>

      <p class="gx-method">Each map is produced by loading the released checkpoint, running
        <code>nicheverse.predict_codes</code> on the dataset, annotating every cell-state code with a
        literature-grounded cell type, and coloring each cell by the lineage of its code. The catalogue and the
        full-resolution vector maps regenerate automatically as datasets are added.</p>
    </div>

  </div>
 </div>

 <div class="gx-lightbox" id="gx-lightbox" hidden aria-modal="true" role="dialog">
   <button class="gx-lb-close" id="gx-lb-close" aria-label="Close preview">&times;</button>
   <figure class="gx-lb-fig">
     <img id="gx-lb-img" src="" alt="">
     <figcaption id="gx-lb-cap"></figcaption>
   </figure>
 </div>

</div>
```
