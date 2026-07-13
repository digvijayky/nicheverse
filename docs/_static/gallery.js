// NICHEVERSE gallery, styled after HHMI Beautiful Biology: pure-black, image-forward grid with the
// title below each map, a left sidebar of collapsible checkbox filter groups (Condition / Tissue /
// Platform) with counts, a "N Results" bar with a grid-density toggle, search, and a shared color key.
// Vanilla JS, no dependencies. Safe on pages without the gallery markup.
(function () {
  "use strict";

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function fmtCells(n) {
    if (n == null) return "";
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
    if (n >= 1e3) return Math.round(n / 1e3) + "k";
    return String(n);
  }
  function platformFamily(p) {
    var s = String(p || "").toLowerCase();
    if (s.indexOf("xenium") !== -1) return "Xenium";
    if (s.indexOf("cosmx") !== -1) return "CosMx";
    if (s.indexOf("merfish") !== -1) return "MERFISH";
    if (s.indexOf("seqfish") !== -1) return "seqFISH";
    if (s.indexOf("ribomap") !== -1) return "RIBOmap";
    if (s.indexOf("eel") !== -1) return "EEL-FISH";
    if (s.indexOf("starmap") !== -1) return "STARmap";
    if (s.indexOf("osmfish") !== -1) return "osmFISH";
    return p || "Other";
  }
  function cellsByDataset(data) {
    var seen = {}, total = 0;
    data.forEach(function (d) { if (!seen[d.dataset]) { seen[d.dataset] = 1; total += d.n_cells || 0; } });
    return total;
  }
  function uniq(data, key) { var s = {}; data.forEach(function (d) { if (d[key]) s[d[key]] = 1; }); return Object.keys(s); }

  // cell-state lineage palette (mirrors the plotting PAL) for the shared color key
  var PAL = [
    ["Tumor/Malignant","#FF3B30"],["Epithelial","#F4A582"],["Nephron","#66A61E"],
    ["Islet/Endocrine","#1B9E77"],["Hepatocyte","#B5651D"],["Neuron","#3399FF"],["Glia","#66CCFF"],
    ["T/NK","#00D97E"],["B/Plasma","#B07CE8"],["Myeloid","#FF4FA3"],["Mast/Granulocyte","#FF9500"],
    ["Endothelial","#FFD400"],["Pericyte/Mural","#C99A3B"],["Fibroblast/CAF","#B8925A"],
    ["Erythroid/Blood-prog","#B22222"],["Melanocyte","#DDDDDD"],["Adipocyte","#D8C79A"]
  ];
  function buildKey() {
    var el = document.getElementById("gx-key"); if (!el) return;
    el.innerHTML = '<span class="gx-key-h">Cell-state lineage</span>' + PAL.map(function (c) {
      return '<span class="gx-k"><i style="background:' + c[1] + '"></i>' + esc(c[0]) + "</span>";
    }).join("");
  }

  function card(d) {
    var img = d.thumb
      ? '<img src="../_static/' + esc(d.thumb) + '" alt="' + esc(d.title) + '" loading="lazy">'
      : '<span class="bb-noimg">rendering&hellip;</span>';
    var search = [d.dataset, d.dataset_title, d.sample, d.title, d.site, d.condition,
                  d.organism, d.platform, (d.cell_types || []).join(" ")].join(" ").toLowerCase();
    var sub = (d.sample ? esc(d.dataset_title) : esc(d.site)) + "  ·  " + esc(platformFamily(d.platform));
    return (
      '<figure class="bb-tile" data-category="' + esc(d.category) + '" data-site="' + esc(d.site) +
      '" data-platform="' + esc(platformFamily(d.platform)) + '" data-dataset="' + esc(d.dataset) +
      '" data-search="' + esc(search) + '">' +
      '<span class="bb-frame">' + img + '<span class="bb-tag">' + esc(d.site) + "</span></span>" +
      '<figcaption class="bb-cap"><b>' + esc(d.title) + "</b>" +
      '<span class="bb-cap-sub">' + sub + "</span></figcaption>" +
      "</figure>"
    );
  }

  // left-sidebar filter groups (checkbox, multi-select, with counts)
  var GROUPS = [
    { key: "category", attr: "category", title: "Condition", order: ["Cancer", "Normal", "Developmental", "Disease"], val: function (d) { return d.category; } },
    { key: "site", attr: "site", title: "Tissue", val: function (d) { return d.site; } },
    { key: "platform", attr: "platform", title: "Platform", val: function (d) { return platformFamily(d.platform); } }
  ];

  function updateStats(data) {
    var set = function (id, v) { var el = document.getElementById(id); if (el) el.textContent = v; };
    var nplat = uniq(data.map(function (d) { return { p: platformFamily(d.platform) }; }), "p").length;
    set("gx-tag", data.length + " samples · " + uniq(data, "dataset").length + " datasets · " +
        fmtCells(cellsByDataset(data)) + " cells · " + uniq(data, "site").length + " tissues · " + nplat + " platforms");
  }

  function init() {
    var grid = document.getElementById("bb-grid");
    if (!grid) return;
    var src = grid.getAttribute("data-gallery-src");
    var fgroupsEl = document.getElementById("gx-fgroups");
    var searchEl = document.getElementById("bb-search");
    var viewBox = document.getElementById("bb-viewtoggle");
    var countEl = document.getElementById("bb-count");
    var clearEl = document.getElementById("gx-clear");

    var active = { category: {}, site: {}, platform: {}, q: "" };

    var lb = document.getElementById("gx-lightbox");
    var lbImg = document.getElementById("gx-lb-img");
    var lbCap = document.getElementById("gx-lb-cap");
    var lbClose = document.getElementById("gx-lb-close");
    function openLB(tile) {
      var img = tile.querySelector("img"); if (!img || !lb) return;
      lbImg.src = img.src; lbImg.alt = img.alt || "";
      var t = tile.querySelector(".bb-cap b"); var s = tile.querySelector(".bb-cap-sub");
      lbCap.textContent = (t ? t.textContent : "") + (s ? "  ·  " + s.textContent : "");
      lb.hidden = false; document.body.style.overflow = "hidden";
    }
    function closeLB() { if (!lb) return; lb.hidden = true; lbImg.src = ""; document.body.style.overflow = ""; }
    if (lb) {
      grid.addEventListener("click", function (e) { var tile = e.target.closest(".bb-tile"); if (tile) openLB(tile); });
      lbClose.addEventListener("click", closeLB);
      lb.addEventListener("click", function (e) { if (e.target === lb) closeLB(); });
      document.addEventListener("keydown", function (e) { if (!lb.hidden && e.key === "Escape") closeLB(); });
    }

    function tiles() { return Array.prototype.slice.call(grid.querySelectorAll(".bb-tile")); }
    function anyActive() { return active.q || GROUPS.some(function (g) { return Object.keys(active[g.key]).length; }); }
    function matches(t) {
      for (var i = 0; i < GROUPS.length; i++) {
        var g = GROUPS[i], sel = active[g.key];
        if (Object.keys(sel).length && !sel[t.getAttribute("data-" + g.attr)]) return false;
      }
      if (active.q && t.getAttribute("data-search").indexOf(active.q) === -1) return false;
      return true;
    }
    function apply() {
      var shown = 0;
      tiles().forEach(function (t) { var ok = matches(t); t.classList.toggle("is-hidden", !ok); if (ok) shown += 1; });
      if (countEl) countEl.innerHTML = "<b>" + shown + "</b> " + (shown === 1 ? "result" : "results");
      if (clearEl) clearEl.style.display = anyActive() ? "" : "none";
    }

    function buildGroups(data) {
      if (!fgroupsEl) return;
      fgroupsEl.innerHTML = GROUPS.map(function (g) {
        var counts = {};
        data.forEach(function (d) { var v = g.val(d); if (v) counts[v] = (counts[v] || 0) + 1; });
        var vals = Object.keys(counts);
        if (g.order) vals.sort(function (a, b) { var ia = g.order.indexOf(a), ib = g.order.indexOf(b); return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib); });
        else vals.sort(function (a, b) { return counts[b] - counts[a]; });
        var opts = vals.map(function (v) {
          return '<label class="gx-fopt"><input type="checkbox" data-group="' + g.key + '" value="' + esc(v) + '">' +
                 '<span class="gx-fname">' + esc(v) + '</span><span class="gx-fcount">' + counts[v] + "</span></label>";
        }).join("");
        return '<section class="gx-fgroup"><button class="gx-fhead" type="button">' + esc(g.title) +
               '<span class="gx-fchevron">&minus;</span></button><div class="gx-fbody">' + opts + "</div></section>";
      }).join("");
      fgroupsEl.addEventListener("change", function (e) {
        var cb = e.target; if (cb.type !== "checkbox") return;
        var g = cb.getAttribute("data-group");
        if (cb.checked) active[g][cb.value] = 1; else delete active[g][cb.value];
        apply();
      });
      fgroupsEl.addEventListener("click", function (e) {
        var h = e.target.closest(".gx-fhead"); if (!h) return;
        var grp = h.parentNode; grp.classList.toggle("is-collapsed");
        h.querySelector(".gx-fchevron").innerHTML = grp.classList.contains("is-collapsed") ? "+" : "&minus;";
      });
    }

    function wire() {
      if (searchEl) searchEl.addEventListener("input", function () { active.q = searchEl.value.trim().toLowerCase(); apply(); });
      if (viewBox) viewBox.addEventListener("click", function (e) {
        var b = e.target.closest(".bb-view"); if (!b) return;
        grid.setAttribute("data-cols", b.getAttribute("data-cols"));
        viewBox.querySelectorAll(".bb-view").forEach(function (v) { v.classList.toggle("is-active", v === b); });
      });
      if (clearEl) clearEl.addEventListener("click", function () {
        active = { category: {}, site: {}, platform: {}, q: "" };
        if (searchEl) searchEl.value = "";
        if (fgroupsEl) fgroupsEl.querySelectorAll("input:checked").forEach(function (c) { c.checked = false; });
        apply();
      });
    }

    function render(data) {
      grid.innerHTML = data.map(function (d) { return card(d); }).join("");
      // smooth fade-in as each map loads (incl. cached images)
      Array.prototype.forEach.call(grid.querySelectorAll("img"), function (im) {
        if (im.complete && im.naturalWidth) im.classList.add("is-loaded");
        else {
          im.addEventListener("load", function () { im.classList.add("is-loaded"); }, { once: true });
          im.addEventListener("error", function () { im.classList.add("is-loaded"); }, { once: true });  // never leave a tile invisible
        }
      });
      updateStats(data); buildKey(); buildGroups(data); wire(); apply();
    }

    // prefer the embedded data global (works when opened locally, file://); fall back to fetch when served
    if (window.NV_GALLERY_DATA) render(window.NV_GALLERY_DATA);
    else if (src) fetch(src).then(function (r) { return r.json(); }).then(render).catch(function () { wire(); apply(); });
    else { wire(); apply(); }
  }

  // homepage live counter
  function initHomeStats() {
    var box = document.getElementById("nv-homestats");
    if (!box) return;
    var src = box.getAttribute("data-src");
    var apply = function (data) {
      var set = function (id, v) { var el = document.getElementById(id); if (el) el.textContent = v; };
      var nplat = uniq(data.map(function (d) { return { p: platformFamily(d.platform) }; }), "p").length;
      set("nv-hs-datasets", uniq(data, "dataset").length);
      set("nv-hs-samples", data.length.toLocaleString());
      set("nv-hs-cells", fmtCells(cellsByDataset(data)));
      set("nv-hs-sites", uniq(data, "site").length);
      set("nv-hs-plat", nplat);
    };
    if (window.NV_GALLERY_DATA) apply(window.NV_GALLERY_DATA);
    else if (src) fetch(src).then(function (r) { return r.json(); }).then(apply).catch(function () {});
  }

  function boot() { init(); initHomeStats(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
