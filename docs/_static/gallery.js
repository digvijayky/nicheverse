// NICHEVERSE gallery: ONE CARD PER INDEPENDENT SAMPLE from gallery_data.json, with category,
// site, platform, and dataset filters, free-text search, column toggle, and live counts.
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

  // collapse the many platform strings into a filterable family (Xenium / CosMx / MERFISH / ...)
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

  // cells are a DATASET property; sum once per dataset so per-sample cards do not multiply the total
  function cellsByDataset(data) {
    var seen = {}, total = 0;
    data.forEach(function (d) { if (!seen[d.dataset]) { seen[d.dataset] = 1; total += d.n_cells || 0; } });
    return total;
  }
  function uniq(data, key) { var s = {}; data.forEach(function (d) { if (d[key]) s[d[key]] = 1; }); return Object.keys(s); }

  function plateNo(idx) { var s = String(idx + 1); while (s.length < 3) s = "0" + s; return "PLATE " + s; }

  function card(d, idx) {
    var fam = platformFamily(d.platform);
    var badges =
      '<span class="bb-badge bb-cat-' + esc(d.category) + '">' + esc(d.category) + "</span>" +
      '<span class="bb-badge">' + esc(d.site) + "</span>" +
      '<span class="bb-badge">' + esc(fam) + "</span>";
    var img = d.thumb
      ? '<img src="../_static/' + esc(d.thumb) + '" alt="' + esc(d.title) + '" loading="lazy">'
      : '<span class="bb-noimg">map rendering&hellip;</span>';
    var search = [d.dataset, d.dataset_title, d.sample, d.title, d.site, d.condition,
                  d.organism, d.platform, (d.cell_types || []).join(" ")].join(" ").toLowerCase();
    // multi-sample datasets show the dataset name as an eyebrow above the sample name
    var eyebrow = d.sample ? '<span class="bb-ds">' + esc(d.dataset_title) + "</span>" : "";
    return (
      '<figure class="bb-tile" data-category="' + esc(d.category) + '" data-site="' + esc(d.site) +
      '" data-platform="' + esc(fam) + '" data-dataset="' + esc(d.dataset) +
      '" data-search="' + esc(search) + '">' +
      '<span class="bb-frame"><span class="bb-plate">' + plateNo(idx) + "</span>" + img + "</span>" +
      '<figcaption class="bb-caption">' +
      '<span class="bb-badges">' + badges + "</span>" +
      eyebrow +
      "<b>" + esc(d.title) + "</b>" +
      '<span class="bb-sub">' + esc(d.condition || d.organism || "") + "</span>" +
      "</figcaption></figure>"
    );
  }

  function updateStats(data) {
    var set = function (id, v) { var el = document.getElementById(id); if (el) el.textContent = v; };
    set("gx-n-samples", data.length);
    set("gx-n-datasets", uniq(data, "dataset").length);
    set("gx-n-sites", uniq(data, "site").length);
    set("gx-n-plat", (function () { var s = {}; data.forEach(function (d) { s[platformFamily(d.platform)] = 1; }); return Object.keys(s).length; })());
    set("gx-n-cells", fmtCells(cellsByDataset(data)));
  }

  function init() {
    var grid = document.getElementById("bb-grid");
    if (!grid) return;
    var src = grid.getAttribute("data-gallery-src");
    var chipBox = document.getElementById("bb-chips");
    var siteSel = document.getElementById("bb-site");
    var platSel = document.getElementById("bb-platform");
    var dsSel = document.getElementById("bb-dataset");
    var searchEl = document.getElementById("bb-search");
    var viewBox = document.getElementById("bb-viewtoggle");
    var countEl = document.getElementById("bb-count");

    var active = { category: "all", site: "all", platform: "all", dataset: "all", q: "" };

    // lightbox: click a plate to open its map large
    var lb = document.getElementById("gx-lightbox");
    var lbImg = document.getElementById("gx-lb-img");
    var lbCap = document.getElementById("gx-lb-cap");
    var lbClose = document.getElementById("gx-lb-close");
    function openLB(tile) {
      var img = tile.querySelector("img"); if (!img || !lb) return;
      lbImg.src = img.src; lbImg.alt = img.alt || "";
      var ds = tile.querySelector(".bb-ds"); var t = tile.querySelector(".bb-caption b");
      lbCap.textContent = (ds ? ds.textContent + "  ·  " : "") + (t ? t.textContent : "");
      lb.hidden = false; document.body.style.overflow = "hidden";
    }
    function closeLB() { if (!lb) return; lb.hidden = true; lbImg.src = ""; document.body.style.overflow = ""; }
    if (lb) {
      grid.addEventListener("click", function (e) {
        var tile = e.target.closest(".bb-tile"); if (tile) openLB(tile);
      });
      lbClose.addEventListener("click", closeLB);
      lb.addEventListener("click", function (e) { if (e.target === lb) closeLB(); });
      document.addEventListener("keydown", function (e) { if (!lb.hidden && e.key === "Escape") closeLB(); });
    }

    function tiles() { return Array.prototype.slice.call(grid.querySelectorAll(".bb-tile")); }

    function matches(t) {
      if (active.category !== "all" && t.getAttribute("data-category") !== active.category) return false;
      if (active.site !== "all" && t.getAttribute("data-site") !== active.site) return false;
      if (active.platform !== "all" && t.getAttribute("data-platform") !== active.platform) return false;
      if (active.dataset !== "all" && t.getAttribute("data-dataset") !== active.dataset) return false;
      if (active.q && t.getAttribute("data-search").indexOf(active.q) === -1) return false;
      return true;
    }

    function apply() {
      var shown = 0;
      tiles().forEach(function (t) {
        var ok = matches(t);
        t.classList.toggle("is-hidden", !ok);
        if (ok) shown += 1;
      });
      if (countEl) countEl.innerHTML = "<b>" + shown + "</b> " + (shown === 1 ? "sample" : "samples");
    }

    function wire() {
      if (chipBox) {
        chipBox.addEventListener("click", function (e) {
          var b = e.target.closest(".bb-chip"); if (!b) return;
          active.category = b.getAttribute("data-filter");
          chipBox.querySelectorAll(".bb-chip").forEach(function (c) { c.classList.toggle("is-active", c === b); });
          apply();
        });
      }
      if (siteSel) siteSel.addEventListener("change", function () { active.site = siteSel.value; apply(); });
      if (platSel) platSel.addEventListener("change", function () { active.platform = platSel.value; apply(); });
      if (dsSel) dsSel.addEventListener("change", function () { active.dataset = dsSel.value; apply(); });
      if (searchEl) searchEl.addEventListener("input", function () { active.q = searchEl.value.trim().toLowerCase(); apply(); });
      if (viewBox) {
        viewBox.addEventListener("click", function (e) {
          var b = e.target.closest(".bb-view"); if (!b) return;
          grid.setAttribute("data-cols", b.getAttribute("data-cols"));
          viewBox.querySelectorAll(".bb-view").forEach(function (v) { v.classList.toggle("is-active", v === b); });
        });
      }
    }

    function fillSelect(sel, values, allLabel) {
      if (!sel) return;
      sel.innerHTML = '<option value="all">' + allLabel + "</option>" +
        values.map(function (v) { return '<option value="' + esc(v) + '">' + esc(v) + "</option>"; }).join("");
    }

    function render(data) {
      grid.innerHTML = data.map(function (d, i) { return card(d, i); }).join("");
      updateStats(data);
      var sites = uniq(data, "site").sort();
      fillSelect(siteSel, sites, "All tissues");
      var plats = {}; data.forEach(function (d) { plats[platformFamily(d.platform)] = 1; });
      fillSelect(platSel, Object.keys(plats).sort(), "All platforms");
      // dataset dropdown: label by dataset_title, value by dataset id
      var dsSeen = {}, dsOpts = [];
      data.forEach(function (d) { if (!dsSeen[d.dataset]) { dsSeen[d.dataset] = 1; dsOpts.push([d.dataset, d.dataset_title]); } });
      dsOpts.sort(function (a, b) { return a[1] < b[1] ? -1 : 1; });
      if (dsSel) dsSel.innerHTML = '<option value="all">All datasets</option>' +
        dsOpts.map(function (o) { return '<option value="' + esc(o[0]) + '">' + esc(o[1]) + "</option>"; }).join("");
      wire();
      apply();
    }

    if (src) {
      fetch(src).then(function (r) { return r.json(); }).then(render).catch(function () { wire(); apply(); });
    } else {
      wire(); apply();  // static markup fallback
    }
  }

  // homepage live counter: samples / datasets / cells / tissues / platforms from the catalogue
  function initHomeStats() {
    var box = document.getElementById("nv-homestats");
    if (!box) return;
    var src = box.getAttribute("data-src");
    if (!src) return;
    fetch(src).then(function (r) { return r.json(); }).then(function (data) {
      var set = function (id, v) { var el = document.getElementById(id); if (el) el.textContent = v; };
      set("nv-hs-datasets", uniq(data, "dataset").length);
      set("nv-hs-samples", data.length.toLocaleString());
      set("nv-hs-cells", fmtCells(cellsByDataset(data)));
      set("nv-hs-sites", uniq(data, "site").length);
      set("nv-hs-plat", (function () { var s = {}; data.forEach(function (d) { s[platformFamily(d.platform)] = 1; }); return Object.keys(s).length; })());
    }).catch(function () {});
  }

  function boot() { init(); initHomeStats(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
