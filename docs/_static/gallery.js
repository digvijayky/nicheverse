// NICHEVERSE gallery: data-driven cards from gallery_data.json with category filter,
// site dropdown, free-text search (site + cell type), column toggle, live count.
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

  function plateNo(idx) { var s = String(idx + 1); while (s.length < 3) s = "0" + s; return "PLATE " + s; }

  function card(d, idx) {
    var badges =
      '<span class="bb-badge bb-cat-' + esc(d.category) + '">' + esc(d.category) + "</span>" +
      '<span class="bb-badge">' + esc(d.site) + "</span>" +
      '<span class="bb-badge">' + esc(platformFamily(d.platform)) + "</span>";
    var stat = [fmtCells(d.n_cells) + " cells",
                (d.n_celltypes ? d.n_celltypes + " cell types" : ""),
                (d.n_samples > 1 ? d.n_samples + " samples" : "")]
      .filter(Boolean).join("  ·  ");
    var img = d.thumb
      ? '<img src="../_static/' + esc(d.thumb) + '" alt="' + esc(d.title) + '" loading="lazy">'
      : '<span class="bb-noimg">map rendering&hellip;</span>';
    var search = [d.id, d.title, d.site, d.condition, d.organism, d.platform,
                  (d.cell_types || []).join(" ")].join(" ").toLowerCase();
    return (
      '<figure class="bb-tile" data-category="' + esc(d.category) + '" data-site="' + esc(d.site) +
      '" data-platform="' + esc(platformFamily(d.platform)) + '" data-search="' + esc(search) + '">' +
      '<span class="bb-frame"><span class="bb-plate">' + plateNo(idx) + "</span>" + img + "</span>" +
      '<figcaption class="bb-caption">' +
      '<span class="bb-badges">' + badges + "</span>" +
      "<b>" + esc(d.title) + "</b>" +
      '<span class="bb-sub">' + esc(d.condition || d.organism || "") + "</span>" +
      '<span class="bb-stat">' + esc(stat) + "</span>" +
      "</figcaption></figure>"
    );
  }

  function updateStats(data) {
    var set = function (id, v) { var el = document.getElementById(id); if (el) el.textContent = v; };
    var sites = {}, plats = {}, cells = 0;
    data.forEach(function (d) {
      if (d.site) sites[d.site] = 1;
      plats[platformFamily(d.platform)] = 1;
      cells += d.n_cells || 0;
    });
    set("gx-n-datasets", data.length);
    set("gx-n-sites", Object.keys(sites).length);
    set("gx-n-plat", Object.keys(plats).length);
    set("gx-n-cells", fmtCells(cells));
  }

  function init() {
    var grid = document.getElementById("bb-grid");
    if (!grid) return;
    var src = grid.getAttribute("data-gallery-src");
    var chipBox = document.getElementById("bb-chips");
    var siteSel = document.getElementById("bb-site");
    var platSel = document.getElementById("bb-platform");
    var searchEl = document.getElementById("bb-search");
    var viewBox = document.getElementById("bb-viewtoggle");
    var countEl = document.getElementById("bb-count");

    var active = { category: "all", site: "all", platform: "all", q: "" };

    // lightbox: click a plate to open its map large
    var lb = document.getElementById("gx-lightbox");
    var lbImg = document.getElementById("gx-lb-img");
    var lbCap = document.getElementById("gx-lb-cap");
    var lbClose = document.getElementById("gx-lb-close");
    function openLB(tile) {
      var img = tile.querySelector("img"); if (!img || !lb) return;
      lbImg.src = img.src; lbImg.alt = img.alt || "";
      var t = tile.querySelector(".bb-caption b"); var s = tile.querySelector(".bb-stat");
      lbCap.textContent = (t ? t.textContent : "") + (s ? "  ·  " + s.textContent : "");
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
      if (countEl) countEl.innerHTML = "<b>" + shown + "</b> " + (shown === 1 ? "plate" : "plates");
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
      if (searchEl) searchEl.addEventListener("input", function () { active.q = searchEl.value.trim().toLowerCase(); apply(); });
      if (viewBox) {
        viewBox.addEventListener("click", function (e) {
          var b = e.target.closest(".bb-view"); if (!b) return;
          grid.setAttribute("data-cols", b.getAttribute("data-cols"));
          viewBox.querySelectorAll(".bb-view").forEach(function (v) { v.classList.toggle("is-active", v === b); });
        });
      }
    }

    function render(data) {
      grid.innerHTML = data.map(function (d, i) { return card(d, i); }).join("");
      updateStats(data);
      if (siteSel) {
        var seen = {}, sites = [];
        data.forEach(function (d) { if (!seen[d.site]) { seen[d.site] = 1; sites.push(d.site); } });
        sites.sort();
        siteSel.innerHTML = '<option value="all">All sites</option>' +
          sites.map(function (s) { return '<option value="' + esc(s) + '">' + esc(s) + "</option>"; }).join("");
      }
      if (platSel) {
        var pseen = {}, plats = [];
        data.forEach(function (d) { var f = platformFamily(d.platform); if (!pseen[f]) { pseen[f] = 1; plats.push(f); } });
        plats.sort();
        platSel.innerHTML = '<option value="all">All platforms</option>' +
          plats.map(function (p) { return '<option value="' + esc(p) + '">' + esc(p) + "</option>"; }).join("");
      }
      wire();
      apply();
    }

    if (src) {
      fetch(src).then(function (r) { return r.json(); }).then(render).catch(function () { wire(); apply(); });
    } else {
      wire(); apply();  // static markup fallback
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
