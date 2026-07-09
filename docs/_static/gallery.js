// NICHEVERSE explore-collection gallery: filter chips, column toggle, live count.
// Vanilla JS, no dependencies. Safe on pages without the gallery markup.
(function () {
  "use strict";

  function init() {
    var grid = document.getElementById("bb-grid");
    var chipBox = document.getElementById("bb-chips");
    var viewBox = document.getElementById("bb-viewtoggle");
    var countEl = document.getElementById("bb-count");
    if (!grid) return;

    var tiles = Array.prototype.slice.call(grid.querySelectorAll(".bb-tile"));

    // active filter can be "all", a kind (cell/niche), or a site (primary/metastasis/brain)
    var active = { type: "all", value: "all" };

    function matches(tile) {
      if (active.type === "all") return true;
      if (active.type === "kind") return tile.getAttribute("data-kind") === active.value;
      if (active.type === "site") return tile.getAttribute("data-site") === active.value;
      return true;
    }

    function apply() {
      var shown = 0;
      tiles.forEach(function (tile) {
        if (matches(tile)) {
          tile.classList.remove("is-hidden");
          shown += 1;
        } else {
          tile.classList.add("is-hidden");
        }
      });
      if (countEl) {
        countEl.innerHTML = "<b>" + shown + "</b> " + (shown === 1 ? "map" : "maps");
      }
    }

    if (chipBox) {
      chipBox.addEventListener("click", function (e) {
        var btn = e.target.closest(".bb-chip");
        if (!btn) return;
        active = { type: btn.getAttribute("data-type"), value: btn.getAttribute("data-filter") };
        chipBox.querySelectorAll(".bb-chip").forEach(function (c) {
          c.classList.toggle("is-active", c === btn);
        });
        apply();
      });
    }

    if (viewBox) {
      viewBox.addEventListener("click", function (e) {
        var btn = e.target.closest(".bb-view");
        if (!btn) return;
        var cols = btn.getAttribute("data-cols");
        grid.setAttribute("data-cols", cols);
        viewBox.querySelectorAll(".bb-view").forEach(function (v) {
          v.classList.toggle("is-active", v === btn);
        });
      });
    }

    apply();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
