// NICHEVERSE scroll-reveal motion (Beautiful Biology feel).
// Elements with class "reveal" fade + rise into place as they enter the
// viewport, staggered among siblings. Vanilla JS, no dependencies, idempotent,
// and a no-op when nothing to reveal. Respects prefers-reduced-motion.
(function () {
  "use strict";

  function init() {
    var nodes = Array.prototype.slice.call(document.querySelectorAll(".reveal"));
    if (!nodes.length) return;

    var reduce =
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // Reduced motion or no IntersectionObserver: just show everything.
    if (reduce || typeof IntersectionObserver === "undefined") {
      nodes.forEach(function (el) {
        el.classList.add("in");
      });
      return;
    }

    var observer = new IntersectionObserver(
      function (entries, obs) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var el = entry.target;
          // Stagger siblings by ~80ms based on position among reveal siblings.
          var sibs = el.parentNode
            ? Array.prototype.slice.call(el.parentNode.children).filter(function (c) {
                return c.classList && c.classList.contains("reveal");
              })
            : [el];
          var idx = Math.max(0, sibs.indexOf(el));
          el.style.transitionDelay = idx * 80 + "ms";
          el.classList.add("in");
          obs.unobserve(el);
        });
      },
      { threshold: 0.12 }
    );

    nodes.forEach(function (el) {
      if (el.classList.contains("in")) return; // idempotent
      observer.observe(el);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
