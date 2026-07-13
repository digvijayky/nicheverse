// NICHEVERSE unique-visitor analytics — privacy-friendly, cookieless. Off until you add a token.
// Pick ONE and paste its value below, then rebuild the docs and deploy:
//   1) Cloudflare Web Analytics (free; best if nicheverse.org is on Cloudflare):
//      dash.cloudflare.com -> Analytics & Logs -> Web Analytics -> Add a site -> copy the JS "token".
//   2) GoatCounter (free, open source, gives a live dashboard + optional public counter):
//      goatcounter.com -> create "nicheverse" site -> use https://nicheverse.goatcounter.com/count
// Both report unique visitors without cookies. Leave both blank to disable.
(function () {
  "use strict";
  var CF_TOKEN = "";      // e.g. "abc123def456..."
  var GOATCOUNTER = "";   // e.g. "https://nicheverse.goatcounter.com/count"
  if (CF_TOKEN) {
    var s = document.createElement("script");
    s.defer = true;
    s.src = "https://static.cloudflareinsights.com/beacon.min.js";
    s.setAttribute("data-cf-beacon", JSON.stringify({ token: CF_TOKEN }));
    document.head.appendChild(s);
  }
  if (GOATCOUNTER) {
    var g = document.createElement("script");
    g.async = true;
    g.src = "//gc.zgo.at/count.js";
    g.setAttribute("data-goatcounter", GOATCOUNTER);
    document.head.appendChild(g);
  }
})();
