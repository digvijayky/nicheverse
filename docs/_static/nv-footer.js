// Appends a "Developer: digvijayky" credit to the footer of every page.
(function () {
  function add() {
    if (document.getElementById("nv-dev-credit")) return;
    var host = document.querySelector(".bd-footer__inner") || document.querySelector(".bd-footer") || document.querySelector("footer") || document.body;
    var d = document.createElement("div");
    d.id = "nv-dev-credit";
    d.innerHTML = 'Developer: <a href="http://digvijayky.com/" target="_blank" rel="noopener noreferrer">digvijayky</a>';
    host.appendChild(d);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", add);
  else add();
})();
