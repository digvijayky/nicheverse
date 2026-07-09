(function () {
  try {
    var d = document.documentElement;
    if (!localStorage.getItem("pst-color-theme")) localStorage.setItem("pst-color-theme", "dark");
    var t = localStorage.getItem("pst-color-theme") || "dark";
    d.setAttribute("data-theme", t);
    d.setAttribute("data-mode", t);
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "dark");
  }
})();
