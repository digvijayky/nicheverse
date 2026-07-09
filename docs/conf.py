"""Sphinx configuration for the nicheverse documentation."""
from __future__ import annotations

from importlib.metadata import metadata, version as _pkg_version

# -- Project information -----------------------------------------------------
_meta = metadata("nicheverse")
project = "NICHEVERSE"
author = _meta["Author"] or "Digvijay Yarlagadda"
copyright = "2026, Digvijay Yarlagadda"
release = _pkg_version("nicheverse")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".rst": "restructuredtext",
}

master_doc = "index"

# -- MyST / myst-nb ----------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
    "attrs_block",
]
myst_heading_anchors = 3
nb_execution_mode = "off"

# -- autosummary / autodoc / napoleon ----------------------------------------
autosummary_generate = False
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports: list[str] = []

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = True
napoleon_use_param = True

typehints_defaults = "comma"
always_document_param_types = True

# -- Bibliography ------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "NICHEVERSE"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["force-dark.js", "gallery.js", "bb-motion.js"]
# Canonical URL for the GitHub Pages deploy (harmless locally).
html_baseurl = "https://nicheverse.github.io/"
html_theme_options = {
    "github_url": "https://github.com/digvijayky/nicheverse",
    "show_prev_next": False,
    "navbar_align": "left",
    "default_mode": "dark",
    "header_links_before_dropdown": 3,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "logo": {"text": "NICHEVERSE"},
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/digvijayky/nicheverse", "icon": "fa-brands fa-github"},
        {"name": "PyPI", "url": "https://pypi.org/project/nicheverse/", "icon": "fa-brands fa-python"},
    ],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "footer_start": [],
    "footer_end": [],
    "secondary_sidebar_items": ["page-toc"],
}
html_sidebars = {"index": [], "guides/gallery": []}

# -- OpenGraph ---------------------------------------------------------------
ogp_site_name = "NICHEVERSE documentation"
ogp_use_first_image = True

# -- Silence expected cross-reference noise ----------------------------------
nitpicky = False
suppress_warnings = ["mystnb.unknown_mime_type"]
