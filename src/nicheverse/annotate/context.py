"""Structured project context for grounding LLM codebook annotation.

A :class:`ProjectContext` lets a user hand the annotation harness rich, structured
knowledge about their study (species, disease, tissue, imaging platform, gene panel,
expected cell types and their markers, expected spatial niches, authoritative
references, and free-text guidance). :meth:`ProjectContext.to_prompt_block` renders a
compact, deterministic text block that can be injected into an LLM annotation prompt,
and :data:`ANNOTATION_RULES` supplies the lab's codebook annotation conventions as
guidance text.

nicheverse targets imaging-based / in-situ spatial transcriptomics (IMST) data only,
so the platform field is validated against :data:`IMST_PLATFORMS` / :data:`NON_IMST`.
"""

from __future__ import annotations

import os
import shutil
import warnings
from dataclasses import asdict, dataclass, field

__all__ = [
    "CellTypePrior",
    "NichePrior",
    "ProjectContext",
    "IMST_PLATFORMS",
    "NON_IMST",
    "ANNOTATION_RULES",
]

# Imaging-based / in-situ spatial transcriptomics platforms and common aliases.
# Matching is case-insensitive (values are lowercase; normalization lowercases input).
IMST_PLATFORMS = {
    "xenium",
    "10x xenium",
    "cosmx",
    "nanostring cosmx",
    "merfish",
    "merscope",
    "vizgen",
    "vizgen merscope",
    "seqfish",
    "seqfish+",
    "seqfishplus",
    "starmap",
    "osmfish",
    "eel fish",
    "eel",
    "molecular cartography",
    "resolve",
    "resolve biosciences",
    "hybiss",
    "iss",
    "in situ sequencing",
}

# Sequencing-based / array-based spatial platforms that nicheverse does NOT support.
NON_IMST = {
    "visium",
    "visium hd",
    "slide-seq",
    "slide-seqv2",
    "stereo-seq",
    "dbit-seq",
    "geomx",
    "hdst",
    "pixel-seq",
}


def _norm(s: str) -> str:
    return (s or "").strip().lower()


@dataclass
class CellTypePrior:
    """A user-provided expectation for a cell type present in the study."""

    name: str
    markers: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class NichePrior:
    """A user-provided expectation for a multicellular spatial niche."""

    name: str
    description: str = ""
    expected_cell_types: list[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    """Structured, user-supplied context grounding LLM codebook annotation.

    All fields are optional. ``platform`` is validated on construction: a known
    non-IMST platform raises :class:`ValueError` (nicheverse is imaging-based only),
    an unknown non-empty platform warns but is kept, and an empty platform is allowed.
    """

    name: str = ""
    species: str = ""
    disease: str = ""
    tissue: str = ""
    platform: str = ""
    panel: str = ""
    sites: list[str] = field(default_factory=list)
    expected_cell_types: list[CellTypePrior] = field(default_factory=list)
    expected_niches: list[NichePrior] = field(default_factory=list)
    site_col: str = ""
    patient_col: str = ""
    context_cols: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        p = _norm(self.platform)
        if p and p in NON_IMST:
            raise ValueError(
                f"platform {self.platform!r} is a sequencing/array-based spatial platform; "
                "nicheverse supports imaging-based / in-situ spatial transcriptomics (IMST) "
                "data only (e.g. Xenium, CosMx, MERFISH/MERSCOPE, seqFISH, STARmap, ISS)."
            )
        if p and p not in IMST_PLATFORMS:
            warnings.warn(
                f"platform {self.platform!r} is not a recognized IMST platform; "
                "keeping it, but confirm it is imaging-based (in-situ) spatial data.",
                stacklevel=2,
            )

    # -- (de)serialization -------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict | None) -> "ProjectContext":
        """Build from a plain dict, tolerating unknown keys and coercing nested priors."""
        d = dict(d or {})
        fields = {f for f in cls.__dataclass_fields__}
        ct = d.get("expected_cell_types", []) or []
        nz = d.get("expected_niches", []) or []
        cell_types = [
            c if isinstance(c, CellTypePrior) else CellTypePrior(
                name=str(c.get("name", "")),
                markers=list(c.get("markers", []) or []),
                notes=str(c.get("notes", "")),
            )
            for c in ct
        ]
        niches = [
            n if isinstance(n, NichePrior) else NichePrior(
                name=str(n.get("name", "")),
                description=str(n.get("description", "")),
                expected_cell_types=list(n.get("expected_cell_types", []) or []),
            )
            for n in nz
        ]
        kept = {k: v for k, v in d.items() if k in fields}
        kept["expected_cell_types"] = cell_types
        kept["expected_niches"] = niches
        return cls(**kept)

    @classmethod
    def from_yaml(cls, path: str) -> "ProjectContext":
        """Load from a YAML file via :func:`yaml.safe_load`, then :meth:`from_dict`."""
        import yaml

        with open(path) as fh:
            d = yaml.safe_load(fh)
        return cls.from_dict(d or {})

    def to_dict(self) -> dict:
        """Return a plain, YAML-serializable dict (nested priors expanded)."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Write the context to a YAML file."""
        import yaml

        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)

    # -- prompt rendering --------------------------------------------------

    def to_prompt_block(self, max_cell_types: int = 12, max_niches: int = 8, max_markers: int = 8) -> str:
        """Render a compact, deterministic plain-text context block for an LLM prompt.

        Omits empty fields, truncates long lists gracefully, and stays well under
        ~1500 chars for modest inputs. Uses ``name: value`` lines and numbered items
        (no markdown ``-`` bullets).
        """
        lines: list[str] = ["PROJECT CONTEXT"]

        def kv(label: str, value: str):
            if value:
                lines.append(f"{label}: {value}")

        kv("study", self.name)
        kv("species", self.species)
        kv("disease", self.disease)
        kv("tissue", self.tissue)
        kv("platform", self.platform)
        kv("panel", self.panel)
        if self.sites:
            kv("sites present", ", ".join(str(s) for s in self.sites))

        if self.expected_cell_types:
            lines.append("expected cell types (name: key markers):")
            shown = self.expected_cell_types[:max_cell_types]
            for i, c in enumerate(shown, 1):
                mk = ", ".join(str(m) for m in c.markers[:max_markers]) if c.markers else "no markers given"
                lines.append(f"  {i}. {c.name}: {mk}")
            if len(self.expected_cell_types) > len(shown):
                lines.append(f"  (+{len(self.expected_cell_types) - len(shown)} more)")

        if self.expected_niches:
            lines.append("expected niches (name: composition):")
            shown_n = self.expected_niches[:max_niches]
            for i, n in enumerate(shown_n, 1):
                comp = ", ".join(str(t) for t in n.expected_cell_types) if n.expected_cell_types else ""
                desc = n.description or comp
                lines.append(f"  {i}. {n.name}: {desc}" if desc else f"  {i}. {n.name}")
            if len(self.expected_niches) > len(shown_n):
                lines.append(f"  (+{len(self.expected_niches) - len(shown_n)} more)")

        if self.references:
            lines.append("authoritative references: " + ", ".join(str(r) for r in self.references[:10]))

        if self.notes:
            note = self.notes.strip()
            if len(note) > 600:
                note = note[:597].rstrip() + "..."
            lines.append(f"notes: {note}")

        return "\n".join(lines)

    # -- template ----------------------------------------------------------

    @staticmethod
    def _template_path() -> str:
        return os.path.join(os.path.dirname(__file__), "templates", "project_context.yaml")

    @staticmethod
    def write_template(path: str) -> None:
        """Write an annotated example ``project_context.yaml`` the user can fill in.

        Copies the shipped template next to this module when present; otherwise writes
        the same content inline (so it does not hard-depend on package-data config).
        """
        src = ProjectContext._template_path()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(path):
            shutil.copy2(src, path)
            return
        with open(path, "w") as fh:
            fh.write(_TEMPLATE_YAML)


# Annotated example template, shipped as templates/project_context.yaml and used as the
# inline fallback for write_template when the file is unavailable.
_TEMPLATE_YAML = """\
# nicheverse project context (annotated example: RCC brain-metastasis Xenium)
# Fill in what you know; leave fields blank/[] to omit them from the LLM prompt.
# nicheverse supports imaging-based / in-situ spatial transcriptomics (IMST) ONLY
# (e.g. Xenium, CosMx, MERFISH/MERSCOPE, seqFISH, STARmap, ISS). A sequencing/array
# platform (Visium, Slide-seq, Stereo-seq, GeoMx, ...) is rejected.

name: RCC brain-metastasis atlas          # short study name
species: human
disease: clear cell renal cell carcinoma  # dominant histology in the cohort
tissue: kidney and brain metastasis
platform: xenium                          # imaging-based platform (10x Xenium)
panel: custom 366-gene Xenium panel       # panel name / size

sites:                                    # anatomical/site classes present in the data
  - BrM
  - Primary
  - Metastasis

# obs columns the harness should use for per-code context summaries
site_col: site_class                      # obs column holding the site label
patient_col: mrn                          # obs column identifying the patient
context_cols:                             # extra obs columns to summarize per code
  - site_class
  - sample_id
  - diagnosis

# Cell types you expect, with a few canonical markers each. These anchor the LLM's
# labels; markers are gene symbols on your panel.
expected_cell_types:
  - name: ccRCC tumor
    markers: [CA9, NDUFA4L2, VHL, PAX8]
    notes: CA9 can be low in some tumor subclones; do not require it.
  - name: T cell
    markers: [CD3D, CD3E, CD8A, IL7R]
    notes: ""
  - name: Macrophage
    markers: [CD68, CD163, LYZ, C1QA]
    notes: ""
  - name: Endothelial
    markers: [PECAM1, VWF, CLDN5]
    notes: ""
  - name: Fibroblast
    markers: [COL1A1, DCN, PDGFRB]
    notes: ""
  - name: Astrocyte
    markers: [GFAP, AQP4, ALDH1L1]
    notes: Expected only at brain-metastasis sites.

# Spatial niches (multicellular microenvironments) you expect, by their composition.
expected_niches:
  - name: tumor core
    description: dense ccRCC tumor with sparse immune infiltrate
    expected_cell_types: [ccRCC tumor, Macrophage]
  - name: tumor-immune boundary
    description: interface of tumor and infiltrating lymphocytes
    expected_cell_types: [ccRCC tumor, T cell, Macrophage]

# PMIDs / DOIs you consider authoritative for this system.
references:
  - PMID:34290408        # Krishna 2021, ccRCC single-cell immune landscape
  - PMID:34019793        # Bi 2021, ccRCC ICI response
  - PMID:33861994        # Braun 2021, ccRCC tumor microenvironment

# Free-text guidance for the annotator (biology, caveats, priorities).
notes: >
  Brain-metastasis samples contain normal CNS cells (astrocytes, oligodendrocytes,
  microglia); do not mistake tumor-adjacent glial transcript leakage for a glial code.
  CA9-low, PAX8-retaining codes spanning most patients are still ccRCC tumor.
"""


# Codebook annotation rules the LLM should follow. Written for prompt injection.
ANNOTATION_RULES = """\
CODEBOOK ANNOTATION RULES

1. Literature support. Every proposed label must be defensible from primary
   literature. Cite at least one source per call as PMID or DOI, with the first
   author and publication year. Do not rely on model familiarity alone.

2. Universal-phenotype check. A code that spans most patients in the cohort and is
   dominated by the study's disease histology and its main tissue site is most likely
   tumor, not normal-tissue contamination, even when a canonical lineage marker reads
   low (marker-low tumor subpopulations are common). Weigh prevalence and context, not
   a single marker.

3. Rare-patient codes. A code seen in only a few patients is more likely a
   patient-specific tumor subclone or a rare cell state than a broadly generalizable
   cell type. Label it cautiously and flag the limited support in the rationale.

4. Site-aware reassignment. If a code is labeled a site-restricted type (for example a
   CNS-only glial type) but contains cells drawn from a non-permissive site (for
   example a kidney primary), reassign those out-of-site cells to the closest general
   label consistent with their markers rather than forcing the site-restricted call.

5. Composite labels. Use a combined "X/Y" label only when BOTH components have clear
   support, both in absolute expression and relative to the cohort. If only one
   component is well supported, pick that dominant type and note the minor signal.

6. Segmentation leakage. In imaging-based data, weak off-lineage markers within a code
   are frequently transcript misassignment bleeding in from physically adjacent cells.
   Treat faint off-lineage signal as possible segmentation leakage and do not
   over-interpret it as a mixed or hybrid identity.
"""
