"""Literature-grounded, iterative LLM annotation of learned codes.

Generate per-code evidence with :func:`code_evidence`, then label every code with
:func:`annotate_codes` using Claude, GPT, or a local model. LLM and network
dependencies are optional: ``pip install nicheverse[llm]``.
"""

from __future__ import annotations

from .annotate import AnnotationConfig, annotate_codes, annotate_niches, attach_labels
from .artifacts import (
    cluster_codes,
    code_context,
    code_evidence,
    code_groundtruth_concordance,
    niche_evidence,
    write_evidence_bundle,
)
from .context import (
    ANNOTATION_RULES,
    IMST_PLATFORMS,
    CellTypePrior,
    NichePrior,
    ProjectContext,
)
from .evaluate import calibration, score_code, scorecard_table, write_provenance_manifest
from .harness import AnnotationResult, annotate_codebook
from .literature import biorxiv_search, literature_for_markers, pubmed_search
from .plots import code_dotplot
from .providers import PROVIDERS, call_llm
from .verify import apply_lab_rules, check_citations, gate, marker_presence, validate_vocabulary

__all__ = [
    "AnnotationConfig",
    "annotate_codes",
    "annotate_niches",
    "attach_labels",
    "annotate_codebook",
    "AnnotationResult",
    "code_evidence",
    "cluster_codes",
    "code_context",
    "code_groundtruth_concordance",
    "write_evidence_bundle",
    "code_dotplot",
    "niche_evidence",
    "ProjectContext",
    "CellTypePrior",
    "NichePrior",
    "ANNOTATION_RULES",
    "IMST_PLATFORMS",
    "gate",
    "marker_presence",
    "check_citations",
    "apply_lab_rules",
    "validate_vocabulary",
    "score_code",
    "calibration",
    "scorecard_table",
    "write_provenance_manifest",
    "pubmed_search",
    "biorxiv_search",
    "literature_for_markers",
    "call_llm",
    "PROVIDERS",
]
