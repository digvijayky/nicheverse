"""Tests for the structured project-context mechanism (annotate/context.py).

No network or GPU required; uses tmp_path for all file I/O.
"""

from __future__ import annotations

import warnings

import pytest

from nicheverse.annotate.context import (
    ANNOTATION_RULES,
    IMST_PLATFORMS,
    NON_IMST,
    CellTypePrior,
    NichePrior,
    ProjectContext,
)


def _ctx_dict():
    return {
        "name": "RCC-BrM",
        "species": "human",
        "disease": "clear cell renal cell carcinoma",
        "tissue": "kidney and brain metastasis",
        "platform": "xenium",
        "panel": "custom 366-gene panel",
        "sites": ["BrM", "Primary", "Metastasis"],
        "site_col": "site_class",
        "patient_col": "mrn",
        "context_cols": ["site_class", "sample_id"],
        "expected_cell_types": [
            {"name": "ccRCC tumor", "markers": ["CA9", "NDUFA4L2", "PAX8"], "notes": "CA9 may be low"},
            {"name": "T cell", "markers": ["CD3D", "CD8A"]},
        ],
        "expected_niches": [
            {"name": "tumor core", "description": "dense tumor", "expected_cell_types": ["ccRCC tumor"]},
        ],
        "references": ["PMID:34290408"],
        "notes": "Brain-met samples contain normal CNS cells.",
        "unknown_key_should_be_ignored": 123,
    }


def test_from_dict_coerces_nested_and_ignores_unknown():
    ctx = ProjectContext.from_dict(_ctx_dict())
    assert isinstance(ctx.expected_cell_types[0], CellTypePrior)
    assert ctx.expected_cell_types[0].markers == ["CA9", "NDUFA4L2", "PAX8"]
    assert isinstance(ctx.expected_niches[0], NichePrior)
    assert ctx.expected_niches[0].expected_cell_types == ["ccRCC tumor"]
    assert not hasattr(ctx, "unknown_key_should_be_ignored")


def test_to_dict_roundtrip():
    ctx = ProjectContext.from_dict(_ctx_dict())
    d = ctx.to_dict()
    ctx2 = ProjectContext.from_dict(d)
    assert ctx2.to_dict() == d
    assert ctx2.species == "human" and ctx2.expected_cell_types[0].name == "ccRCC tumor"


def test_from_yaml_to_yaml_roundtrip(tmp_path):
    ctx = ProjectContext.from_dict(_ctx_dict())
    p = tmp_path / "ctx.yaml"
    ctx.to_yaml(str(p))
    assert p.exists()
    reloaded = ProjectContext.from_yaml(str(p))
    assert reloaded.to_dict() == ctx.to_dict()


def test_to_prompt_block_contents():
    ctx = ProjectContext.from_dict(_ctx_dict())
    block = ctx.to_prompt_block()
    assert "human" in block
    assert "clear cell renal cell carcinoma" in block
    assert "CA9" in block  # a marker is rendered
    assert "xenium" in block
    # empty fields are omitted, no markdown "-" bullet lines
    assert not any(ln.lstrip().startswith("- ") for ln in block.splitlines())


def test_to_prompt_block_omits_empty_and_is_bounded():
    ctx = ProjectContext(species="human", disease="ccRCC")
    block = ctx.to_prompt_block()
    assert "human" in block and "ccRCC" in block
    assert "panel:" not in block  # empty field omitted
    assert len(ctx_full_block()) < 1500


def ctx_full_block():
    return ProjectContext.from_dict(_ctx_dict()).to_prompt_block()


def test_prompt_block_truncates_long_lists():
    many = [{"name": f"type{i}", "markers": [f"G{i}"]} for i in range(30)]
    ctx = ProjectContext(expected_cell_types=[CellTypePrior(**m) for m in many])
    block = ctx.to_prompt_block(max_cell_types=5)
    assert "more)" in block  # truncation marker
    assert block.count("type") <= 12  # not all 30 rendered


def test_write_template_reloads(tmp_path):
    p = tmp_path / "project_context.yaml"
    ProjectContext.write_template(str(p))
    assert p.exists()
    ctx = ProjectContext.from_yaml(str(p))
    assert isinstance(ctx, ProjectContext)
    assert ctx.species == "human"
    assert ctx.platform.lower() == "xenium"
    assert any(c.name and c.markers for c in ctx.expected_cell_types)


def test_write_template_matches_inline_fallback(tmp_path):
    # write_template should produce a valid, reloadable file whether or not the shipped
    # template file exists; both paths yield a human-species ccRCC Xenium example.
    p = tmp_path / "t.yaml"
    ProjectContext.write_template(str(p))
    ctx = ProjectContext.from_yaml(str(p))
    assert ctx.disease.lower().startswith("clear cell")


def test_imst_platform_ok():
    for plat in ["xenium", "Xenium", "COSMX", "merfish", "MERSCOPE", "seqfish+", "ISS"]:
        ctx = ProjectContext(platform=plat)
        assert ctx.platform == plat  # preserved as given


def test_non_imst_platform_raises():
    with pytest.raises(ValueError, match="IMST"):
        ProjectContext(platform="visium")
    with pytest.raises(ValueError, match="IMST"):
        ProjectContext(platform="Visium HD")
    with pytest.raises(ValueError):
        ProjectContext(platform="stereo-seq")


def test_unknown_platform_warns_but_keeps():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ctx = ProjectContext(platform="some_new_imager_9000")
    assert ctx.platform == "some_new_imager_9000"
    assert any(issubclass(x.category, UserWarning) for x in w)


def test_empty_platform_allowed():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning should fire
        ctx = ProjectContext(platform="")
    assert ctx.platform == ""


def test_platform_sets_disjoint():
    assert IMST_PLATFORMS.isdisjoint(NON_IMST)
    assert "xenium" in IMST_PLATFORMS and "visium" in NON_IMST


def test_annotation_rules_nonempty_str():
    assert isinstance(ANNOTATION_RULES, str)
    assert len(ANNOTATION_RULES.strip()) > 100
    assert "literature" in ANNOTATION_RULES.lower()
