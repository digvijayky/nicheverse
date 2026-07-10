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


# --------------------------------------------------------------------------- #
# Hardening coverage added in this review pass.
# --------------------------------------------------------------------------- #

import os

import nicheverse.annotate.context as _ctxmod


def test_from_dict_none_and_empty_yield_empty_context():
    for d in (None, {}):
        ctx = ProjectContext.from_dict(d)
        assert isinstance(ctx, ProjectContext)
        assert ctx.name == "" and ctx.platform == ""
        assert ctx.expected_cell_types == [] and ctx.expected_niches == []


def test_from_yaml_empty_file_is_empty_context(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")  # yaml.safe_load -> None
    ctx = ProjectContext.from_yaml(str(p))
    assert isinstance(ctx, ProjectContext) and ctx.species == ""
    p2 = tmp_path / "comments.yaml"
    p2.write_text("# only a comment, no keys\n")
    ctx2 = ProjectContext.from_yaml(str(p2))
    assert isinstance(ctx2, ProjectContext) and ctx2.expected_cell_types == []


def test_from_dict_nested_as_dataclass_dict_and_mixed():
    # dataclass instances pass through; dicts are coerced; a mix works.
    d = {
        "expected_cell_types": [
            CellTypePrior("A", ["G1"]),
            {"name": "B", "markers": ["G2"], "notes": "n"},
        ],
        "expected_niches": [
            NichePrior("N1", "d", ["A"]),
            {"name": "N2", "expected_cell_types": ["B"]},
        ],
    }
    ctx = ProjectContext.from_dict(d)
    assert all(isinstance(c, CellTypePrior) for c in ctx.expected_cell_types)
    assert all(isinstance(n, NichePrior) for n in ctx.expected_niches)
    assert ctx.expected_cell_types[0].name == "A" and ctx.expected_cell_types[1].markers == ["G2"]
    assert ctx.expected_niches[0].description == "d" and ctx.expected_niches[1].expected_cell_types == ["B"]


def test_from_dict_skips_malformed_nested_items():
    # a bare string (or other non-dict/non-dataclass) among nested priors must be skipped,
    # not raise AttributeError('str' has no attribute 'get').
    d = {
        "expected_cell_types": ["oops", {"name": "Good", "markers": ["G1"]}, 42, None],
        "expected_niches": ["bad", {"name": "GoodNiche"}],
    }
    ctx = ProjectContext.from_dict(d)
    assert [c.name for c in ctx.expected_cell_types] == ["Good"]
    assert [n.name for n in ctx.expected_niches] == ["GoodNiche"]


def test_from_dict_missing_and_unknown_keys():
    ctx = ProjectContext.from_dict({"species": "human", "totally_unknown": 1, "another": [1, 2]})
    assert ctx.species == "human" and ctx.disease == "" and ctx.sites == []
    assert not hasattr(ctx, "totally_unknown")


def test_platform_case_and_whitespace_normalized():
    # leading/trailing whitespace and case must not defeat NON_IMST detection.
    with pytest.raises(ValueError, match="IMST"):
        ProjectContext(platform="  Visium  ")
    with pytest.raises(ValueError, match="IMST"):
        ProjectContext(platform="SLIDE-SEQ")
    # and must not defeat IMST recognition (no warning fires for a padded known platform)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ctx = ProjectContext(platform="  CosMx ")
    assert ctx.platform == "  CosMx "  # raw value preserved


def test_platform_aliases_recognized_without_warning():
    for alias in ["nanostring cosmx", "vizgen merscope", "seqfishplus", "in situ sequencing", "eel"]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ctx = ProjectContext(platform=alias)
        assert ctx.platform == alias


def test_every_non_imst_entry_raises_naming_imst():
    for plat in sorted(NON_IMST):
        with pytest.raises(ValueError, match="IMST"):
            ProjectContext(platform=plat)


def test_unknown_non_null_platform_warns_not_raises():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ctx = ProjectContext(platform="brandnew_imager_v2")
    assert ctx.platform == "brandnew_imager_v2"
    assert any(issubclass(x.category, UserWarning) for x in w)


def test_non_string_platform_does_not_crash():
    # a malformed YAML can hand us an int/float/bool for platform; must not AttributeError.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ProjectContext(platform=123).platform == 123
        assert ProjectContext(platform=True).platform is True
    assert _ctxmod._norm(123) == "123"
    assert _ctxmod._norm(None) == "" and _ctxmod._norm("  X ") == "x"


def test_to_prompt_block_deterministic():
    ctx = ProjectContext.from_dict(_ctx_dict())
    assert ctx.to_prompt_block() == ctx.to_prompt_block()


def test_to_prompt_block_truncates_markers():
    ct = CellTypePrior("big", [f"G{i}" for i in range(30)])
    ctx = ProjectContext(expected_cell_types=[ct])
    block = ctx.to_prompt_block(max_markers=8)
    line = [ln for ln in block.splitlines() if ln.strip().startswith("1. big")][0]
    rendered = line.split(":", 1)[1]
    assert rendered.count("G") == 8  # only first 8 markers shown, no "(+N more)" per-marker


def test_to_prompt_block_truncates_niches():
    niches = [NichePrior(f"n{i}", "d") for i in range(20)]
    ctx = ProjectContext(expected_niches=niches)
    block = ctx.to_prompt_block(max_niches=6)
    assert "more)" in block
    assert sum(1 for ln in block.splitlines() if ln.strip()[:2].rstrip(".").isdigit()) <= 6 + 1


def test_to_prompt_block_char_budget_for_modest_input():
    # the docstring promises "well under ~1500 chars for modest inputs".
    ctx = ProjectContext.from_dict(_ctx_dict())
    assert len(ctx.to_prompt_block()) < 1500


def test_to_prompt_block_notes_truncated():
    ctx = ProjectContext(notes="x" * 2000, species="human")
    block = ctx.to_prompt_block()
    note_line = [ln for ln in block.splitlines() if ln.startswith("notes:")][0]
    assert note_line.endswith("...") and len(note_line) < 700


def test_to_prompt_block_no_emdash_no_bullets_even_from_user_content():
    # user content carrying an em-dash / leading bullet marker must be sanitized so the
    # rendered block always complies with the repo prose rules.
    ctx = ProjectContext(
        name="study — with dash",
        notes="alpha — beta – gamma",
        sites=["- bulleted site"],
        expected_cell_types=[CellTypePrior("- weird name", ["- G1", "G2"], notes="")],
        expected_niches=[NichePrior("* star niche", "desc — here", ["- x"])],
        references=["- PMID:1"],
        species="human",
    )
    block = ctx.to_prompt_block()
    assert "—" not in block and "–" not in block  # no em/en dash
    for ln in block.splitlines():
        assert not ln.lstrip().startswith("- ")
        assert not ln.lstrip().startswith("* ")
        assert not ln.lstrip().startswith("+ ")


def test_to_prompt_block_default_structure_has_no_emdash_or_bullets():
    ctx = ProjectContext.from_dict(_ctx_dict())
    block = ctx.to_prompt_block()
    assert "—" not in block
    assert not any(ln.lstrip().startswith(("- ", "* ", "+ ")) for ln in block.splitlines())


def test_write_template_roundtrip_equivalent(tmp_path):
    p = tmp_path / "project_context.yaml"
    ProjectContext.write_template(str(p))
    ctx = ProjectContext.from_yaml(str(p))
    # write -> from_yaml yields a fully-formed, self-consistent context
    assert ctx.to_dict() == ProjectContext.from_dict(ctx.to_dict()).to_dict()
    assert ctx.species == "human" and ctx.platform.lower() == "xenium"
    assert len(ctx.expected_cell_types) >= 3 and len(ctx.expected_niches) >= 1


def test_write_template_inline_fallback_matches_shipped(tmp_path, monkeypatch):
    # when the shipped template is unavailable, the inline fallback must produce a file
    # byte-identical to the shipped template.
    shipped = ProjectContext._template_path()
    assert os.path.isfile(shipped), "shipped template should exist in the package"
    with open(shipped, "rb") as fh:
        shipped_bytes = fh.read()
    # force the fallback branch by pointing _template_path at a nonexistent file
    monkeypatch.setattr(
        ProjectContext, "_template_path", staticmethod(lambda: str(tmp_path / "nope.yaml"))
    )
    p = tmp_path / "fallback.yaml"
    ProjectContext.write_template(str(p))
    with open(p, "rb") as fh:
        fallback_bytes = fh.read()
    assert fallback_bytes == shipped_bytes


def test_shipped_template_byte_consistent_with_inline_constant():
    # the shipped templates/project_context.yaml must stay byte-identical to the inline
    # _TEMPLATE_YAML constant used as the fallback.
    shipped = ProjectContext._template_path()
    with open(shipped, "r") as fh:
        shipped_text = fh.read()
    assert shipped_text == _ctxmod._TEMPLATE_YAML


def test_annotation_rules_cover_all_conventions():
    low = ANNOTATION_RULES.lower()
    for kw in ["literature", "universal", "rare", "site-aware", "composite", "segmentation"]:
        assert kw in low, f"ANNOTATION_RULES missing '{kw}'"
    assert "—" not in ANNOTATION_RULES  # no em-dash in the rules text
    assert not any(ln.lstrip().startswith(("- ", "* ", "+ ")) for ln in ANNOTATION_RULES.splitlines())
