# Contributing to nicheverse

Thanks for your interest in contributing. This package accompanies a peer
reviewed manuscript, so we hold contributions to the same standard the
manuscript itself was held to: reproducibility, clear documentation, and tests
that fail when they should.

## Development setup

```bash
git clone https://github.com/digvijayky/nicheverse.git
cd nicheverse
pip install -e ".[dev,doc,test]"
pre-commit install
```

The `dev` extra pulls `pre-commit`, `ruff`, `mypy`, `build`, and `twine`; `test`
pulls `pytest`, `pytest-cov`, `coverage`; `doc` pulls the Sphinx toolchain.
Installing the `pre-commit` hooks means `ruff` runs automatically on every
commit. The build backend is [hatchling](https://hatch.pypa.io/); building
distributions is `python -m build`.

## Running the test suite

```bash
pytest -q
```

The suite covers the VQ-VAE forward pass shapes and save/load round trip, the
spatial dataset edge cases, train-then-predict end-to-end, gene-panel
mismatch detection, the CLI via subprocess, the Xenium loader against a
synthetic-but-realistic output directory, and predict-time determinism.

Add a regression test for every bug fix and a coverage test for every new
feature. New tests go under `tests/` and follow the existing naming pattern
(`test_<module>.py`).

## Style

We use `ruff` with the configuration in `pyproject.toml`:

```bash
ruff check src tests
ruff format src tests
```

Type hints are required on every public function and method. `from __future__ import annotations` is on by default so postponed evaluation of annotations applies.

Docstrings follow numpy style with at least `Parameters`, `Returns`, and `Raises` sections when they apply. Public docstrings should give enough context for a Cancer Cell reviewer to understand what the function does without reading the implementation.

## Building the docs

```bash
python -m sphinx -M html docs docs/_build
# open docs/_build/html/index.html
```

## Pull request checklist

1. `pytest -q` passes.
2. `pre-commit run --all-files` is clean (runs `ruff check` and `ruff format`).
3. Public APIs touched in the PR have updated docstrings, and new public symbols
   are listed in `docs/api.md`.
4. `CHANGELOG.md` has a new bullet under `Unreleased` describing the change.
5. If the PR changes a determinism contract, `DETERMINISM.md` is updated.

## Reporting issues

Please open issues against
[github.com/digvijayky/nicheverse/issues](https://github.com/digvijayky/nicheverse/issues)
with a minimal reproducer (a small AnnData or a script that synthesizes one).
