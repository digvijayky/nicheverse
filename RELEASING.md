# Releasing nicheverse

This is the maintainer runbook for cutting a new release. Releases are published
to PyPI automatically via GitHub Actions using PyPI trusted publishing, so no
API tokens or passwords are stored anywhere.

## One-time setup

Do this once per repository, before the first automated release:

1. **Configure a PyPI trusted publisher.** On PyPI, go to the project's
   *Publishing* settings (or the *Pending publishers* page if the project does
   not exist yet) and add a trusted publisher with:
   - Owner: the GitHub org or user that owns this repo
   - Repository name: `nicheverse`
   - Workflow name: `release.yaml`
   - Environment name: `pypi`
2. **Create the GitHub environment.** In the repo settings under
   *Environments*, create an environment named `pypi`. This matches the
   `environment.name` in `.github/workflows/release.yaml` and lets you attach
   optional protection rules (required reviewers, deployment branches).

## Cutting a release

1. **Bump the version.** Update `version` in `pyproject.toml` to the new
   `X.Y.Z` (follow semantic versioning).
2. **Update the changelog.** Move the accumulated notes in `CHANGELOG.md` from
   the unreleased section into a new `X.Y.Z` section with today's date.
3. **Commit the changes.**

   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Release vX.Y.Z"
   git push
   ```

4. **Tag and push the tag.**

   ```bash
   git tag vX.Y.Z
   git push --tags
   ```

5. **Create a GitHub Release from the tag.** In the GitHub UI go to
   *Releases -> Draft a new release*, choose the `vX.Y.Z` tag, add release
   notes (the changelog section works well), and publish. Publishing the
   release triggers `.github/workflows/release.yaml`, which builds the sdist and
   wheel, runs `twine check`, and uploads to PyPI via trusted publishing.
6. **Verify.** Confirm the release job in the Actions tab is green and that the
   new version appears at https://pypi.org/project/nicheverse/.

## Manual fallback

If the automated workflow is unavailable, publish from a clean checkout:

```bash
python -m build
twine check dist/*
twine upload dist/*
```

This path requires a PyPI API token configured locally (for example in
`~/.pypirc` or via the `TWINE_USERNAME`/`TWINE_PASSWORD` environment variables).

## Future: conda-forge

Once the package is stable on PyPI, a conda-forge feedstock can be added so
users can `conda install -c conda-forge nicheverse`. This is done by
submitting a recipe to
[staged-recipes](https://github.com/conda-forge/staged-recipes); afterward the
conda-forge bot opens update pull requests automatically whenever a new PyPI
release is detected.
