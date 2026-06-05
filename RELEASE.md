# SOMA-X Release Checklist

This checklist covers public `py-soma-x` package releases from the public-safe
GitHub mirror.

## One-time PyPI setup

Configure Trusted Publishing for both PyPI and TestPyPI before creating a
release tag.

Use these publisher settings for the PyPI project `py-soma-x`:

- Owner: `NVlabs`
- Repository: `SOMA-X`
- Workflow: `pypi.yml`
- Environment: `pypi`

Use the same settings on TestPyPI with environment `testpypi`.

Protect the `pypi` and `testpypi` GitHub environments so publishing requires
maintainer approval. Do not configure long-lived PyPI API tokens for this
workflow.

## Release steps

1. Merge the public-release prep MRs into internal `main`.
2. Cut or refresh the minor-line internal release branch, for example
   `release-0.2`. Patch releases reuse the same minor-line branch and create a
   new patch tag, for example `v0.2.1` from `release-0.2`.
3. Confirm `setup.cfg` and `soma/__init__.py` both contain the intended package
   version, for example `0.2.0`.
4. Mirror the public-safe release branch to public GitHub.
5. Confirm the generated public mirror candidate passed the internal
   public-release validation gate before pushing.
6. Confirm the public GitHub Actions build job passes on the release branch.
7. Create the release tag from the public-safe release branch:

   ```bash
   git tag -a vX.Y.Z -m "SOMA-X X.Y.Z"
   git push origin vX.Y.Z
   ```

8. Verify the tag-triggered workflow publishes to TestPyPI first.
9. Approve the protected `pypi` environment only after TestPyPI verification.
10. Verify PyPI shows the new `py-soma-x` release.
11. Record release links for the GitHub tag, PyPI release, docs, and validation
    artifact.

## Local checks

Run these from the release branch before tagging:

```bash
python tools/ci/check_release_version.py --expected vX.Y.Z
python -m build --sdist --wheel
python -m twine check --strict dist/*
```
