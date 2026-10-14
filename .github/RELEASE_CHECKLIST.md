# v0.11.0 release verification

Issue #184 stays open until every publication check below has evidence.
Merging the preparation PR alone does not complete the release. Do not mark
TestPyPI/PyPI or hosted-documentation checks complete from a PR-only run.

## Before publication

- Confirm #175-#183 are merged: encoder loading, roles/gradients, objectives,
  evaluation/mining, batched encoding, versioned persistence, compatibility,
  API docs/examples and benchmark protocol. Record the preparation PR evidence.
- Review and merge the preparation PR; record the exact main commit with package
  version 0.11.0 and matching `.github/RELEASE_NOTES.md`.
- Confirm Python 3.10/3.11/3.12 CI and the Sphinx warning-as-error build.
- Confirm minimum/current Encoder compatibility jobs, including offline unit
  tests and pinned MiniLM/E5 real-stack gates, on that main commit.
- Run the Release workflow on that commit. PR runs verify and build only;
  manual runs on main also publish to TestPyPI via Trusted Publishing.
- Confirm bounded real-stack acceptance for v0.4-v0.11, wheel/sdist build, twine
  checks and CI's installed-wheel smoke. Unit gates retain no-role raw text,
  eight immutable versionless v0.10 model fixtures, and prior geometry contracts.
- Confirm TestPyPI upload and installed-wheel smoke, including SHA-256 equality
  with the validated build artifact, v0.4-v0.10 examples, v0.11 roles/batching and
  metadata/legacy-load checks. Confirm the qualified dependency floors remain
  PyTorch 2.3.0, Sentence Transformers 5.4.1 and Geoopt 0.5.1 (see the compatibility
  guide and `tests/compatibility/minimum.txt` for the complete qualified stack).
- Record the successful TestPyPI run and source SHA. If main changes, repeat
  verification for the commit intended for release.

## Production release

- Create the exact tag v0.11.0 at that verified main commit. The workflow checks
  tag/package-version equality and that the tag points at current main.
- Confirm the tag-triggered Release workflow publishes through the pypi
  Trusted Publishing environment.
- Confirm the production wheel matches the validated build artifact and passes
  installed-version, v0.4-v0.11 smoke and legacy/no-role checks.
- Confirm hosted Read the Docs single-manifold, mixed-curvature, public API,
  encoder compatibility and benchmark-protocol pages all carry the
  `neembed 0.11.0` marker and their respective guide content.
- Confirm the workflow creates the v0.11.0 GitHub Release from
  `.github/RELEASE_NOTES.md` only after production smoke and hosted docs succeed.
- Record tag/commit, workflow URLs, TestPyPI/PyPI package URLs, hosted docs and
  GitHub Release evidence in Issue #184, then close it.

## Saved models and scope

Versionless v0.10 and older configs remain readable without rewriting. Re-saving
writes model format 1 with prompt/dtype/source metadata; package version 0.11.0
is not the model format or a minimum-loader requirement. Preserve caller-owned
optimizer and prototype checkpoints separately. See `docs/user_guide/saving_loading.rst`.

This release retains fixed Product curvature/scales and adds no API freeze,
real-data performance conclusion, ANN/vector database, distributed training,
nested products, product prototypes or automatic geometry selection.
