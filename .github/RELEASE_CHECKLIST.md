# v0.10.0 release verification

Issue #137 stays open until every publication check below has evidence.
Merging the preparation PR alone does not complete the release.

## Before publication

- Review and merge the preparation PR.
- Record the exact main commit with package version 0.10.0.
- Confirm Python 3.10/3.11/3.12 CI and the Sphinx warning-as-error build.
- Run the Release workflow on that commit. PR runs verify and build only;
  manual runs on main also publish to TestPyPI via Trusted Publishing.
- Confirm bounded real-stack acceptance, wheel/sdist build and twine checks.
- Confirm TestPyPI upload and smoke success, including the downloaded wheel's
  SHA-256 equality with the build artifact and all v0.4-v0.10 regression examples.
- Record the successful TestPyPI run and its source SHA. If main changes, repeat
  verification for the commit intended for the release.

## Production release

- Create the exact tag v0.10.0 at that verified main commit. The workflow checks
  tag/package-version equality and that the tag points at current main.
- Confirm the tag-triggered Release workflow publishes through the pypi
  Trusted Publishing environment.
- Confirm the production wheel matches the validated build artifact and passes
  the installed-version and v0.4-v0.10 smoke checks.
- Confirm hosted Read the Docs single-manifold and mixed-curvature pages both
  carry the neembed 0.10.0 version marker and their respective guide content.
- Confirm the workflow creates the v0.10.0 GitHub Release only after production
  smoke and hosted-documentation verification succeed.
- Record tag/commit, workflow URLs, TestPyPI/PyPI package URLs, hosted docs, and
  GitHub Release evidence in Issue #137, then close it.

Product curvature and scales remain fixed. This release does not add nested
products, product prototypes, advanced matrix manifolds, automatic component
selection, ANN/vector databases, or distributed retrieval/training.
