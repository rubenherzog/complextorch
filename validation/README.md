# Scientific validation suites

This directory contains reproducible scientific validation and study-level checks that are intentionally excluded from the default pull-request regression suite (`pytest`, whose `testpaths` is `tests`).

The default suite protects public API contracts, mathematical identities, batching, dtype/device behavior, numerical backends, estimators, and focused regression cases. The suites here are broader evidence used when developing or auditing the corresponding scientific methods, including external-parity boundaries, Monte Carlo checks, large spectral grids, and SSDI study workflows.

Run all validation suites explicitly with:

```bash
python -m pytest -ra validation
```

Run an individual validation family when changing the corresponding implementation, for example:

```bash
python -m pytest -ra validation/test_faes_scientific_validation.py
python -m pytest -ra validation/test_ssdi_validation.py
```

These suites remain version-controlled so scientific evidence is reproducible without imposing study-level checks on every merge.
