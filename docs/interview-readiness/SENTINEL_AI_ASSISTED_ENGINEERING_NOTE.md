# Sentinel AI-Assisted Engineering Note

## Position

Using AI assistance on Sentinel is not something to apologize for. The engineering question is whether the current repository can be inspected, explained, tested, debugged, documented, and evolved responsibly.

The ownership standard for this project is:

- Understand what the code currently does.
- Separate implemented behavior from tested behavior.
- Avoid inflating simulated results into production claims.
- Run verification before claiming something passes.
- Add targeted tests around risky behavior.
- Document boundaries and future work honestly.

## How To Frame It In Interviews

Safe wording:

> I used AI as an accelerator, but I treat AI-assisted code like inherited code. My responsibility is to inspect it, understand the architecture, verify behavior with tests, find edge cases, document the system, and avoid unsupported claims.

Avoid:

> AI wrote it, so I am not sure how it works.

Avoid:

> I wrote every line from scratch.

## Evidence Of Engineering Ownership

### Repo Inspection

You should be able to explain:

- Why `ml/constants.py` matters.
- How `ModelService` loads model artifacts.
- How `POST /api/predict` validates, predicts, stores, and responds.
- Why `predictions` and `vehicle_metrics` are different tables.
- How demo mode differs from full-stack mode.

### Behavior Verification

Current verified commands from this pass:

```bash
python3 -m pytest tests/ -q --tb=short
```

```text
242 passed in 3.92s
```

```bash
cd dashboard
npm test -- --run --reporter=dot
```

```text
8 test files passed
59 tests passed
```

```bash
cd dashboard
npm run lint
npm run build
```

```text
lint passed
build passed with a 905.83 kB chunk-size warning
```

### Test Writing Strategy

Prefer small tests around high-risk boundaries:

- Feature order and encoding parity.
- Numeric edge cases like `hour_of_day=0` and `ev_distance=0`.
- API validation and auth behavior.
- Frontend API failure paths.
- Demo mode vs full-stack mode.

### Debugging Strategy

For model-serving issues:

1. Inspect payload validation in `api/models.py`.
2. Inspect feature transformation in `api/services/model_service.py`.
3. Compare with training feature logic in `ml/prepare_data.py`.
4. Check `ml/constants.py` and `xgboost_config.joblib` feature order.
5. Add a focused regression test before changing logic.

For API/dashboard issues:

1. Reproduce through the smallest endpoint or component.
2. Check whether the problem is API response shape, frontend parsing, or UI state.
3. Add a test at the boundary that failed.
4. Document any demo-mode differences.

## Responsible Claim Setting

Say:

- "Simulated dataset"
- "Portfolio-scale full-stack ML project"
- "Optional API-key auth"
- "Docker Compose demo environment"
- "Tests currently pass locally"
- "Static demo mode uses generated JSON and heuristic prediction"

Do not say:

- "Production-grade"
- "Deployed to a real AV fleet"
- "Guaranteed 64% reduction in real-world operations"
- "Fully secure"
- "Fully reproduced metrics" unless you have just rerun and archived the output

## Interview Narrative

The strongest story is not "I hand-wrote every line." The strongest story is:

> I took a complex, AI-assisted full-stack ML repo and made it understandable and defensible. I mapped claims to evidence, ran tests, identified weak points, and created a focused hardening backlog. That is the same ownership pattern I would apply to any inherited production codebase.
