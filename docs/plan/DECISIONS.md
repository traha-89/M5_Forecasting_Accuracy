# Decisions log

Append-only. At each phase gate, record **what was decided and the number that justified it.**

This is what carries context between sessions. Without it, later phases re-derive choices
inconsistently — which is how the pilot and the full run end up with different feature definitions
and an invalid comparison. Keep entries short; link to the notebook or figure for detail.

Format:

```
## P<n> — <phase name>            (YYYY-MM-DD)
- **Decision:** …
- **Evidence:** … (number, figure, or notebook cell)
```

---

## Pre-P0 — plan decisions (2026-08-08)

Made when the plan was drafted, before any code.

- **Model shortlist:** LightGBM Tweedie, LightGBM Poisson, Nixtla StatsForecast ensemble, XGBoost.
  **Rationale:** best accuracy-per-hour on a CPU-only 8-core box; the Tweedie/Poisson pair isolates
  the objective's contribution, XGBoost guards against a LightGBM-specific artifact, and
  StatsForecast provides the only per-series arm for contrast.
- **Sequencing:** pilot the full pipeline on store `CA_1`, then scale to all ten.
  **Rationale:** iteration on 3,049 series takes minutes; on 30,490 it takes hours. Design errors get
  caught cheaply.
- **Forecast granularity:** global gradient-boosted models **partitioned by store** (10 models,
  ~3,049 series each), not one model per series.
  **Rationale:** per-series fitting discards transferable structure (SNAP, events, price response) and
  30,490 fits is infeasible here; the store seam is principled because SNAP schedules are
  state-specific and assortment differs by store. Also fits memory: ~1.4 GB per partition vs ~14 GB
  for all ten.
- **Deliverable form:** numbered Jupyter notebooks per phase, plus thin `src/` modules for metrics
  (P3) and features (P5) only.
  **Rationale:** those two are imported by four to five downstream notebooks each; copy-pasting them
  causes silent drift between the pilot and the full run.
- **Excluded metrics:** sMAPE and MAPE. **Rationale:** both divide by actuals; ~68% of this dataset is
  zero, so they are undefined or explosive.

### Still open

- **Outlier policy** — decided in P2 on evidence. Note the tension: clipping promotional spikes will
  improve RMSSE while making the forecast less useful operationally.
- **Prediction intervals** — out of scope. M5 had a companion Uncertainty track using pinball loss
  across nine quantiles; LightGBM can produce these with quantile objectives at ~9× training cost.
- **Horizon strategy** — single non-recursive model with lag ≥28 is the plan; the 4×7-day block
  alternative is tested in P6 and decided there.
- **Segment routing** — whether to route different models to different ADI/CV² quadrants. Decided in
  P7, only if the breakdown supports it.

---

## P0 — Environment & scaffolding            (2026-08-09)

- **Decision:** `src/` import convention is an editable install (`pip install -e .`) backed by a
  minimal `pyproject.toml` declaring `src` as a package, not a per-notebook `sys.path` shim.
  **Evidence:** `import src` resolves correctly to `src/__init__.py` when run with `notebooks/` as
  the working directory, confirmed via `cd notebooks && python -c "import src; print(src.__file__)"`.
- **Decision:** directory skeleton (`notebooks/`, `src/`, `tests/`, `reports/figures/`, and gitignored
  `models/`, `data/interim/`, `data/processed/`) created as specified in P0-setup.md.
  **Evidence:** all four model libraries import cleanly, `pytest --version` succeeds, `M5` kernel
  registered in `jupyter kernelspec list`, `from src import …` verified from `notebooks/`.
- **Installed versions:** lightgbm 4.7.0, xgboost 3.2.0, statsforecast 2.1.1, pyarrow 25.0.0,
  scikit-learn 1.9.0, pandas 2.3.3, numpy 2.4.6. No source builds required — all installed from
  wheels on Windows/Python 3.11.
- **Note:** installing `statsforecast` downgraded `pandas` from 3.0.5 to 2.3.3 to satisfy its pin.
  No other packages in this repo depend on pandas 3.x features, so left as resolved by pip.

<!-- Append phase entries below as gates are passed. -->
