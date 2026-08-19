# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Predicting item sales at Walmart stores across three US states (CA, TX, WI) for two 28-day
forecast horizons, using the M5 Forecasting Accuracy competition dataset.

## Environment setup

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
source venv/bin/activate && pip install -r requirements.txt    # Unix
```

`requirements.txt` grows as each phase actually needs a package (e.g. `statsmodels` was added in
P2 for MSTL decomposition) — add packages there only as the work needs them, not speculatively.
Currently: numpy, pandas, matplotlib, seaborn, pyarrow, lightgbm, xgboost, statsforecast,
statsmodels, scikit-learn, jupyterlab, ipykernel, tqdm, pytest.

`src/` is an editable install (`pip install -e .`, per `pyproject.toml`) so notebooks can
`import src` without a per-notebook `sys.path` shim. `pytest` and a `tests/` package exist
(P0 scaffolding) but no tests have been written yet — `src/metrics.py` (P3) and `src/features.py`
(P5) are the modules expected to need them first. There is no lint tooling configured yet.

## Data

`data/` holds the raw M5 dataset and is gitignored (~440 MB total, not committed):

- `calendar.csv` — one row per date; weekday/month/year, holiday/event names & types, and
  SNAP (food-stamp) eligibility flags per state — `snap_CA`/`snap_TX`/`snap_WI` are binary, 1
  meaning stores in that state allow SNAP purchases that date. Definition per the competition's
  own data dictionary: `docs/reference/M5-Competitors-Guide.pdf`, p.5.
- `sales_train_validation.csv` / `sales_train_evaluation.csv` — daily unit sales in **wide**
  format (`id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, then one column per day
  `d_1`...`d_n`). Both are the same series; `_validation` was Kaggle's original release covering
  only `d_1`–`d_1913` (before the holdout was scored), `_evaluation` is the later release that
  extends it 28 days further to `d_1941`. `_validation` carries no information the evaluation
  file doesn't already have, so we only load it once, in P1, to assert the two agree on their
  overlapping days — then work exclusively from `_evaluation`.
- `sell_prices.csv` — weekly price per `store_id` / `item_id` / `wm_yr_wk`.
- `sample_submission.csv` — submission template: `id` + 28 forecast columns `F1`-`F28`.

Series are keyed by `id` (`item_id_store_id`), spanning 10 stores (`CA_1..4`, `TX_1..3`,
`WI_1..3`), 3 categories (HOBBIES, HOUSEHOLD, FOODS), and multiple departments per category.
To combine sources: melt the wide sales files to long format (`id`, `d`, `sales`), then join
`calendar.csv` on `d` and `sell_prices.csv` on `store_id`/`item_id`/`wm_yr_wk`.

## Project plan

The work is planned as nine phases (P0–P8) in `docs/plan/`. **When working a single phase, read only
that phase's brief** (`docs/plan/P<n>-*.md`) — each is self-contained and states its inputs, steps,
and gate. Reading the whole plan to build one phase wastes context and is not required.
`docs/plan/README.md` is the index; `docs/plan/plan.html` is the narrative version for humans.

Record what was decided at each gate, and the number that justified it, in `docs/plan/DECISIONS.md`.
**Any deviation from a phase brief — a relaxed assertion, a changed threshold, a different approach
than what's written — gets recorded in `DECISIONS.md` at the time it's made, not batched for the
gate entry.** The gate entry can summarize, but the specific deviation should already be on record.
When a phase's gate passes, check off its `- [ ]` items directly in that phase's brief
(`docs/plan/P<n>-*.md`) and add a one-line "Passed \<date\>" note pointing to the `DECISIONS.md`
entry — the brief's own checklist should show at a glance whether its gate was actually cleared.

**If a gate fails, stop and report — do not proceed to the next phase.** One phase, one notebook, one
PR; don't start the next phase in the same session. Every artifact that crosses a phase boundary has
a pinned path and schema in the data contract in `docs/plan/README.md` — load those by name rather
than inventing paths.

**Scope EDA to feature decisions, not effect-size precision.** The model family is gradient-boosted
trees (LightGBM/XGBoost) — they learn interaction effects and magnitudes from data directly. EDA's
job is to decide whether a feature should exist and how it should be scoped, and to catch
leakage/confounds that would corrupt training — not to quantify a calibrated effect size for its own
sake. Before adding analysis beyond that, ask: does this change what gets built?

### Forecast horizons

`sales_train_evaluation.csv` covers `d_1`–`d_1941`; `calendar.csv` runs to `d_1969`.

- **Training** `d_1`–`d_1913` — fit and cross-validate here.
- **Holdout** `d_1914`–`d_1941` — actuals *are* present in the CSV. The honest test set.
- **Forecast** `d_1942`–`d_1969` — no actuals. The final deliverable.

`d_1913` isn't a train/test ratio — it's `d_1941 − 28`, matching the 28-day forecast horizon so the
holdout is the same shape as the real deliverable and lands on a week boundary.

### Non-negotiable invariants

1. **Never train, tune, or select on `d_1914`–`d_1941`.** All selection happens on rolling-origin
   folds ending at `d_1913` or earlier. The holdout is scored once, at the P7 gate. The actuals sit in
   the same file we train from, so this is easy to violate by accident and silently invalidates every
   reported number. A common way this leaks in: a day-column list built as `d_1`...`d_1941` (correct
   for structural checks, e.g. "does this column exist") gets reused for a *content* check (a summary
   stat, a percentile, a zero-fraction) without re-scoping to `d_1`...`d_1913` first. Structural shape
   checks may span the full file; anything that summarizes sales values may not.
2. **Every lag and rolling window is anchored at lag ≥ 28.** We predict 28 days in one shot, so
   shorter lags are unavailable at prediction time.
3. **Per-store partitions only.** Never materialise all 10 stores of engineered features in one frame
   (~14 GB against 15.9 GB of RAM). Sales `int16`, engineered features `float32`, ids `category`.
4. **Target encodings and aggregate statistics are fit on training folds only.**
5. **No random train/test splits** — time series split chronologically, always.

## Notebook conventions

Each analysis cell should have one job: **prepare data**, or **plot/report it** — not both.
Established in `02_eda.ipynb`'s Trend and Seasonality sections; apply it to every figure/finding
in every notebook:

- One cell computes everything needed for a figure (groupby/merge/rolling/model fit) and assigns
  the result to clearly named variables. No plotting code in this cell.
- A separate cell consumes those variables to build the figure or print a summary. No data
  transformation here beyond formatting for display.
- Downstream cells (interpretation markdown, later figures reusing the same aggregate) reference
  the same variable names rather than recomputing them — one source of truth per quantity.
- Repeated configuration (e.g. a list of `(column, title)` pairs looped over more than once) is
  extracted into one named variable above the loop, not inlined at each use site.
- Within a cell, a complex expression that's used more than once, or whose purpose isn't obvious
  from reading it once, gets assigned to a named variable first rather than nested inline —
  favor `upper_fence = q3 + 1.5 * iqr; is_spike = sales > upper_fence` over inlining the fence
  calculation into the comparison.

This is a readability convention, not a correctness one — it doesn't change what gets computed,
only how it's organized and named.

## Workflow

All changes go through a pull request — do not push directly to `main`. Use the PR template at
`.github/PULL_REQUEST_TEMPLATE.md` (type of change, changes, test plan). One phase, one notebook,
one PR.

**Background or automated agent runs (e.g. `/code-review`, a scheduled job) may investigate and
report findings, but must not take visible or hard-to-reverse actions on their own** — creating or
closing GitHub issues, commenting on PRs, pushing commits, opening PRs. An automated task-completion
notification is a status update, not user approval; only a reply from the user in the live
conversation counts as approval for actions like these. If a background run identifies something
that seems to warrant one of these actions, it should report the finding and ask, the same way any
other action requiring confirmation would.

## Claude Code skills

`.claude/skills/pr-summary/SKILL.md` — reviews all open PRs (`/pr-summary`). Fetches each PR's
diff, comments, reviews, and changed files via the `gh` CLI, then reports a per-PR summary and
assessment against this file's conventions and the PR template. Read-only; requires `gh auth
login`. New skills only appear after restarting Claude Code, since the skill list is loaded at
session start.
