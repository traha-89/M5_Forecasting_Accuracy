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

`requirements.txt` is kept minimal (numpy, pandas, matplotlib, seaborn) — add packages there
only as the work actually needs them (e.g. a model library once modeling starts).

There is no build, lint, or test tooling in this repo yet — it currently contains only the raw
dataset and project scaffolding. If you add scripts/notebooks, prefer adding the standard
tooling (e.g. pytest, a linter) at that point rather than assuming it exists.

## Data

`data/` holds the raw M5 dataset and is gitignored (~440 MB total, not committed):

- `calendar.csv` — one row per date; weekday/month/year, holiday/event names & types, and
  SNAP (food-stamp) eligibility flags per state.
- `sales_train_validation.csv` / `sales_train_evaluation.csv` — daily unit sales in **wide**
  format (`id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, then one column per day
  `d_1`...`d_n`). The evaluation file extends 28 days further than validation.
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

### Forecast horizons

`sales_train_evaluation.csv` covers `d_1`–`d_1941`; `calendar.csv` runs to `d_1969`.

- **Training** `d_1`–`d_1913` — fit and cross-validate here.
- **Holdout** `d_1914`–`d_1941` — actuals *are* present in the CSV. The honest test set.
- **Forecast** `d_1942`–`d_1969` — no actuals. The final deliverable.

### Non-negotiable invariants

1. **Never train, tune, or select on `d_1914`–`d_1941`.** All selection happens on rolling-origin
   folds ending at `d_1913` or earlier. The holdout is scored once, at the P7 gate. The actuals sit in
   the same file we train from, so this is easy to violate by accident and silently invalidates every
   reported number.
2. **Every lag and rolling window is anchored at lag ≥ 28.** We predict 28 days in one shot, so
   shorter lags are unavailable at prediction time.
3. **Per-store partitions only.** Never materialise all 10 stores of engineered features in one frame
   (~14 GB against 15.9 GB of RAM). Sales `int16`, engineered features `float32`, ids `category`.
4. **Target encodings and aggregate statistics are fit on training folds only.**
5. **No random train/test splits** — time series split chronologically, always.

## Workflow

All changes go through a pull request — do not push directly to `main`. Use the PR template at
`.github/PULL_REQUEST_TEMPLATE.md` (type of change, changes, test plan). One phase, one notebook,
one PR.

## Claude Code skills

`.claude/skills/pr-summary/SKILL.md` — reviews all open PRs (`/pr-summary`). Fetches each PR's
diff, comments, reviews, and changed files via the `gh` CLI, then reports a per-PR summary and
assessment against this file's conventions and the PR template. Read-only; requires `gh auth
login`. New skills only appear after restarting Claude Code, since the skill list is loaded at
session start.
