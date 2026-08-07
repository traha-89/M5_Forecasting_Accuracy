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

## Workflow

All changes go through a pull request — do not push directly to `main`. Use the PR template at
`.github/PULL_REQUEST_TEMPLATE.md` (type of change, changes, test plan).
