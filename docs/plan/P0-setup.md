# P0 — Environment & scaffolding

**Output:** updated `requirements.txt`, directory skeleton
**Effort:** ~half a session
**Gate:** all four model libraries import cleanly; Jupyter kernel registered

## Objective

Get the toolchain installed and the directory structure in place, and confirm the heavier
dependencies actually build on this machine before any phase depends on them.

## Assumes

- `data/` contains the five raw M5 CSVs (gitignored, not committed).
- `venv/` exists with Python 3.11.

## Build

Add to `requirements.txt`, keeping the existing entries:

```
pyarrow          # parquet round-trips between phases
lightgbm
xgboost
statsforecast
scikit-learn
jupyterlab
ipykernel
tqdm
```

Create:

```
notebooks/                 # 01_… through 08_…, one per phase
src/                       # thin modules imported by notebooks (P3, P5 only)
reports/figures/           # saved EDA and diagnostic plots
data/interim/              # gitignored
data/processed/            # gitignored
```

Add `data/interim/` and `data/processed/` coverage to `.gitignore` (the existing `data/` entry
already covers them, but confirm — parquet files here run to several GB).

## Verify

```bash
./venv/Scripts/python.exe -m pip install -r requirements.txt
./venv/Scripts/python.exe -c "import lightgbm, xgboost, statsforecast, pyarrow; print('ok')"
./venv/Scripts/python.exe -m ipykernel install --user --name m5 --display-name "M5"
```

Likely friction points on this stack, worth surfacing now rather than in P6:

- `statsforecast` pulls `numba`, which is the most common install failure on Windows/Python 3.11.
- `lightgbm` and `xgboost` should install from wheels without needing a compiler. If either tries to
  build from source, stop and resolve it here.

## Gate

- [ ] `import lightgbm, xgboost, statsforecast, pyarrow` succeeds
- [ ] Kernel `M5` appears in `jupyter kernelspec list`
- [ ] Directory skeleton committed (with `.gitkeep` files where needed)
- [ ] Installed versions recorded in `docs/plan/DECISIONS.md`

## Invariants

Not yet applicable — no data handling in this phase. They begin at P1; see
[README](README.md#invariants).
