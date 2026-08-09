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
pytest           # P3's metric validation gate depends on this
```

Create:

```
notebooks/                 # 01_… through 08_…, one per phase
src/                       # thin modules imported by notebooks (P3, P5 only)
  __init__.py
tests/                     # pytest; P3 writes test_metrics.py here
  __init__.py
reports/figures/           # saved EDA and diagnostic plots
models/                    # gitignored — trained model files
data/interim/              # gitignored
data/processed/            # gitignored
```

Add `models/` to `.gitignore`. The existing `data/` entry already covers the data subdirectories,
but confirm — parquet files there run to several GB.

`reports/*.csv` are deliberately **not** gitignored: they are small, and committing them is what
lets scores survive between sessions. Only `reports/figures/` holds large files; keep image output
reasonable rather than ignoring the directory.

## The `src/` import convention

Notebooks live in `notebooks/` and import from `src/`, which does not work by default. Fix it once,
here, so all five importing notebooks do it identically. Install the repo as editable:

```bash
./venv/Scripts/python.exe -m pip install -e .
```

with a minimal `pyproject.toml` at the repo root declaring `src` as a package. Then notebooks use a
plain `from src.metrics import wrmsse` regardless of working directory.

If that proves awkward, the fallback is a two-line cell at the top of each notebook:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
```

Pick one and record it in `DECISIONS.md`. **Do not hardcode absolute paths** — they break on any
other machine, including CI.

## Verify

```bash
./venv/Scripts/python.exe -m pip install -r requirements.txt
./venv/Scripts/python.exe -c "import lightgbm, xgboost, statsforecast, pyarrow; print('ok')"
./venv/Scripts/python.exe -m ipykernel install --user --name m5 --display-name "M5"
./venv/Scripts/python.exe -m pytest --version
```

Likely friction points on this stack, worth surfacing now rather than in P6:

- `statsforecast` pulls `numba`, which is the most common install failure on Windows/Python 3.11.
- `lightgbm` and `xgboost` should install from wheels without needing a compiler. If either tries to
  build from source, stop and resolve it here.

## Gate

- [x] `import lightgbm, xgboost, statsforecast, pyarrow` succeeds
- [x] `pytest --version` succeeds
- [x] Kernel `M5` appears in `jupyter kernelspec list`
- [x] `from src import …` works from a notebook in `notebooks/` (test with a stub)
- [x] Directory skeleton committed (with `.gitkeep` files where needed)
- [x] `models/` added to `.gitignore`
- [x] Installed versions and the chosen import convention recorded in `docs/plan/DECISIONS.md`

Passed 2026-08-09 (PR #7). See `docs/plan/DECISIONS.md` (`## P0 — Environment & scaffolding`) for
the recorded evidence.

## Invariants

Not yet applicable — no data handling in this phase. They begin at P1; see
[README](README.md#invariants).
