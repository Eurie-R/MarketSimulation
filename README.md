# Market ABM: transaction taxes and trader heterogeneity

An agent-based model of a stock market (one synthetic asset, a continuous double-auction limit
order book, fundamentalist / chartist / noise traders) used to study a seller-side transaction
tax. The motivating policy is the Philippine stock transaction tax cut from 0.6% to 0.1% of the
selling price (RA 12214, July 2025).

**Results describe the model's mechanism, not a forecast for the Philippine Stock Exchange.**

- Full specification: [`CLAUDE.md`](CLAUDE.md)
- Model description (ODD protocol): [`docs/ODD.md`](docs/ODD.md)
- Results write-up: [`docs/results.md`](docs/results.md)

---

## 1. Install

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
pip install uv                      # skip if uv is already installed
python -m uv sync                   # creates .venv with all dependencies from uv.lock
```

Every command below uses `python -m uv run ...`. If `uv` is on your PATH, `uv run ...` is the same.

## 2. Test

```bash
python -m uv run pytest -q                         # ~20 s, 102 tests
python -m uv run ruff check . && python -m uv run ruff format --check .
```

## 3. Reproduce the results

All experiment outputs go to `results/<experiment>/` (not tracked by git):
`runs.parquet` (one row per run), `analysis/` (tables and `hypotheses.md`) and `figures/`
(PNG at 200 dpi and PDF).

The experiments use the calibrated parameter set `configs/calibrated.yaml`, which is in the
repository, so **E2, E2b, E3 and E4 need no data download and no calibration run**.

### Reproduce figure F3 (E2: tax × trader mix)

```bash
python -m uv run python scripts/run_experiment.py --config configs/e2_tax_mix.yaml
python -m uv run python scripts/analyze.py --experiment e2_tax_mix
python -m uv run python scripts/make_figures.py --experiment e2_tax_mix
```

F3 is then at `results/e2_tax_mix/figures/F3_heatmaps.png` (F4 and F5 are next to it).
Use `--n-jobs N` on `run_experiment.py` to limit CPU cores (default: all) and `--dry-run` to list
the runs without executing them.

### All experiments

Each experiment is the same three commands with its name:

| Experiment | Config | Runs | Figures | Time on 11 cores (one core) |
|---|---|---|---|---|
| E2 tax × trader mix (H1, H2, PH cut) | `configs/e2_tax_mix.yaml` | 600 | F3, F4, F5 | ~7 min (~75 min) |
| E2b tax × market thinness (H3) | `configs/e2b_thin_market.yaml` | 300 | F6 | ~3 min (~35 min) |
| E3 tax × strategy switching | `configs/e3_adaptive.yaml` | 300 | F7 | ~4 min (~40 min) |
| E4 Sobol sensitivity of the PH cut | `configs/e4_sensitivity.yaml` | 1,024 | F8 | ~12 min (~2 h) |

```bash
for e in e2_tax_mix e2b_thin_market e3_adaptive e4_sensitivity; do
  python -m uv run python scripts/run_experiment.py --config configs/$e.yaml
  python -m uv run python scripts/analyze.py --experiment $e
  python -m uv run python scripts/make_figures.py --experiment $e
done
```

One simulation (T = 50,000 steps, N = 1,000 agents) takes about 5–10 s.

### E1: calibration (optional, needs data)

`configs/calibrated.yaml` is the output of E1. To redo it:

```bash
python -m uv run python scripts/fetch_psei.py                 # empirical target (see below)
python -m uv run python scripts/run_calibration.py --stage 1  # 1,728 points x 3 seeds, ~75 min on 11 cores
python -m uv run python scripts/run_calibration.py --stage 2  # top points x 30 seeds, ~10 min
python -m uv run python scripts/run_calibration.py --stage report
python -m uv run python scripts/e1_compromise.py              # best-compromise point -> configs/calibrated.yaml
python -m uv run python scripts/make_figures.py --experiment e1_calibration   # F1, F2
```

No grid point passes all five validation checks, so the final parameter set comes from
`e1_compromise.py` (see `CLAUDE.md` §12.4 and `docs/results.md`).

**Empirical data.** Yahoo Finance has no history for the PSEi ticker `PSEI.PS`, so
`fetch_psei.py` downloads the iShares MSCI Philippines ETF (**EPHE**, daily, 2015 onward) instead.
To use real PSEi data, place a CSV with columns `Date,Close` at `data/raw/psei.csv` and run
`fetch_psei.py`; the CSV takes priority. F2 needs these data; the other figures do not.

## 4. Notebook sandbox

```bash
python -m uv run jupyter lab notebooks/sandbox.ipynb
```

Run all cells, then:
1. **Single run:** set the tax, trader shares, N, T, seed and switching, then press *Run*.
   It shows price vs fundamental value, returns, spread, volume, who trades, and a stylized-facts
   card with pass/fail marks.
2. **Compare two tax rates:** the same seed at two taxes (default 0.6% vs 0.1%). Because all
   randomness comes from fixed per-purpose streams, any difference is caused by the tax.
3. **Live market:** watch prices, the order book and trading activity update in real time
   (Play / Pause / Reset).

## 5. Layout

```
configs/      baseline.yaml (defaults), calibrated.yaml (E1 output), one YAML per experiment
src/marketabm/
  config.py       parameter validation, YAML loading, overrides
  rng.py          the five named random streams (common random numbers)
  orderbook.py    limit order book and matching engine
  agents.py       agent types, parameters, decision rule
  market.py       the simulation loop: run(config, seed), Simulation (step-by-step), live view
  switching.py    Brock–Hommes strategy switching (E3)
  results.py      SimulationResult and its summary metrics
  metrics.py, stylized_facts.py   metric definitions
  empirical.py, calibration.py    E1
  experiments.py, analysis.py     running grids, paired differences, hypothesis checks, Sobol
  viz.py          all figures and notebook widgets
scripts/      fetch_psei, run_calibration, e1_compromise, run_experiment, analyze, make_figures
notebooks/    sandbox.ipynb
tests/        pytest suite
docs/         ODD.md, results.md, figures/ (copies of F1-F8 used by results.md)
data/, results/   generated, not tracked
```

## 6. Run a single simulation from Python

```python
from marketabm.config import load_model_config, with_overrides
from marketabm.market import run

cfg = with_overrides(load_model_config("configs/calibrated.yaml"), {"tau": 0.006})
r = run(cfg, seed=1000)
print(r.summary()["vol"], r.summary()["mispricing"])
```

Same config and seed always give identical results, serially or in parallel.
