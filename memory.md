# Session memory: Market ABM (transaction tax × trader heterogeneity)

Use this file to resume work. Read `CLAUDE.md` (the full spec, now v1.1) first, then this file.
Last updated: 2026-11-18.

---

## 1. Where we are

| Phase (CLAUDE.md §15) | Status |
|---|---|
| 1. Scaffold | **Done** |
| 2. Order book | **Done** |
| 3. Agents, tax, loop | **Done** (plus the v1.1 stability fixes, see §3) |
| 4. Metrics and calibration (E1) | **Done**: v1.2 E1 chose point 139 → `configs/calibrated.yaml` (see §8) |
| 5. Experiments | **Code done** (experiments.py, analysis.py, run_experiment/analyze/make_figures scripts, e2/e2b/e4 configs, F1-F6, F8). E2, E2b, E4 runs launched 2026-11-18 |
| 6. Sandbox and switching | **Partly done**: the notebook sandbox and `switching.py` exist. E3 beta levels have **not** been chosen |
| 7. Documentation | Not started (README.md is the empty uv stub) |

**Next step:** see §10 (TO-DO, paused 2026-11-18 at the user's request). Start there.

**Status checks:** `python -m uv run pytest -q` gives 79 passed, 2 skipped (the skipped tests wait
for Phase 5 code). `python -m uv run ruff check .` passes.

---

## 2. Environment

- Windows 10, Python 3.12.10. The project sits at the repo root, **not** in a `market-abm/`
  subfolder (an ASSUMPTION; CLAUDE.md is at the root).
- `uv` is **not on PATH**. It was installed with `pip install --user uv`, so run everything as
  `python -m uv run ...`, for example:
  - Tests: `python -m uv run pytest -q`
  - Lint: `python -m uv run ruff check . && python -m uv run ruff format .`
  - Notebook: `python -m uv run jupyter lab notebooks/sandbox.ipynb`
- The package is `marketabm` (src layout). `pyproject.toml` sets
  `[tool.uv.build-backend] module-name = "marketabm"`.
- `git init` was run, but **nothing has been committed**.

---

## 3. The key decision: stability fixes (v1.1, approved by the user)

**Problem.** The original spec rule diverged (price overflow) on every seed whenever chartists were
present, even at a 10% chartist share. Experiments found two causes:

1. **Explosion.** Orders were priced at the full forecast `p_hat`, often far from the market. A
   resting order trades at its own limit price, so one far-away chartist order sets an extreme
   price, which feeds the chartists' trend signal. These fixes did **not** stop it: `h = L`,
   pre-filling the book, recording the mid price, smaller `sigma_eps`, or g2 in [0.4, 0.6].
2. **Runaway bubbles at 40–55% chartists.** Fundamentalist reversion was weak (`h/tau_f` ≈ 0.55)
   compared with chartist extrapolation (`h/L` ≈ 2.8). Unit orders also made the market a head
   count, so a big mispricing could not outweigh a chartist majority.

**Fixes (implemented, and written into CLAUDE.md §5.5 and the new §5.9):**

| Config key | Default | Meaning |
|---|---|---|
| `price_band` | `0.01` | Order price = `p * exp(clip(r_hat*h, ±band)) * (1 ∓ k)`. It bounds only the order price; the forecast and the tax filter use the unclipped gain. `null` gives the v1 rule. |
| `tau_f` | `10` (was 100) | Stronger fundamentalist reversion. |
| `order_size` | `signal` | `q = min(size_max, max(1, ceil(gain/size_unit)))`; `unit` means q = 1. |
| `size_unit` | `0.01` | |
| `size_max` | `20` | |

**Result** (50,000-step runs, seeds 1–3, about 6 s per run):

| Chartist share | Price range | Mispricing | Validation targets passed |
|---|---|---|---|
| 0.10 | 68–116 | 0.005 | 1/4 |
| 0.25 | 67–119 | 0.006 | 1–2/4 |
| 0.40 | 66–122 | 0.011 | 2/4 |
| 0.55 | 34–260 | 0.21 (bounded bubbles, an intended H2-relevant outcome) | 0/4 |

`tests/test_market.py::test_original_rule_diverges` records that the v1 rule still diverges.

---

## 4. Open issues and risks (raise these with the user)

1. **Lag-1 return autocorrelation is about −0.4 at the defaults** (bid/ask bounce of the
   last-trade price). It fails the §12.3 target `|acf_r_lag1| < 0.1`. E1 may not be able to fix
   it. Mid-price recording was tested and did not help. Noted as a known risk in CLAUDE.md §12.4
   step 8.
2. **The E1 grid grew to 1,728 points** (`tau_f` ∈ {10, 25, 50} and `price_band` ∈ {0.005, 0.01,
   0.02} were added), which is about 5,184 runs and about 9 CPU-hours. We offered to trim one
   dimension (for example `order_life` to 3 values) if that is too slow. The user has not answered.
3. **E3 beta levels have not been chosen.** By spec they must be picked in Phase 6, after
   inspecting the size of `pi_k`. The sandbox β box (default 1.0) is a sandbox-only placeholder.
4. **PSEi ticker `PSEI.PS` is unverified.** If the download fails, follow §12.1: ask the user for
   `data/raw/psei.csv`.

---

## 5. Assumptions made (each is marked `# ASSUMPTION` in the code)

- The project is at the repo root rather than in `market-abm/`.
- `switching.enabled: true` with `beta: null` is rejected by validation.
- With switching on, the composition metrics count agents by their **final** types.
- Sandbox: `T_burn = 10% of T`, and T and T_burn are rounded to multiples of `return_interval`.
- Sandbox: the β default of 1.0 is a placeholder, not an E3 level.
- `volume_share_k` and `fills_per_agent_k` count **participations** (one per fill, whatever its
  quantity); `volume_total` counts units. Now stated in CLAUDE.md §11.5.

---

## 6. Code map (what exists)

- `src/marketabm/config.py`: pydantic `ModelConfig` (frozen, extra keys forbidden),
  `ExperimentConfig`, `with_overrides(cfg, dict)` (handles `chartist_share` and nested merges),
  `mix_from_chartist_share`, `config_hash`, `load_model_config`, `load_experiment_config`,
  `resolve_base`.
- `src/marketabm/rng.py`: `make_streams(seed)` returns the streams fundamental, init,
  activation, noise, switching (spawn order fixed).
- `src/marketabm/orderbook.py`: heaps with lazy deletion, an expiry deque, heap compaction,
  `expire_orders(t)` (returns the removed orders), `is_live(order_id)`, plus the API from §6.3.
- `src/marketabm/agents.py`: `FUND/CHART/NOISE`, `init_agents` (draw count independent of the
  mix), and `decide(...)`, which returns `(side | SKIP_TAX | SKIP_NO_SIGNAL, price_ticks, qty)`.
- `src/marketabm/market.py`: `run(config, seed, *, debug=False)`. With `debug=True` it checks
  money and unit conservation and at most one live order per agent after every step. Also has
  `fundamental_path`, `draw_activation`, `draw_noise` (pre-drawn, for common random numbers),
  `Ledger` (seller-side tax), `_Market`, `_Recorder`, and `run_metadata` (config hash, git commit,
  versions).
- `src/marketabm/results.py`: `SimulationResult` (adds `snapshot_t`, `switch_t`, and
  `agent_params` with the initial parameters), `summary()` (all §11 metrics, never raises),
  `series_frame()`, `save_series()`.
- `src/marketabm/metrics.py`, `stylized_facts.py`: §11 metrics, plus `moments()` and
  `validation_checks()` on standardized returns.
- `src/marketabm/switching.py`: Brock–Hommes logit switching. It uses a fixed number of draws per
  round (one type draw plus 3 weight draws per non-noise agent).
- `src/marketabm/viz.py`: matplotlib panels, `plot_run`, `plot_compare`, `sandbox_config`,
  `safe_run`, `sandbox_ui`, `compare_ui`. Type colors: fundamentalist `#2a78d6`, chartist
  `#eb6834`, noise `#1baf7a`.
- `notebooks/sandbox.ipynb`: a thin notebook with two sections (single run; compare two tax
  rates on the same seed, defaulting to 0.6% vs 0.1%).
- `src/marketabm/{empirical,calibration,experiments,analysis}.py`: placeholders.
- `scripts/`, `docs/`: empty.
- `tests/`: config, orderbook (with hypothesis invariants), agents, tax, market, reproducibility,
  metrics, analysis. The `small_stable` fixture is the baseline mix with T=2000, T_burn=500, N=100.

---

## 7. How to resume

1. Read `CLAUDE.md` and this file.
2. Run `python -m uv run pytest -q` to confirm the baseline state (66 passed, 2 skipped).
3. Ask the user about open issues 1–2 (the autocorrelation risk and the E1 grid size), then start
   Phase 4: `empirical.py` + `scripts/fetch_psei.py`, then `calibration.py` (parallelize with
   joblib/loky and tqdm), then run E1 and write `configs/calibrated.yaml` and
   `results/e1_calibration/report.md`.
4. After each phase: run pytest and ruff, report (built / assumptions / anything off), and wait
   for confirmation.

---

## 8. Session 2026-11-18: E1, data source, live viewer

**Data.** Yahoo has no history for `PSEI.PS` (returns only today's quote). The user approved a
substitute; we use **EPHE** (iShares MSCI Philippines ETF), 2,986 daily returns 2015-01-05 to
2026-11-17. `data/processed/source.json` records the source; a user CSV at `data/raw/psei.csv`
still takes priority. Noted in CLAUDE.md §12.1. EPHE moments (standardized): kurtosis 38.3 (boot
sd 19.9, 2020 crash), Hill 3.22, acf1 -0.100 (just fails |acf1|<0.1 itself), acf|R| 1-10 0.200,
11-50 0.051.

**E1 result: no point passes 12.3.** Stage 1 (1,728 x 3 seeds, ~75 min on 11 cores) pass counts:
kurt>1 705, |acf1|<0.1 158, acf|R|>band lags 1-20 **2**, hill 1106; all four: 0. Stage 2 (top 10
by D x 30 seeds) also 0. Report: `results/e1_calibration/report.md`.
- Binding constraint is volatility clustering, not the bid-ask bounce. Clustering only appears at
  grid edges (k_max 0.01, price_band 0.005, order_life 50) with tau_f 50, where acf1 is +0.3..0.45
  (trending). At tau_f 25 acf1 is ~+0.07 but only 3-6 of 20 lags clear the band.
- Pilot beyond the grid (`results/e1_calibration/pilot_frontier.csv`): tau_f 30/35/40 trade acf1
  +0.18/+0.26/+0.28 for 12/18/13 lags; larger k_max (0.015, 0.02) or band 0.0025 make it worse.
- Conclusion: in this model clustering and positive return autocorrelation come from the same
  chartist trend episodes. Options put to the user: (A) finer grid near tau_f 25-30; (B) change the
  12.3 clustering target; (C) add a clustering mechanism to the model. **User chose (C).**

**v1.2: volatility feedback (CLAUDE.md §5.10).** `E = eps * m_t`, `m_t = clip((s_fast/s_slow)^gamma,
1/cap, cap)`, EWMA vols of one-step returns (half-lives 20 / 2000), no new random draws, off by
default (`vol_feedback.enabled: false` = exact v1.1, tested). `SimulationResult.vol_mult` records m_t.
- Pilots: the feedback alone did little at price_band 0.005 (the band pins order prices). At band
  0.00625, tau_f 30 (anchor = v1 point 102: sigma_eps 0.0002, order_life 50, k_max 0.01,
  g2 [1.2, 1.8]) gamma 2-3 passes all four checks (3 seeds): kurt ~2.2, Hill ~5.0-5.3, acf1 ~+0.03,
  20/20 lags. gamma 0 there: 19 lags but Hill 6.02 (fails) - feedback mainly fattens tails.
  Stable at chartist 0.10 (mispricing 0.002) and 0.55 (0.14, acf1 +0.79 trending).
- v1.2 E1 grid (1,728 pts): sigma_eps {.0002,.0005,.001}, order_life {50,100,200}, k_max {.005,.01},
  g2 {[.8,1.2],[1.2,1.8]}, tau_f {25,27.5,30,32.5}, band {.005,.00625,.0075}, gamma {0,1,2,3}
  (grid key `vol_feedback_gamma`). v1 results archived in `results/e1_calibration/v1_grid/`.

**Live viewer (user request).** `market.Simulation` (now used by `run()`; `advance(n)`,
`frame(window)`), `market.live()`, `LiveFrame`, `OrderBook.levels(n)`, and `viz.LiveViewer` /
`viz.live_ui(base)` (ipywidgets Image updated from a background thread). Notebook section 3.
Render ~1 s/frame under load (~340 ms is matplotlib's fixed cost).

**New assumptions:** E1 driver is its own script; calibration seeds 1-3 / 100-129 (apart from
experiment seeds); "averaged" in 12.3 = average each statistic over seeds, then test; downloads
under 250 rows count as failed; stage-2 "mean D" = D of the 30-seed mean moments.

**E1 v1.2 result.** Stage 1: kurt 792, acf1 268, clustering 92, hill 905, all four 8 points.
Top 10 by 3-seed D all failed at 30 seeds (mostly clustering at lags 15-20). Re-running the 8
stage-1 passers with 30 seeds: 4 pass (139, 174, 138, 175). Chosen **point 139**: sigma_eps 0.0002,
order_life 50, k_max 0.01, chartist g2 [0.8, 1.2], tau_f 32.5, price_band 0.00625,
vol_feedback gamma 3 (enabled). 30-seed: kurt 4.16, Hill 4.36, acf1 +0.030, acf|R| 1-10 0.341,
11-50 0.038; D 15.89. Several values sit on grid edges (sigma_eps, order_life, k_max, tau_f, gamma).
gamma 0 at the same point (136) fails clustering at lags 18-20, so the feedback matters.
Stage-2 rule changed to "top 10 by D + all stage-1 passers" (CLAUDE.md §12.4 step 6 note;
ASSUMPTION in scripts/run_calibration.py). Files: stage2_top10_runs.parquet (original top 10),
stage2b_passers_runs.parquet, stage2_runs.parquet (merged), report.md.
Calibrated stability (seed 1000, tau 0.6%): mispricing 0.0035 / 0.018 / 0.056 / 0.086 at chartist
0.10 / 0.25 / 0.40 / 0.55; ~4-5 s per run.

**Phase 5 assumptions:** H2 needs vol and mispricing CIs < 0 at the same tau (supported = both
shares); H3 supported = vol up + spread up + depth down at one tau; Sobol NaN outputs replaced by
the mean (count reported); ExperimentConfig gained an optional `sobol:` block; F5/F6 tau axes are
categorical (evenly spaced).

## 9. Problem found 2026-11-18: calibrated model cycles

F2 showed the model's return ACF alternating at short lags. Measured over seeds 1000-1009:
- calibrated (139): ACF R lags 1-8 = +0.04 -0.69 -0.31 +0.46 +0.39 -0.24 -0.39 +0.07;
  ACF |R| lags 1-8 = 0.39 0.60 0.37 0.39 ...; spectral peak of step returns at 83-96 steps,
  50-240x the mean power, in every seed.
- same point with feedback off: same cycle (so it is the parameter region, not vol_feedback).
- v1.1 baseline: no cycle (only the lag-1 bounce, -0.44).
So the 12.3 checks were passed for the wrong reason: acf1 ~ 0 because 20 steps is about a quarter
of the cycle, and the |R| "clustering" is mostly the cycle. 12.3 only checks lag 1.
Phase 5 ran anyway (E2 8 min, E2b 4 min, E4 14 min; all outputs in results/). Provisional
verdicts: H1 not supported (0/16 cells), H2 supported, H3 supported; PH cut effect on vol flips
sign: cut lowers vol at 10-25% chartists, raises it at 40-55%. E4: sigma_f dominates; ST > 1
(noise from one seed per point). Do not report these as findings until the cycle is resolved.

**User's decision (2026-11-18):** "Add cycle test, re-run E1". Done so far: a fifth 12.3 check
`|acfR 2-5|<0.1` (|ACF of R| < 0.1 at lags 2-5) in `calibration.checks_from_means` and
`stylized_facts.validation_checks`; per-run `acf_r_2..5` stored by `calibration.return_stats`;
tests updated (94 pass, ruff clean). Verified: point 139 fails it (acf_r_2 -0.71), v1.1 baseline
passes it. **CLAUDE.md §12.3 is NOT yet updated with the fifth check** (do this first).

**Cycle-map pilot** (`results/e1_calibration/pilot_cycle_map.csv`; anchor = point 139 with
tau_f {10,15,20,25} x band {.005,.00625,.0075,.01} x order_life {50,200}, 3 seeds):
- Cycles (acf_r_2 <= -0.15) grow with tau_f and appear at tau_f >= 15 with band <= 0.00625;
  wider bands (0.0075-0.01) remove the cycle (acf_r_2 ~ 0) but bring back the lag-1 reversal
  (acf_r_lag1 -0.2 to -0.6).
- Clustering (lags above band) needs band >= 0.0075 or tau_f >= 20.
- **No pilot point passes all 5 checks.** Best 4/5: tau_f 15 or 20, band 0.01, order_life 200
  (20/20 lags, no cycle, kurt 6-8, Hill 3.0-3.1) but acf1 -0.38/-0.50; and tau_f 10-15, band
  0.00625, order_life 200 (acf1 ok, no cycle) but no clustering (2-3 lags).
- So the remaining conflict is lag-1 reversal vs clustering, now without the cycle loophole.

## 10. TO-DO (resume here)

Done 2026-11-19: CLAUDE.md 12.3/12.4 updated; v1.2 E1 outputs archived to
results/e1_calibration/v1_2_cycle/; old E2/E2b/E4 results moved to results/provisional_cycle/.
Mid-price returns tested: do NOT remove the lag-1 reversal (genuine overshoot). User chose
"best compromise": scripts/e1_compromise.py chose tau_f 12.5, band 0.00875 (anchor point 139
otherwise, order_life 200) -> configs/calibrated.yaml (partial calibration failure, documented).
At 55% chartists the calibrated model oscillates strongly (vol ~20x, acf up to 0.67) - intended
boom-bust regime, flag in Phase 5 report.
E2/E2b/E4 + analyze + figures re-run launched (log: results/logs/phase5_rerun.log).

1. DONE 2026-11-19: rerun finished (no failures), F2 clean, 94 tests pass, ruff clean.
   Verdicts: H1 mixed (8/16), H2 mixed (55% yes, 40% no: tax raises vol), H3 supported.
2. Phase 5 report delivered; user said continue to Phase 6 (2026-11-19).
   Phase 6: beta pilot (results/e3_adaptive/pilot_beta.csv, median |piF-piC| 0.23); user
   approved beta 2 (mild) and 15 (strong). Added switch_pi record, share_mean_*/share_sd_chartist
   summary, 'beta' grid key (enables switching), type_shares.parquet, F7 fig_switching,
   tests/test_switching.py (102 tests pass). Notebook fresh-kernel run OK. E3 run launched
   (log results/logs/e3.log) -> then Phase 6 report + wait.
3. Phase 6 DONE 2026-11-19: E3 ran (300 runs, 3.5 min), F7 ok, analyze.py now includes share_*
   metrics. E3: switching raises vol (3x at beta 2, 13x at beta 15); PH cut n.s. at beta 2, raises
   vol/mispricing at beta 15. User pre-approved going straight to Phase 7.
4. Phase 7 (in progress 2026-11-19): README.md, docs/ODD.md (19 assumptions), docs/results.md and
   docs/figures/ (F1-F8 copies) written. Acceptance check PASSED: README followed in a clean copy
   reproduced F3; runs.parquet identical to the original. Phase 7 report delivered; waiting on
   user review and the git-commit question.
5. Nothing is committed to git yet (branch master; main branch is "main"). Ask before committing.
