# Model description (ODD protocol)

This document describes the Market ABM following the ODD protocol (Overview, Design concepts,
Details; Grimm et al. 2020). It is based on the model specification and the final
code in `src/marketabm/`. Parameter values are those of `configs/calibrated.yaml` unless stated.
The results describe the model's mechanism, not a forecast for the Philippine Stock Exchange.

---

## 1. Purpose and patterns

**Purpose.** To study how a per-trade, seller-side transaction tax changes volatility, liquidity
and price efficiency in a market of heterogeneous traders, and whether the answer depends on the
share of trend-following (chartist) traders. The policy motivation is the Philippine stock
transaction tax cut from 0.6% to 0.1% of the gross selling price (RA 12214, 1 July 2025). A
simulation can hold everything fixed and change only the tax, which real data cannot.

The model is built to be able to show both effects found in the literature:
- *composition effect*: the tax drives out noise traders and speculators, so prices calm down;
- *liquidity effect*: fewer traders thin the order book, so each order moves the price more.

**Hypotheses** (tested in `docs/results.md`):
- H1: the tax suppresses chartist and noise trading more than fundamentalist trading (E2).
- H2: in chartist-heavy markets, a higher tax lowers volatility and mispricing (E2).
- H3: in thin markets, a higher tax raises volatility through larger price impact (E2b).

**Patterns used to evaluate the model** (stylized facts of daily returns, Cont 2001), checked on
standardized period returns in calibration (E1):
1. fat tails: excess kurtosis > 1;
2. no linear predictability: |ACF of returns at lag 1| < 0.1;
3. volatility clustering: ACF of |returns| above the 95% band at every lag 1–20;
4. tail index (Hill) between 2 and 6;
5. no deterministic cycle: |ACF of returns| < 0.1 at lags 2–5 (added after a calibrated point
   was found to cycle).

The calibrated model meets patterns 1, 4 and 5 and partly meets 3 (16 of 20 lags); it misses 2
(lag-1 ACF ≈ −0.27). See Section 7 and `docs/results.md`.

---

## 2. Entities, state variables and scales

**Agents (traders).** N agents (default 1,000), each with:

| Variable | Meaning | Fundamentalist | Chartist | Noise |
|---|---|---|---|---|
| type | strategy | F | C | N |
| g1 | weight on fundamental signal | U(0.8, 1.2) | U(0.0, 0.2) | 0 |
| g2 | weight on trend signal | U(0.0, 0.2) | U(0.8, 1.2) | 0 |
| n | weight on noise | U(0.1, 0.3) | U(0.1, 0.3) | 1 |
| L | trend lookback (steps) | U{5..50} | U{5..50} | U{5..50} |
| h | forecast horizon (steps) | U{10..100} | U{10..100} | U{10..100} |
| k | order price offset | U(0, 0.01) | U(0, 0.01) | U(0, 0.01) |
| cash, pos | money and units held | start at 0; unbounded (borrowing and short selling allowed) |||
| live order | at most one resting order | | | |

Parameters are drawn once at initialization. Under switching (E3), type and (g1, g2, n) can change.

**Order book.** A continuous double auction with price-time priority, integer tick prices
(tick = 0.01 currency, i.e. 1 basis point at the initial price of 100). Each order has an id,
owner, side, limit price in ticks, remaining quantity, submission time, expiry time
(submission + `order_life` = 200 steps) and a sequence number for time priority.

**Market (global) state.** Last trade price p (starts at p0 = 100), recorded end-of-step price
P_t, fundamental value f_t, cumulative tax revenue, and the volatility-feedback state (two EWMA
variances of one-step returns).

**Environment.** The fundamental value follows an exogenous geometric random walk,
log f_{t+1} = log f_t + σ_f z_t with σ_f = 0.001 per step, generated in full before the run.
It never depends on trading or on the tax.

**Scales.** Time is discrete: T = 50,000 steps, the first T_burn = 5,000 are burn-in. A step has
no calendar meaning. Return statistics use **periods** of 20 steps, giving 2,250 period returns
per run (comparable to about 9 years of daily data). One asset, one exchange, no space.

---

## 3. Process overview and scheduling

Each step t = 0 … T−1:

1. **Expire** all resting orders with expiry ≤ t.
2. **Activate** M = 10 agents, drawn uniformly without replacement. They act **sequentially** in the
   drawn order; the book updates after each action, so later agents see earlier ones' effects.
   Each activated agent:
   1. forms a forecast (Section 7.2);
   2. applies the participation filter, which is where the tax enters; a skipped agent submits
      nothing and its old order stays;
   3. otherwise cancels its old order and submits a new limit order. If the order crosses the
      book it trades immediately at the resting orders' prices. The seller pays the tax on each
      fill, and any remainder rests.
3. **Record** P_t = p, best bid/ask, volume, fills, skips and activity by type, tax revenue.
   Update the volatility-feedback state.
4. Every 20 steps, **snapshot** depth within ±1% of the mid price and the book size.
5. **E3 only:** every S = 500 steps, run a strategy-switching round (Section 7.6).

Per-step records are taken after all M agents have acted.

---

## 4. Design concepts

- **Basic principles.** A heterogeneous-agent market in the tradition of Brock & Hommes (1998),
  with an explicit limit order book as in Chiarella, Iori & Perelló (2009). The tax works as in
  Westerhoff (2003) and Mannaro et al. (2008): it removes trades whose expected gain is too small
  to cover it.
- **Emergence.** Prices, volatility, spreads, depth, the trading mix and the tax's net effect all
  emerge from order flow. None is imposed.
- **Adaptation.** Fixed rules, except in E3, where non-noise agents choose between fundamentalist
  and chartist strategies by a logit rule on recent profits.
- **Objectives.** Agents trade when the expected log gain over their horizon exceeds the tax
  plus a fixed cost (c_min = 0).
- **Prediction.** Each agent mixes three signals: reversion to the fundamental, the recent trend,
  and noise. The noise is scaled up after volatile periods (volatility feedback).
- **Sensing.** Agents observe the true fundamental value, the last trade price and their own past
  price window. They do not see the book beyond what their own order meets.
- **Interaction.** Indirect, through the order book: trades happen between a new order and resting
  orders.
- **Stochasticity.** Fundamental shocks, agent parameters, activation order, forecast noise and
  switching draws, each from its own random stream (Section 5).
- **Collectives.** None besides the type groups.
- **Observation.** All metrics are computed after burn-in (Section 7.7). Experiments save one
  summary row per run.

---

## 5. Initialization

- p = p0 = 100, f_0 = 100, empty book, all cash and positions 0.
- **Types** come from the mix (default 45% F, 25% C, 30% N): n_C = round(N·s_C),
  n_N = round(N·s_N), and fundamentalists absorb rounding. The type array is shuffled.
- **Parameters** are drawn as U(0,1) arrays for all agents, then mapped to each agent's type range,
  in a fixed order (g1, g2, n, then L, h, k). The number of random draws is therefore the same for
  every mix.
- **Mix from a chartist share c** (E2): fundamentalists 0.6(1−c), noise 0.4(1−c), chartists c.
- **Random streams (common random numbers).** One root `SeedSequence(seed)` is spawned into five
  generators: fundamental, init, activation, noise, switching. The fundamental path, the activation
  sequence (T×M) and the noise draws (T×M) are all drawn before the loop. The agent in slot j at step
  t uses noise[t, j] whether or not it trades. So two runs with the same seed and different taxes
  share the same fundamental path, agents, activation order and noise, and any difference is
  caused by the tax. Experiments use seeds 1000–1029 in every cell, so cells are matched pairs.
  Results are identical serially or in parallel (tested).

---

## 6. Input data

The model uses no external input while it runs. Calibration (E1) compares the shape of model
returns with daily log returns of the **iShares MSCI Philippines ETF (EPHE)**, 2015-01-05 to
2026-11-17 (2,986 returns, from Yahoo Finance). EPHE stands in for the PSEi, whose Yahoo ticker
has no history. Caveat: EPHE is priced in USD and traded in the US, so its returns include the
PHP/USD exchange rate.

---

## 7. Submodels

### 7.1 Signals

When agent i acts at step t with current price p:
```
F = (1 / tau_f) * log(f_t / p)               fundamental reversion, tau_f = 12.5 steps
C = (log p − log P_{t−L}) / L                 trend (P before step 0 is p0)
E = eps * m_t,  eps ~ N(0, sigma_eps)         noise, sigma_eps = 0.0002
r_hat = (g1 F + g2 C + n E) / (g1 + g2 + n)   expected per-step log return
expected move = r_hat * h,  p_hat = p * exp(r_hat * h)
```

### 7.2 Volatility feedback (v1.2)

`m_t = clip((s_fast / s_slow)^gamma, 1/cap, cap)`, where s_fast and s_slow are bias-corrected
EWMA standard deviations of one-step log returns up to t−1, with half-lives 20 and 2,000 steps.
Calibrated values: gamma = 3, cap = 5. m_t is the same for all agents within a step, and m_0 = 1.
The feedback uses no random numbers. After large moves forecasts get noisier, so more agents pass
the tax filter, orders grow and prices move more. This follows Raberto et al. (2001). It is a
modelling device, not a derived behaviour.

### 7.3 Participation filter (where the tax enters)

```
expected_gain = |r_hat * h|
participate if expected_gain > tau + c_min
```
The filter applies to buyers and sellers alike: a round trip pays the seller-side tax once. A skip
counts as **caused by the tax** if expected_gain > c_min, and as "no signal" otherwise.

### 7.4 Order submission

1. Cancel the agent's resting order, if any (at most one live order per agent, so no self-trades).
2. Side: buy if p_hat > p, sell if p_hat < p.
3. Order price with a **price band** δ = 0.00875: `p_ord = p * exp(clip(r_hat*h, −δ, +δ))`; then
   buy at `p_ord (1 − k)`, sell at `p_ord (1 + k)`, rounded to ticks (minimum 1 tick). The band
   bounds only the order price; the forecast, side and filter use the unclipped values.
4. Quantity (signal-sized): `q = min(20, max(1, ceil(expected_gain / 0.01)))`.
5. Expiry: t + 200.
6. Match against the opposite side at the resting orders' prices, best price first and FIFO
   within a price. Any remainder rests.

**Why the band, signal sizing and tau_f (v1.1).** The original rule (orders at the full forecast,
unit size, tau_f = 100) diverged whenever chartists were present. A far-away chartist order sets
an extreme trade price, which strengthens the trend signal, and so on. At high chartist shares,
weak reversion and head-count voting gave runaway bubbles. A test documents the divergence.

### 7.5 Tax accounting

On each fill of q units at price x:
```
buyer:  cash −= x q,            pos += q
seller: cash += x q (1 − tau),  pos −= q
tax_revenue += tau x q
```
The seller pays, whether its order was incoming or resting. Money is conserved (cash summed over
agents plus tax revenue equals 0), and positions sum to 0. Both are tested.

### 7.6 Strategy switching (E3)

Every S = 500 steps:
1. For F and C, `pi_k` = mean over agents of type k of (W_i(t) − W_i(t−S)) / p0, where
   W_i = cash_i + pos_i · p. A type with no agents gets 0.
2. Each non-noise agent draws a new type with `P(k) = exp(beta pi_k) / Σ_j exp(beta pi_j)`, using a
   stable softmax.
3. An agent whose type changed redraws (g1, g2, n) from the new profile. L, h and k stay fixed.

Noise traders never switch. The switching stream is used only when switching is on, and it uses a
fixed number of draws per round. beta ∈ {2, 15} (mild and strong) was chosen from a pilot: the
median |pi_F − pi_C| per window is about 0.23, so at the median gap an agent picks the better type
with probability 0.61 at beta = 2 and 0.97 at beta = 15.

### 7.7 Metrics (after burn-in)

Period returns: `R_j = log P` at the end of period j+1 minus `log P` at the end of period j
(periods of 20 steps).
- **Volatility:** `vol` = sd(R). ACF of R at lag 1. Mean ACF of |R| at lags 1–10 and 11–50.
- **Tails:** excess kurtosis; Hill index on the top 5% of |R|; maximum drawdown of log P;
  extreme moves (|R| > 5 × the vol of the τ = 0 run with the same seed).
- **Liquidity:** total volume; mean relative spread; share of one-sided steps; mean depth within
  ±1% of the mid price; mean book size.
- **Efficiency:** `mispricing` = mean |log P − log f|.
- **Composition, by type:** fills per agent, volume share, aggressive share, skip rate due to the
  tax, skip rate for no signal.
- **Revenue:** total tax revenue and revenue per step.
- **Switching (E3):** mean type shares over switching rounds, and sd of the chartist share.

### 7.8 Parameter values

| Parameter | Baseline | Calibrated (E1) |
|---|---|---|
| T, T_burn, return_interval | 50,000, 5,000, 20 | same |
| N, M | 1,000, 10 | same |
| p0, tick, σ_f | 100, 0.01, 0.001 | same |
| tau_f | 10 | 12.5 |
| σ_eps | 0.0005 | 0.0002 |
| k_max | 0.005 | 0.01 |
| price_band | 0.01 | 0.00875 |
| order_life | 200 | 200 |
| chartist g2 | [0.8, 1.2] | [0.8, 1.2] |
| vol feedback | off | on, gamma 3, cap 5, half-lives 20 / 2,000 |
| order size | signal, unit 0.01, max 20 | same |
| mix F / C / N | 0.45 / 0.25 / 0.30 | same |
| switching | off (E3: S = 500, beta ∈ {2, 15}) | same |

---

## 8. Assumptions

Choices not fixed by the specification. Each is marked `# ASSUMPTION` in the code or the configs,
except the first, which concerns the repository layout.

**Project and configuration**
1. The project sits at the repository root rather than in a `market-abm/` subfolder.
2. Turning switching on with `beta: null` is rejected rather than given a default
   (`config.py`).
3. `beta` is a flat experiment grid key, like `chartist_share`. Setting it also turns switching
   on (`config.py`).

**Metrics**
4. With switching on, the composition metrics count agents per type by their **final** types
   (`results.py`).
5. E3 adds per-run summaries `share_mean_<type>` (mean over rounds after burn-in) and
   `share_sd_chartist` (`results.py`).
6. `volume_share` and `fills_per_agent` count participations (one per fill, whatever its quantity);
   `volume_total` counts units.

**Calibration (E1)**
7. E1 has its own driver script (`scripts/run_calibration.py`).
8. Calibration seeds (1–3 and 100–129) are kept apart from the experiment seeds (1000+), so the
   parameters are not tuned on the draws the experiments use (`configs/e1_calibration.yaml`).
9. A data download with fewer than 250 rows counts as failed (`scripts/fetch_psei.py`).
10. Rows with missing volume are kept; only an explicit zero volume marks a holiday
    (`empirical.py`).
11. "Averaged over 30 seeds" in the validation targets means each statistic, including each
    |R| ACF lag, is averaged over seeds first, then compared with the target. The ACF band is
    the mean per-run band (`calibration.py`).
12. Stage 2 re-runs the top 10 points by distance **and** every point whose 3-seed means pass all
    checks. With 3 seeds the distance is too noisy to rank on (`scripts/run_calibration.py`).
13. A fifth validation check (no cycle at lags 2–5) was added after the
    first calibrated point turned out to have an ~88-step price cycle.
14. Fallback rule when no point passes all checks (`scripts/e1_compromise.py`): the point must pass
    kurtosis, Hill and no-cycle; among those, the lowest distance D wins.

**Analysis**
15. H2 needs the volatility **and** mispricing CIs below 0 at the same τ. "Supported" means at
    both chartist shares 0.40 and 0.55; "mixed" means at one share or for one metric only
    (`analysis.py`).
16. H3 is "supported" when, at N = 250, vol and spread rise and depth falls (all CIs excluding 0)
    at at least one τ.
17. Sobol analysis: a NaN output is replaced by the mean of that output, and the count is reported
    (`analysis.py`).
18. Figure tax axes are categorical (evenly spaced) so that 0.1% and 0.25% stay readable.

**Notebook**
19. The sandbox sets T_burn = 10% of T, with T and T_burn rounded to multiples of
    `return_interval` (`viz.py`).

---

## 9. Known limitations

- One asset, one exchange; no cross-market trading.
- Order size grows with expected gain up to a cap; no wealth or budget constraints; unlimited
  short selling.
- Order prices are bounded by a price band around the last trade price. Without it the model
  diverges when chartists are present.
- Agents see the true fundamental value.
- Volatility clustering comes partly from an imposed feedback: forecast noise scales with recent
  volatility. It is a modelling device, not a derived behaviour.
- Fixed agent rules (except in E3); no learning beyond switching.
- One model step has no calendar meaning, so only the **shape** of returns is calibrated.
- The calibration is partial: the model's lag-1 return autocorrelation (≈ −0.27) is more negative
  than the target allows, and clustering clears the band at 16 of 20 lags. The empirical target is
  an ETF (EPHE), not the PSEi itself.
- At 55% chartists the calibrated model enters a boom-bust regime (volatility about 20× the
  baseline); effects there reflect that regime.
- In E2b, M = 10 for both N values, so thinness means fewer distinct traders and fewer resting
  orders, while each agent acts more often.
- The tax is seller-side only, as in the Philippine STT; other designs are not studied.
- Results show a mechanism in a stylized model, not a prediction for the PSE.

---

## References

- Brock, W. A., & Hommes, C. H. (1998). Heterogeneous beliefs and routes to chaos in a simple asset pricing model. *JEDC*, 22, 1235–1274.
- Chiarella, C., Iori, G., & Perelló, J. (2009). The impact of heterogeneous trading rules on the limit order book and order flows. *JEDC*, 33(3), 525–537. https://arxiv.org/abs/0711.3581
- Colliard, J.-E., & Hoffmann, P. (2017). Financial transaction taxes, market composition, and liquidity. *Journal of Finance*, 72(6), 2685–2716.
- Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.
- Franke, R., & Westerhoff, F. (2012). Structural stochastic volatility in asset pricing dynamics: Estimation and model contest. *JEDC*, 36(8), 1193–1211.
- Fricke, D., & Lux, T. (2015). The effects of a financial transaction tax in an artificial financial market. *JEIC*, 10(1), 119–150.
- Grimm, V., et al. (2020). The ODD protocol for describing agent-based and other simulation models: A second update. *JASSS*, 23(2), 7.
- Mannaro, K., Marchesi, M., & Setzu, A. (2008). Using an artificial financial market for assessing the impact of Tobin-like transaction taxes. *JEBO*, 67(2), 445–462.
- Pellizzari, P., & Westerhoff, F. (2009). Some effects of transaction taxes under different microstructures. *JEBO*, 72(3), 850–863.
- Raberto, M., Cincotti, S., Focardi, S. M., & Marchesi, M. (2001). Agent-based simulation of a financial market. *Physica A*, 299(1–2), 319–327.
- Westerhoff, F. (2003). Heterogeneous traders and the Tobin tax. *Journal of Evolutionary Economics*, 13(1), 53–70.
