# CLAUDE.md — Market ABM: Transaction Taxes and Trader Heterogeneity

This file is the complete specification for this repository. Read all of it before writing code.

**Rules of engagement**
- Follow this file over your own defaults.
- If something is ambiguous, contradictory or looks wrong, **stop and ask**. Do not guess.
- If you must make a small choice this file does not cover, pick the simplest option, mark it in
  code with `# ASSUMPTION: ...`, and list it in your phase summary.
- Never silently change an equation, default value, metric definition or experiment design.
- Work in phases (Section 15). Stop after each phase for review.

---

## Table of contents

1. Project context
2. Research question, hypotheses and how each is tested
3. Environment and dependencies
4. Repository layout
5. Model specification
6. Order book specification
7. Simulation loop (pseudocode)
8. Reproducibility and common random numbers
9. Configuration system
10. Outputs and storage
11. Metrics (exact definitions)
12. Validation and calibration (E1)
13. Experiments (E2, E2b, E3, E4) and statistical analysis
14. Figures, notebook sandbox and documentation
15. Build phases and acceptance criteria
16. Testing requirements
17. Performance guidance
18. Coding conventions
19. Known limitations (for the report)
20. References

---

## 1. Project context

- **What:** A computational modeling class final project (two students). We build an agent-based
  model (ABM) of a stock market for one synthetic asset and use it to study a transaction tax.
- **Why the tax:** The Philippines cut its stock transaction tax (STT) from 0.6% to 0.1% of the
  gross selling price on July 1, 2025 (RA 12214, CMEPA). The tax is paid by the **seller**.
  Real PSE data cannot isolate the tax's effect because many other things changed at the same time.
  A simulation can hold everything fixed and change only the tax.
- **The debate we test:** Prior work splits into two views.
  - *Composition effect:* a tax pushes out noise traders and speculators, so prices calm down.
  - *Liquidity effect:* fewer traders means a thinner order book, so each order moves the price
    more and volatility rises.
  Agent-based studies disagree (Westerhoff 2003; Mannaro et al. 2008; Pellizzari & Westerhoff 2009;
  Fricke & Lux 2015). Our model must be able to show either effect.
- **What results mean:** The findings describe the model's mechanism, not a forecast for the PSE.
  No docstring, README text, figure caption or notebook cell may claim otherwise.
- **Main deliverables:** (1) a tested, reproducible simulator; (2) experiment results and figures;
  (3) a Jupyter notebook sandbox with sliders; (4) an ODD-format model description.

---

## 2. Research question, hypotheses and how each is tested

**Research question:** How does a per-trade transaction tax change volatility, liquidity and price
efficiency in a market of heterogeneous traders, and does the answer depend on the share of
trend-following (chartist) traders?

| ID | Hypothesis | Tested in | Operational test |
|---|---|---|---|
| H1 | The tax suppresses chartist and noise trading more than fundamentalist trading. | E2 | For each type, compute the relative change in fills per agent between τ = 0 and each τ > 0 (paired by seed). H1 holds if the drop for chartists and noise traders is larger than for fundamentalists, with 95% CIs not overlapping. Also report `skip_rate_tax` by type. |
| H2 | In chartist-heavy markets, a higher tax lowers volatility and mispricing. | E2 | At chartist share 0.40 and 0.55: the paired difference `vol(τ) − vol(0)` and `mispricing(τ) − mispricing(0)` has a 95% CI entirely below 0 for at least one τ > 0. |
| H3 | In thin markets, a higher tax raises volatility through larger price impact. | E2b | At N = 250: the paired difference `vol(τ) − vol(0)` has a 95% CI entirely above 0 for at least one τ > 0, alongside a rise in relative spread and a fall in depth. |

**Headline policy quantity ("PH cut effect"):** for each setting, the paired difference
`metric(τ = 0.001) − metric(τ = 0.006)` across the same seeds. This answers "what does the model
predict the 0.6% → 0.1% cut did, and under which trader mix does the sign flip?"

---

## 3. Environment and dependencies

- Python **3.12**, managed with **uv**. Use a src layout package named `marketabm`.

```bash
uv python install 3.12
uv init --package market-abm --python 3.12
cd market-abm
uv add numpy pandas pyarrow scipy statsmodels SALib matplotlib pyyaml pydantic yfinance tqdm joblib jupyterlab ipywidgets
uv add --dev pytest pytest-cov ruff hypothesis
```

- Run everything through uv:
  - Tests: `uv run pytest -q`
  - Lint and format: `uv run ruff check . && uv run ruff format .`
  - Experiments: `uv run python scripts/run_experiment.py --config configs/e2_tax_mix.yaml`
  - Notebook: `uv run jupyter lab notebooks/sandbox.ipynb`
- Ruff config in `pyproject.toml`: line length 100, rules `E,F,I,B,UP,N,SIM`.
- **Do not add** Mesa, Streamlit, Plotly, Numba or any other dependency without asking.
  Numba may be proposed later only if profiling shows the order book is the bottleneck.

---

## 4. Repository layout

```
market-abm/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .gitignore                 # data/, results/, .ipynb_checkpoints/, __pycache__/
├── configs/
│   ├── baseline.yaml          # full default parameter set (Section 9.2)
│   ├── e1_calibration.yaml
│   ├── e2_tax_mix.yaml
│   ├── e2b_thin_market.yaml
│   ├── e3_adaptive.yaml
│   └── e4_sensitivity.yaml
├── src/marketabm/
│   ├── __init__.py
│   ├── config.py              # pydantic models, YAML loading, mix derivation
│   ├── rng.py                 # seed handling: build the named random streams
│   ├── orderbook.py           # limit order book + matching engine (pure, no RNG)
│   ├── agents.py              # agent population as NumPy arrays; type profiles; decision rule
│   ├── market.py              # run(config, seed) -> SimulationResult; the main loop
│   ├── switching.py           # Brock–Hommes logit switching (E3 only)
│   ├── results.py             # SimulationResult dataclass, summary dict, serialization
│   ├── metrics.py             # all metrics (pure functions on arrays)
│   ├── stylized_facts.py      # kurtosis, Hill index, ACFs, validation checks
│   ├── empirical.py           # load PSEi data, compute empirical moments, bootstrap
│   ├── calibration.py         # E1 grid search, moment distance
│   ├── experiments.py         # grid expansion, parallel execution, summary tables
│   ├── analysis.py            # paired differences, CIs, hypothesis checks
│   └── viz.py                 # all plotting functions (matplotlib only)
├── scripts/
│   ├── fetch_psei.py
│   ├── run_experiment.py      # --config path [--n-jobs N] [--dry-run]
│   ├── analyze.py             # --experiment name -> tables in results/<name>/analysis/
│   └── make_figures.py        # --experiment name -> PNG + PDF in results/<name>/figures/
├── notebooks/
│   └── sandbox.ipynb          # thin: imports from marketabm only
├── docs/
│   ├── ODD.md                 # model description in ODD format (Phase 7)
│   └── results.md             # short results write-up (Phase 7)
├── tests/
│   ├── test_config.py
│   ├── test_orderbook.py
│   ├── test_agents.py
│   ├── test_tax.py
│   ├── test_market.py
│   ├── test_reproducibility.py
│   ├── test_metrics.py
│   └── test_analysis.py
├── data/                      # gitignored
│   ├── raw/
│   └── processed/
└── results/                   # gitignored
```

---

## 5. Model specification

### 5.1 Notation and units

| Symbol | Meaning | Unit / type |
|---|---|---|
| t | step index, 0 … T−1 | int |
| p_t | current market price (last trade price) | currency (float), stored internally as integer ticks |
| P_t | price at the **end** of step t (recorded) | currency |
| f_t | fundamental value at step t | currency |
| τ (`tau`) | transaction tax rate, a fraction (0.001 = 0.1%) | float |
| N | number of agents | int |
| M | agents activated per step | int |
| tick | minimum price increment | currency |

All logs are natural logs. All "returns" are log returns.

### 5.2 Time and activation

- Discrete steps `t = 0 … T−1`. Defaults: `T = 50_000`, `T_burn = 5_000`.
- In each step, exactly `M` agents (default 10) are drawn **uniformly without replacement** from
  all N agents, using the `activation` random stream (Section 8). They act **sequentially** in the
  drawn order. The order book updates after each action, so later agents in the step see the
  effect of earlier ones.
- One step has no calendar meaning. For return statistics we aggregate steps into **periods** of
  `return_interval` steps (default 20). With the defaults this gives (50,000 − 5,000) / 20 = 2,250
  period returns, comparable in length to about 9 years of daily PSEi data.

### 5.3 Fundamental value

```
log f_{t+1} = log f_t + sigma_f * z_t,     z_t ~ N(0, 1)
```

- `f_0 = p0 = 100.0`, `sigma_f = 0.001` per step.
- The whole path `f_0 … f_{T−1}` is generated **before** the loop from the `fundamental` stream.
  It never depends on agent behavior or on τ.
- Agents observe `f_t` exactly (no observation noise in v1).

### 5.4 Agent types and parameters

Each agent i has a fixed type in {fundamentalist, chartist, noise} (fixed unless switching is on,
Section 5.8) and these parameters, all drawn **once** at initialization from the `init` stream:

| Parameter | Meaning | Fundamentalist | Chartist | Noise |
|---|---|---|---|---|
| g1 | weight on fundamental signal | U(0.8, 1.2) | U(0.0, 0.2) | 0 |
| g2 | weight on chartist signal | U(0.0, 0.2) | U(0.8, 1.2) | 0 |
| n | weight on noise | U(0.1, 0.3) | U(0.1, 0.3) | 1.0 |
| L | chartist lookback (steps) | U{5, …, 50} | U{5, …, 50} | U{5, …, 50} |
| h | forecast horizon (steps) | U{10, …, 100} | U{10, …, 100} | U{10, …, 100} |
| k | order price offset | U(0, k_max) | U(0, k_max) | U(0, k_max) |

- `k_max = 0.005` by default. All ranges live in config (`profiles`, `L_range`, `h_range`, `k_max`).
- Integer uniforms `U{a, …, b}` include both ends.
- L is drawn for every agent (even noise traders) so that the number of random draws does not
  depend on type. This keeps the `init` stream aligned across mixes.

**Type assignment from the mix.** Given shares `(s_F, s_C, s_N)` summing to 1:
1. `n_C = round(N * s_C)`, `n_N = round(N * s_N)`, `n_F = N − n_C − n_N` (fundamentalists absorb rounding).
2. Build the type array `[F]*n_F + [C]*n_C + [N]*n_N`.
3. Shuffle it with the `init` stream.
4. Then draw the per-agent parameters in this fixed order, each as one vectorized call over all N
   agents: `u_g1, u_g2, u_n` (three `U(0,1)` arrays, mapped to each agent's type range), then `L`,
   `h`, `k`.

Mapping a `U(0,1)` draw `u` to a type range `[a, b]` is `a + u * (b − a)`. Using U(0,1) arrays
for every agent, mapped by type, keeps the number of draws identical regardless of the mix.

### 5.5 Decision rule

When agent i is activated at step t with current price p (the last trade price at that moment):

**Signals**
```
F = (1 / tau_f) * log(f_t / p)                 # fundamental: expected per-step reversion
C = (log p − log P_{t−L}) / L                  # chartist: mean per-step log return over last L steps
E = eps                                        # noise: eps ~ N(0, sigma_eps), from the `noise` stream
```
- `tau_f = 10` steps (config `tau_f`): the time scale over which fundamentalists expect the gap to close.
  (Changed from 100 in v1.1; see Section 5.9.)
- `P_{t−L}` is the recorded end-of-step price L steps ago. If `t − L < 0`, use `P_0 = p0` (so C is
  computed against the initial price; this only affects burn-in).
- `sigma_eps = 0.0005` per step.

**Expected per-step return and expected price**
```
r_hat = (g1 * F + g2 * C + n * E) / (g1 + g2 + n)
p_hat = p * exp(r_hat * h)
```

**Participation filter (where the tax enters)**
```
expected_gain = abs(r_hat * h)                 # = abs(log(p_hat / p))
participate   = expected_gain > tau + c_min
```
- `c_min = 0.0` by default (a fixed per-trade cost; kept for sensitivity checks).
- The filter applies to buyers and sellers alike: a round trip pays the seller-side tax once.
- If `participate` is False: record `skipped_tax[type] += 1` **only if** `expected_gain > c_min`
  (so the skip is caused by the tax), otherwise record `skipped_no_signal[type] += 1`. The agent
  submits nothing. Its existing resting order (if any) **stays** in the book.

**Order submission (if participating)**
1. Cancel the agent's existing resting order, if any (each agent has at most one live order).
2. Side: buy if `p_hat > p`, sell if `p_hat < p`.
3. Limit price, with the **order price band** `delta = price_band` (default 0.01):
   ```
   move   = r_hat * h                                  # = log(p_hat / p)
   move_b = clip(move, −delta, +delta)                 # only if price_band is not null
   p_ord  = p * exp(move_b)
   ```
   - buy: `price = p_ord * (1 − k)`
   - sell: `price = p_ord * (1 + k)`
   The band bounds only the order price. The forecast `p_hat`, the side and the participation
   filter (which uses the unclipped `expected_gain`) are unchanged. With `price_band: null` the
   rule reduces to the v1 rule `price = p_hat * (1 ∓ k)`.
4. Convert to integer ticks: `price_ticks = max(1, round(price / tick))`. No other rejection or
   special-casing: an order that sits away from the market simply rests until it fills or expires.
5. Quantity (config `order_size`, default `signal`):
   - `unit`: `q = 1`.
   - `signal`: `q = min(size_max, max(1, ceil(expected_gain / size_unit)))`, with
     `size_unit = 0.01` and `size_max = 20` by default.
   Expiry step = `t + order_life` (default `order_life = 200`).
6. Submit to the order book. If it crosses the opposite best quote, it executes immediately against
   resting orders at their prices (Section 6). Any unfilled remainder rests in the book.

Note: a buy limit at `p_ord * (1 − k)` can sit above the best ask when `p_ord` is high enough.
That order is "marketable" and trades at once. This is how aggressive orders arise naturally.

### 5.6 Transaction tax accounting

On every fill of `q` units at price `x` (currency):
```
buyer.cash   -= x * q
buyer.pos    += q
seller.cash  += x * q * (1 − tau)
seller.pos   -= q
tax_revenue  += tau * x * q
```
- The tax is charged to the **seller**, whether the seller was the incoming or the resting order.
- v1 has no budget or inventory limits: cash and positions can go negative (unlimited borrowing
  and short selling). They are tracked for profit accounting only.
- Mark-to-market wealth of agent i at time t: `W_i = cash_i + pos_i * p_t`. Initial cash = 0,
  initial position = 0, so wealth equals cumulative trading profit net of tax.

### 5.7 Price recording

- `p` (the current price) starts at `p0` and updates to the fill price after every fill.
- At the end of each step, record `P_t = p`. If no trade has happened yet, `P_t = p0`.
- One-step log return: `r_t = log P_t − log P_{t−1}` for t ≥ 1.
- Period returns (for stylized facts): `R_j = log P_{T_burn + (j+1)·Δ − 1} − log P_{T_burn + j·Δ − 1}`
  with Δ = `return_interval`, i.e. the log change between consecutive period-end prices after burn-in.

### 5.8 Strategy switching (E3 only; `switching.enabled = true`)

- Every `S` steps (default 500), at the end of the step:
  1. For each type k in {fundamentalist, chartist}, compute
     `pi_k = mean over agents of type k of (W_i(t) − W_i(t − S)) / p0`.
     If a type has no agents, set its `pi_k = 0`.
  2. For each non-noise agent, draw a new type with
     `P(type = k) = exp(beta * pi_k) / sum_j exp(beta * pi_j)` over the two types.
     Use a numerically stable softmax (subtract the max).
  3. If an agent's type changed, redraw its `(g1, g2, n)` from the new type's profile using the
     `switching` stream. L, h and k stay fixed.
- Noise traders never switch.
- Record the type shares after each switching round.
- All switching randomness comes from the dedicated `switching` stream (Section 8). With switching
  off, this stream is never used.
- `beta` levels are set in Phase 6 after inspecting the typical size of `pi_k` (aim for one level
  where switching is mild and one where it is strong). Do not invent values before then.

### 5.9 Stability changes (v1.1) and why they were made

The v1 rule (`price_band: null`, `order_size: unit`, `tau_f: 100`) diverged on every tested seed
whenever chartists were present (price overflow within the first few hundred steps at the baseline
mix; prices up to 1e10 even at 10% chartists). Experiments found two separate causes:

1. **Explosion: limit prices become trade prices.** Orders were priced at the full forecast
   `p_hat`, often far from the market. A resting order trades at its own limit, so during a trend
   one far-away chartist order sets an extreme price, which strengthens the trend signal, and so on.
   Lowering the chartist gain (`h = L`), pre-filling the book or recording the mid price did not
   stop it. **Fix:** the order price band (Section 5.5 step 3), analogous to exchange price bands.
2. **Runaway bubbles at high chartist shares: reversion too weak.** With the band, 40% and 55%
   chartists still produced persistent bubbles (mispricing ≈ 2.4 at 55%). Fundamentalists expect
   only `h / tau_f` (≈ 0.55 on average with `tau_f = 100`) of the gap to close, while chartists
   extrapolate the trend by `h / L` (≈ 2.8 on average); and with unit orders the market is a head
   count, so a large mispricing cannot outweigh a chartist majority. **Fix:** `tau_f = 10` and
   signal-sized orders.

With all three changes, 50,000-step runs stay bounded at every E2 chartist share (mispricing ≈ 0.005
at 10% chartists, ≈ 0.01 at 40%, ≈ 0.21 at 55%). Larger mispricing in chartist-heavy markets is an
intended model outcome (it is what H2 tests), not a failure. The v1 rule remains available through
config for comparison; `tests/test_market.py::test_original_rule_diverges` documents the divergence.

### 5.10 Volatility feedback on the noise term (v1.2, approved by the user)

**Why.** The first E1 run (v1 grid, Section 12.4) found no point passing 12.3. The blocker was
volatility clustering: only 2 of 1,728 points had the |R| ACF above the band at all lags 1–20, and
those had strongly trending returns (`acf_r_lag1` ≈ +0.3 to +0.45). The user chose to add an
explicit clustering mechanism. It follows Raberto et al. (2001), whose order prices scale with
recent volatility; that model family underlies Mannaro et al. (2008).

**Rule.** When `vol_feedback.enabled` is true, the noise term of Section 5.5 becomes
```
E   = eps * m_t
m_t = clip( (s_fast / s_slow) ** gamma, 1 / cap, cap )
```
- `s_fast`, `s_slow`: bias-corrected EWMA standard deviations of one-step log returns
  `r_u = log P_u − log P_{u−1}` (with `P_{−1} = p0`) over `u ≤ t − 1`, with half-lives
  `fast_halflife` and `slow_halflife` steps (EWMA weight `λ = 0.5 ** (1 / halflife)`; bias
  correction divides by `1 − λ^n`). `m_t` is fixed for all M agents within step t; `m_0 = 1`.
- Defaults: `enabled: false`, `gamma: 1.0`, `fast_halflife: 20`, `slow_halflife: 2000`, `cap: 5.0`.
  Validation: `gamma ≥ 0`, `cap ≥ 1`, `fast_halflife < slow_halflife`.
- It uses no random numbers, so CRN (Section 8) is unaffected. With `enabled: false` or
  `gamma = 0` the model is exactly v1.1 (tested).
- `SimulationResult.vol_mult` records `m_t` per step.

**Mechanism.** After large moves, forecasts get noisier, so more agents pass the tax filter, orders
grow (signal sizing) and prices move more, which raises `s_fast` again. In calm periods `m_t ≈ 1`.
The feedback mainly fattens the tails; the clustering itself needs the narrower price band found in
the v1.2 grid. The tax interacts with it: a higher τ filters out more of the extra noise trades.

---

## 6. Order book specification

### 6.1 Requirements

- Continuous double auction, **price-time priority**.
- Integer tick prices only (no floats inside the book).
- Each order: `order_id` (int, increasing), `agent_id`, `side` (+1 buy, −1 sell), `price_ticks`,
  `qty_remaining`, `t_submitted`, `expiry_t`, `seq` (submission counter for time priority).
- An incoming buy with `price_ticks >= best_ask` matches against asks from the lowest price up,
  FIFO within a price, until filled or no longer crossing. Symmetric for sells.
- Trades execute at the **resting** order's price.
- Any remainder rests in the book.
- Expired orders are removed at the start of each step, before any agent acts:
  remove all orders with `expiry_t <= t`.
- Cancellation by order_id.
- No self-trades can occur because each agent cancels its own order before submitting a new one.
  Still assert in debug mode that `buyer_id != seller_id` on every fill.

### 6.2 Data structures (recommended)

- `bids`: heap of `(-price_ticks, seq, order_id)`; `asks`: heap of `(price_ticks, seq, order_id)`.
- `orders`: dict `order_id -> Order` holding only live orders.
- Lazy deletion: cancel or expire removes from `orders`; when popping a heap top, discard entries
  whose `order_id` is not in `orders`.
- `expiry_queue`: `collections.deque` of `(expiry_t, order_id)`. Because `order_life` is constant,
  expiry times are non-decreasing in submission order, so expiry is a pop from the left while
  `expiry_t <= t`.
- `agent_order`: list of length N with the agent's live order_id or −1.

### 6.3 API

```python
@dataclass(frozen=True, slots=True)
class Fill:
    t: int
    price_ticks: int
    qty: int
    buyer_id: int
    seller_id: int
    aggressor_side: int   # +1 if incoming order was a buy, −1 if a sell

class OrderBook:
    def __init__(self, tick: float) -> None: ...
    def submit(self, agent_id: int, side: int, price_ticks: int, qty: int,
               t: int, expiry_t: int) -> tuple[int | None, list[Fill]]:
        """Match, then rest any remainder. Returns (resting order_id or None, fills)."""
    def cancel(self, order_id: int) -> bool: ...
    def expire(self, t: int) -> int:            # returns number of orders removed
    def best_bid(self) -> int | None: ...
    def best_ask(self) -> int | None: ...
    def depth_within(self, mid_ticks: float, pct: float) -> tuple[int, int]:
        """Total qty on bid side with price >= mid*(1-pct), and ask side with price <= mid*(1+pct)."""
    def __len__(self) -> int: ...                # number of live orders
```

The order book holds no RNG and no tax logic. Tax is applied by the market loop using the fills.

---

## 7. Simulation loop (pseudocode)

```text
run(config, seed) -> SimulationResult
  streams = make_streams(seed)                       # Section 8
  f = fundamental_path(streams.fundamental, T)       # array of length T
  agents = init_agents(config, streams.init)         # NumPy arrays + Python lists for the hot loop
  activation = streams.activation.permuted(...)      # or draw per step; see Section 8
  noise = streams.noise.normal(0, sigma_eps, size=(T, M))
  book = OrderBook(tick)
  p = p0; logP = empty array(T)
  per-step recorders: P, f, best_bid, best_ask, volume, tax_rev, fills_by_type(3),
                      skipped_tax_by_type(3), skipped_no_signal_by_type(3), active_by_type(3)
  snapshot recorders (every return_interval steps): depth_bid, depth_ask, book_size

  for t in 0..T−1:
      book.expire(t); clear agent_order entries for expired orders
      chosen = activation_indices(t)                 # M agent ids, from the activation stream
      for j, i in enumerate(chosen):
          eps = noise[t, j]                          # drawn regardless of what the agent does
          compute F, C, E, r_hat, p_hat              # Section 5.5
          if not participate: record skip; continue
          cancel agent i's live order (if any)
          side, price_ticks, q = ...                 # q from Section 5.5 step 5
          resting_id, fills = book.submit(i, side, price_ticks, q, t, t + order_life)
          for fill in fills:
              apply cash/position/tax updates        # Section 5.6
              p = fill.price_ticks * tick
              record volume, fills_by_type for BOTH counterparties' types
              clear agent_order for the counterparty whose resting order was fully filled
          agent_order[i] = resting_id (or −1)
      P[t] = p; logP[t] = log(p)
      record best_bid, best_ask (None -> NaN), cumulative tax revenue
      if (t + 1) % return_interval == 0: record depth_within(mid, 0.01) and len(book)
      if switching.enabled and (t + 1) % S == 0: run switching round   # Section 5.8
  return SimulationResult(...)
```

Notes:
- Per-step records are taken **after** all M agents have acted.
- Mid price for depth: `(best_bid + best_ask) / 2` if both exist; otherwise skip the snapshot (NaN).
- `fills_by_type` counts **participations**: each fill adds 1 to the buyer's type and 1 to the
  seller's type. Also record passive vs aggressive participations by type.

---

## 8. Reproducibility and common random numbers (CRN)

Every tax comparison uses **common random numbers**: running the same seed at different τ values
must give the same fundamental path, the same agent parameters, the same activation sequence and
the same noise draws. Any difference must come from the tax.

**Streams.** Build all generators from one root:
```python
ss = np.random.SeedSequence(seed)
fundamental, init, activation, noise, switching = (np.random.default_rng(s) for s in ss.spawn(5))
```
The order of `spawn` is fixed as written. Never use `np.random.seed` or the global RNG.

**Rules that keep CRN intact**
1. The fundamental path is generated in full before the loop.
2. Agent initialization draws the same number of values regardless of mix (Section 5.4).
3. Activation: for each step draw M distinct agent ids from the `activation` stream with
   `rng.choice(N, size=M, replace=False)`, or pre-draw a `(T, M)` array before the loop.
   The draw never depends on the market state.
4. Noise: pre-draw a `(T, M)` array; agent in slot j at step t uses `noise[t, j]` whether or not
   it trades.
5. No branch (skip, fill, expiry, switching) may change how many numbers any stream consumes,
   except the `switching` stream, which is used only when switching is enabled.

**Seeds in experiments.** Each experiment uses one list of seeds (e.g. 1000–1029), shared by every
cell. So cell (τ = 0.006, mix = X) and cell (τ = 0.001, mix = X) with seed 1012 are a matched
pair.

**Determinism.** Results must be identical whether run serially or with `joblib` using any
`n_jobs`. Each worker builds its own streams from the seed it receives.

**Run metadata.** Every saved run records: config (full, resolved), seed, config hash (SHA-256 of
the canonical JSON), git commit hash if available, Python and package versions.

---

## 9. Configuration system

### 9.1 Pydantic models

`config.py` defines a `ModelConfig` (all model parameters) and an `ExperimentConfig` (base config +
grid + seeds + options). Validate:
- shares in `mix` are each in [0, 1] and sum to 1 within 1e-9;
- all ranges have `low <= high`;
- `T > T_burn`, `(T − T_burn) % return_interval == 0`;
- `M <= N`; `tau >= 0`; `tick > 0`;
- `price_band` is null or > 0; `order_size` ∈ {`unit`, `signal`}; `size_unit > 0`; `size_max >= 1`.

### 9.2 `configs/baseline.yaml` (complete default parameter set)

```yaml
# Market
T: 50000
T_burn: 5000
M: 10
N: 1000
p0: 100.0
tick: 0.01
sigma_f: 0.001
order_life: 200
return_interval: 20

# Tax and costs
tau: 0.0
c_min: 0.0

# Agent decision rule
tau_f: 10
sigma_eps: 0.0005
k_max: 0.005
L_range: [5, 50]
h_range: [10, 100]

# Order price band and order size (Section 5.5)
price_band: 0.01        # null = no band (v1 rule; diverges with chartists)
order_size: signal      # "unit" = always 1 unit; "signal" = grows with expected gain
size_unit: 0.01         # expected log gain per unit of order size
size_max: 20

# Population
mix: {fundamentalist: 0.45, chartist: 0.25, noise: 0.30}
profiles:
  fundamentalist: {g1: [0.8, 1.2], g2: [0.0, 0.2], n: [0.1, 0.3]}
  chartist:       {g1: [0.0, 0.2], g2: [0.8, 1.2], n: [0.1, 0.3]}
  noise:          {g1: [0.0, 0.0], g2: [0.0, 0.0], n: [1.0, 1.0]}

# Volatility feedback on the noise term (Section 5.10, v1.2; off = v1.1 model)
vol_feedback: {enabled: false, gamma: 1.0, fast_halflife: 20, slow_halflife: 2000, cap: 5.0}

# Strategy switching (E3 only)
switching: {enabled: false, S: 500, beta: null}

# Output
save_series: true
depth_pct: 0.01
```

These are **starting values**. E1 calibration (Phase 4) may change `sigma_eps`, `order_life`,
`k_max`, the chartist g2 range, `tau_f`, `price_band` and `vol_feedback`; the calibrated set is saved as
`configs/calibrated.yaml` and used by E2–E4.

### 9.3 Deriving the mix from a chartist share

Experiments vary `chartist_share` (a derived parameter, not a raw config key). Given baseline
shares F = 0.45, N = 0.30 and a target chartist share c:
```
fundamentalist = (1 − c) * 0.45 / (0.45 + 0.30) = 0.6 * (1 − c)
noise          = (1 − c) * 0.30 / (0.45 + 0.30) = 0.4 * (1 − c)
chartist       = c
```
Implement as a function `mix_from_chartist_share(c, base_mix) -> dict`.

### 9.4 Experiment config format

```yaml
name: e2_tax_mix
base: configs/calibrated.yaml     # falls back to baseline.yaml before E1 is done
seeds: {start: 1000, count: 30}
grid:
  tau: [0.0, 0.001, 0.0025, 0.006, 0.01]
  chartist_share: [0.10, 0.25, 0.40, 0.55]
overrides: {}                     # fixed overrides applied to every cell
save_series: false
```
The grid is the Cartesian product of all listed keys. Any grid key that is a raw config key
overrides it directly; `chartist_share` goes through Section 9.3.

---

## 10. Outputs and storage

### 10.1 `SimulationResult`

A dataclass holding:
- `config`, `seed`, `metadata`
- Per-step arrays (length T): `price`, `fundamental`, `best_bid`, `best_ask`, `volume`,
  `tax_revenue_cum`, `fills_by_type` (T×3), `aggressive_fills_by_type` (T×3),
  `skipped_tax_by_type` (T×3), `skipped_no_signal_by_type` (T×3), `active_by_type` (T×3)
- Snapshot arrays (length T / return_interval): `depth_bid`, `depth_ask`, `book_size`
- Switching: `type_shares` (rounds × 3), empty if switching is off
- Final agent state: `types`, `cash`, `pos`
- `summary()` → flat dict of all metrics in Section 11 (computed after burn-in)

### 10.2 Files

- Single runs and the notebook: keep everything in memory; optionally save to
  `results/single/<run_id>.parquet` when `save_series: true`.
- Experiments: per-step series are **not** saved by default (1,000+ runs × 50,000 rows would take
  gigabytes). Save one row per run: `results/<experiment>/runs.parquet` with columns for every
  grid key, seed and every summary metric.
- If `save_series: true` in an experiment config, save series downsampled to period-end values only.
- `results/<experiment>/config_resolved.yaml` records the expanded experiment.

---

## 11. Metrics (exact definitions)

All metrics use data **after burn-in** only. Let `R` be the period returns (Section 5.7), `n = len(R)`.

### 11.1 Volatility and clustering
- `vol`: sample standard deviation of R (ddof = 1).
- `acf_r_lag1`: autocorrelation of R at lag 1.
- `acf_absr_1_10`: mean autocorrelation of |R| over lags 1–10.
- `acf_absr_11_50`: mean autocorrelation of |R| over lags 11–50.
- Use `statsmodels.tsa.stattools.acf(x, nlags=50, fft=True)`. The 95% band is `±1.96 / sqrt(n)`.

### 11.2 Tail risk
- `excess_kurtosis`: `scipy.stats.kurtosis(R, fisher=True, bias=False)`.
- `hill_alpha`: Hill tail index on |R| using the top 5% of observations. With `x_(1) ≥ x_(2) ≥ …`
  and `k = floor(0.05 n)`: `alpha = 1 / ( (1/k) * sum_{i=1..k} log(x_(i) / x_(k+1)) )`.
  Lower alpha means fatter tails (real stock returns typically fall around 3–5).
- `max_drawdown`: largest peak-to-trough fall in `log P` after burn-in, reported as a positive number.
- `extreme_moves`: count of period returns with `|R| > 5 * vol_ref`, where `vol_ref` is the `vol`
  of the **τ = 0 run with the same seed and the same other parameters**. Computed in `analysis.py`
  (it needs the paired baseline), not in `summary()`.

### 11.3 Liquidity
- `volume_total`: total units traded after burn-in.
- `rel_spread_mean`: mean over steps of `(ask − bid) / ((ask + bid) / 2)`, only on steps where both
  sides exist.
- `one_sided_share`: share of steps where at least one side of the book is empty.
- `depth_mean`: mean of `depth_bid + depth_ask` over snapshots where both sides exist.
- `book_size_mean`: mean number of live orders over snapshots.

### 11.4 Efficiency
- `mispricing`: mean over steps of `abs(log P_t − log f_t)`.

### 11.5 Composition (H1)
For each type k:
- `fills_per_agent_k`: total participations of type k / number of agents of type k.
- `volume_share_k`: participations of type k / total participations (sums to 1). A participation is
  one fill, whatever its quantity; `volume_total` counts units.
- `aggressive_share_k`: aggressive participations of type k / participations of type k.
- `skip_rate_tax_k`: skipped_tax of type k / activations of type k.
- `skip_rate_no_signal_k`: skipped_no_signal of type k / activations of type k.

### 11.6 Revenue
- `tax_revenue`: total tax collected after burn-in.
- `tax_revenue_per_step`: the same divided by (T − T_burn).

### 11.7 Robustness to NaN
If a metric cannot be computed (for example no trades), return NaN and log a warning. Never raise
inside `summary()`.

---

## 12. Validation and calibration (E1)

### 12.1 Real data
- `scripts/fetch_psei.py` downloads PSEi daily closes for 2015-01-01 to today with
  `yfinance.download("PSEI.PS", ...)`. **This ticker has not been verified.**
- If the download fails or returns no rows: do not try other sites or tickers. Print exactly what
  the user must do: place a CSV with columns `Date,Close` at `data/raw/psei.csv`. Then load it.
- Save cleaned data to `data/processed/psei_returns.parquet`: daily log returns, dropping zero-volume
  holidays and missing values.
- **Update (Phase 4, approved by the user):** `PSEI.PS` returns only the current day's quote on
  Yahoo, so the empirical target is **EPHE** (iShares MSCI Philippines ETF, daily, 2015 onward),
  downloaded by `scripts/fetch_psei.py` (default `--ticker EPHE`). A user CSV at
  `data/raw/psei.csv` still takes priority. `data/processed/source.json` records the source, and
  the E1 report names it. Caveat for the report: EPHE is USD-priced and US-traded, so its returns
  include the PHP/USD rate. Text that says "PSEi" for E1 means this empirical target.

### 12.2 Moments
Compare the **shape** of returns, not their level. A model step has no calendar meaning, so the
volatility level is arbitrary. Before computing moments, standardize both model and PSEi returns to
zero mean and unit variance. The moment vector is:

```
m = [excess_kurtosis, hill_alpha, acf_r_lag1, acf_absr_1_10, acf_absr_11_50]
```

### 12.3 Pass/fail targets for the baseline (τ = 0, 30 seeds, averaged)
- excess kurtosis > 1
- |acf_r_lag1| < 0.1
- ACF of |R| above the 95% band at every lag from 1 to 20
- hill_alpha between 2 and 6
- **no deterministic cycle:** |ACF of R| < 0.1 at every lag from 2 to 5 (added after the v1.2
  run, approved by the user; see 12.4 step 6)

Also print the PSEi values next to the model values.

### 12.4 Calibration procedure
1. Compute empirical moments `m_emp` on PSEi returns.
2. Estimate the variance of each empirical moment with a **moving-block bootstrap** (block length
   20, 1,000 resamples). Weights are `w = 1 / var` (diagonal weighting matrix).
3. Grid (1,728 points, **v1.2**; `configs/e1_calibration.yaml`):
   - `sigma_eps` ∈ {0.0002, 0.0005, 0.001}
   - `order_life` ∈ {50, 100, 200}
   - `k_max` ∈ {0.005, 0.01}
   - chartist g2 range ∈ {[0.8, 1.2], [1.2, 1.8]}
   - `tau_f` ∈ {25, 27.5, 30, 32.5}
   - `price_band` ∈ {0.005, 0.00625, 0.0075}
   - `vol_feedback.gamma` ∈ {0, 1, 2, 3} (0 = feedback off; Section 5.10)
   At ≈ 6–10 s per run this is ≈ 75 min on 11 cores.
   The **v1 grid** (sigma_eps {0.0002, 0.0005, 0.001, 0.002}, order_life {50, 100, 200, 400}, k_max
   {0.001, 0.0025, 0.005, 0.01}, g2 {[0.4, 0.6], [0.8, 1.2], [1.2, 1.8]}, tau_f {10, 25, 50},
   price_band {0.005, 0.01, 0.02}) failed step 8; its results are archived in
   `results/e1_calibration/v1_grid/`. Pilots (`results/e1_calibration/pilot_*.csv`) located the
   v1.2 region: clustering without trending returns lies between tau_f 25 and 32.5 at price_band
   0.005–0.0075, and the feedback lowers the Hill index into range.
4. For each grid point, run 3 seeds and average the simulated moments `m_sim`.
5. Distance: `D = sum_i w_i * (m_sim_i − m_emp_i)^2`.
6. Re-run the 10 best points with 30 seeds each; pick the lowest mean D that also passes 12.3.
   **Update (v1.2 run):** step 6 also re-runs every point whose 3-seed means pass all 12.3
   checks. With 3 seeds D is too noisy to rank on: in the v1.2 run none of the top 10 by D passed
   at 30 seeds, while 4 of the 8 stage-1 passers did, and the best (point 139, D = 15.9) had a
   lower 30-seed D than every top-10 point (best 20.3). "Mean D" is D of the 30-seed mean moments.
   **Withdrawn:** point 139 turned out to have an ≈ 88-step deterministic price cycle (ACF of R
   −0.69 at lag 2, +0.46 at lag 4), present even with the feedback off. The user approved the
   fifth 12.3 check (no cycle) and a re-run; v1.2 outputs are archived in
   `results/e1_calibration/v1_2_cycle/`. A pilot (`pilot_cycle_map.csv`) found the cycle grows
   with tau_f at band ≤ 0.00625; wider bands remove it but bring back a lag-1 reversal.
   **Outcome (step 8, user-approved compromise):** no point passes all five checks; clustering and
   `|acf_r_lag1| < 0.1` conflict, and mid-price returns do not remove the reversal. The user chose
   a best-compromise point (`scripts/e1_compromise.py`): anchor sigma_eps 0.0002, k_max 0.01,
   g2 [0.8, 1.2], gamma 3, order_life 200; tau_f {12.5, 15, 17.5} × band {0.0075, 0.00875};
   30 seeds; must pass kurtosis, Hill and no-cycle, then lowest D. Chosen: tau_f 12.5, band
   0.00875 (D = 45.4; kurtosis 4.2, Hill 3.3, acf1 −0.27, |R| ACF above band at 16/20 lags).
   Saved as `configs/calibrated.yaml`; report in `results/e1_calibration/compromise_report.md`.
   This is reported as a partial calibration failure.
7. Save it as `configs/calibrated.yaml` and write `results/e1_calibration/report.md` with the top 10
   table, the chosen point, and the model-vs-PSEi moment table.
8. If **no** point passes 12.3: stop and report. Propose changes (for example a wider sigma_eps
   range, a shorter tau_f or a different price_band). Do not move to Phase 5. Known risk: before
   calibration, `acf_r_lag1` is about −0.4 at the defaults (prices alternate between bid and ask
   fills), which fails the `|acf_r_lag1| < 0.1` target.

---

## 13. Experiments and statistical analysis

### 13.1 Experiment table

| ID | Purpose | Grid | Seeds | Runs |
|---|---|---|---|---|
| E1 | Calibration | Section 12.4 | 3 per point, top 10 × 30 | 5,184 + 300 |
| E2 | Tax × trader mix (H1, H2, PH cut) | tau ∈ {0, 0.001, 0.0025, 0.006, 0.01} × chartist_share ∈ {0.10, 0.25, 0.40, 0.55} | 30 | 600 |
| E2b | Tax × market thinness (H3) | tau (same 5) × N ∈ {250, 1000}, baseline mix | 30 | 300 |
| E3 | Tax with switching | tau (same 5) × beta ∈ {beta_low, beta_high} | 30 | 300 |
| E4 | Global sensitivity of the PH cut effect | Sobol over N ∈ [250, 2000], order_life ∈ [50, 500], sigma_f ∈ [0.0005, 0.002] | 1 per sample point | 2 × 512 |

### 13.2 E2b details
In E2b, keep `M = 10` for both N values, so each agent acts more often in the small market. Note
this in the report: thinness here means fewer distinct traders and fewer resting orders.

### 13.3 E4 details
- Use `SALib.sample.saltelli.sample` (or `SALib.sample.sobol.sample` if the former is deprecated in
  the installed version) with base N = 64 and `calc_second_order=True`, giving 64 × (2·3 + 2) = 512
  points. N must be rounded to an integer and order_life too.
- For each point and one fixed seed (vary the seed across points: `seed = 5000 + point_index`), run
  **two** simulations: τ = 0.006 and τ = 0.001. The output is the PH cut effect
  `Y = metric(0.001) − metric(0.006)` for `vol`, `mispricing` and `rel_spread_mean`.
- Analyze with `SALib.analyze.sobol.analyze`; report S1 and ST with confidence intervals.

### 13.4 Statistical analysis (`analysis.py`)
- **Cell means:** mean and 95% t-interval across seeds for every metric in every cell.
- **Paired differences:** for each cell with τ > 0, compute per-seed `metric(τ) − metric(τ = 0)`
  (same seed, same other parameters), then the mean and 95% paired t-interval.
- **PH cut effect:** per-seed `metric(0.001) − metric(0.006)`, mean and 95% paired t-interval.
- **Hypothesis checks:** implement H1–H3 exactly as in Section 2 and write
  `results/<experiment>/analysis/hypotheses.md` stating, for each, "supported", "not supported" or
  "mixed", with the numbers behind it.
- Do not run significance tests across dozens of cells without saying so: report the number of
  comparisons next to any claim.

---

## 14. Figures, notebook sandbox and documentation

### 14.1 Figures (`viz.py`, saved as PNG at 200 dpi and PDF)
| ID | Figure | Source |
|---|---|---|
| F1 | Price vs fundamental value over time, one baseline seed, burn-in shaded | single run |
| F2 | Stylized facts panel: return histogram vs normal (log y-axis), ACF of R, ACF of abs(R); model vs PSEi | E1 |
| F3 | Heatmaps over tau × chartist share: vol, mispricing, rel_spread_mean, volume_total, tax_revenue | E2 |
| F4 | PH cut effect (0.1% − 0.6%) with 95% CIs, one panel per metric, x = chartist share | E2 |
| F5 | Skip rate due to tax by agent type vs tau (H1) | E2 |
| F6 | vol and rel_spread_mean vs tau for N = 250 and N = 1000 with CIs (H3) | E2b |
| F7 | Chartist share over time under switching, per tau | E3 |
| F8 | Sobol S1 and ST bars with CIs | E4 |

Rules: label axes with units, show tau as a percent (0.1%, 0.6%), mark the old and new PH rates,
use one colormap per metric family, and use a diverging colormap centered on 0 for differences.

### 14.2 Notebook sandbox (`notebooks/sandbox.ipynb`)
- Thin wrapper only. All logic lives in `marketabm` (put any widget helper in `viz.py`).
- Widgets (`ipywidgets`): tau (0–1%, step 0.05%), fundamentalist / chartist / noise shares (three
  sliders, normalized to sum to 1 on run), N, T (default 10,000 for speed), seed, switching toggle,
  and a **Run** button (`interact_manual`), since a run takes seconds.
- Output after each run: price vs fundamental; period returns; spread and volume; volume share by
  type; a stylized-facts card (kurtosis, Hill alpha, ACF plots) with pass/fail marks from 12.3.
- A second section: "Compare two tax rates", which runs the same seed at two chosen τ values and
  plots them on shared axes, demonstrating CRN.
- The first markdown cell states that results illustrate the model's mechanism, not the PSE.

### 14.3 Documentation (Phase 7)
- `README.md`: install, test, run each experiment, reproduce figures, open the notebook, expected
  runtimes.
- `docs/ODD.md`: model description following the ODD protocol (Grimm et al. 2020): Purpose and
  patterns; Entities, state variables and scales; Process overview and scheduling; Design concepts;
  Initialization; Input data; Submodels. Generate it from this specification and the final code;
  list every `# ASSUMPTION` made.
- `docs/results.md`: one page per experiment with the key figure, the hypothesis verdict and the
  numbers.

---

## 15. Build phases and acceptance criteria

At the end of every phase: run `uv run pytest -q` and ruff, then report (a) what was built,
(b) assumptions made, (c) anything that looks off. **Wait for confirmation before the next phase.**

| Phase | Build | Done when |
|---|---|---|
| 1. Scaffold | uv project, layout, `config.py` with validation, `rng.py`, `baseline.yaml`, empty modules, `.gitignore` | `test_config.py` passes; invalid configs are rejected with clear errors |
| 2. Order book | `orderbook.py` + full tests (Section 16) | All order book tests pass, including hypothesis-based invariants |
| 3. Agents, tax, loop | `agents.py`, `market.py`, `results.py`; tax accounting; CRN | Tax, market and reproducibility tests pass; one baseline run completes; runtime reported |
| 4. Metrics and calibration | `metrics.py`, `stylized_facts.py`, `empirical.py`, `calibration.py`, PSEi fetch, E1 | Validation targets in 12.3 pass for `calibrated.yaml`, or a clear failure report with proposals |
| 5. Experiments | `experiments.py`, `analysis.py`, `run_experiment.py`, `analyze.py`, `make_figures.py`; run E2, E2b, E4 | Summary tables, hypothesis report and figures F1–F6, F8 exist |
| 6. Sandbox and switching | notebook; `switching.py`; choose beta levels; run E3 | Notebook runs top to bottom on a fresh kernel; F7 exists |
| 7. Documentation | README, `docs/ODD.md`, `docs/results.md` | A new user can reproduce F3 from the README alone |

---

## 16. Testing requirements

### `test_orderbook.py`
- Buy limit below best ask rests; best bid updates.
- Buy limit at or above best ask fills at the **ask** price (resting price), not the limit price.
- Price priority: better-priced resting orders fill first.
- Time priority: at equal price, earlier `seq` fills first.
- An incoming order crossing several levels produces several fills at each resting price, in order.
- Partial fill leaves the correct remainder resting.
- `cancel` removes an order; a cancelled order never fills; cancelling twice returns False.
- `expire(t)` removes exactly the orders with `expiry_t <= t`.
- Empty-book cases: `best_bid`/`best_ask` return None; submitting into an empty side rests.
- `depth_within` counts quantities correctly at the boundary prices.
- Hypothesis property tests over random order sequences:
  - after every `submit`, the book is not crossed (`best_bid < best_ask` when both exist);
  - total quantity is conserved: submitted = filled + resting + cancelled + expired;
  - every fill price lies within `[min_limit, max_limit]` of the two orders involved.

### `test_tax.py`
- With τ = 0: `tax_revenue == 0` and no `skipped_tax` counts.
- Tax revenue equals `sum(tau * price * qty)` over all fills to within 1e-9.
- Only sellers are charged: for a hand-built two-agent scenario, check both cash balances exactly.
- An agent whose expected gain is below τ is skipped and recorded under `skipped_tax`; with
  τ = 0 the same agent trades.
- Total cash across agents + tax revenue == 0 at every point (money is conserved).
- Total position across agents == 0 (every buy has a matching sell).

### `test_agents.py`
- Type counts match the mix exactly, with rounding absorbed by fundamentalists.
- Parameters fall within their configured ranges, by type.
- Noise traders have g1 = g2 = 0 and n = 1.
- The price band clips only the order price; the participation filter uses the unclipped gain.
- Signal-sized quantity equals `min(size_max, max(1, ceil(gain / size_unit)))`.
- The number of `init` stream draws does not depend on the mix (compare stream state after init).

### `test_market.py`
- A short run (T = 2,000, N = 100) completes and all arrays have the right lengths.
- Runs at every E2 chartist share stay bounded (finite prices, mispricing < 1).
- The v1 rule (`price_band: null`, `order_size: unit`, `tau_f: 100`) diverges (documents 5.9).
- Prices are always positive and on the tick grid.
- Each agent has at most one live order at any time.
- With all agents as fundamentalists and sigma_eps tiny, mispricing after burn-in is small
  (a sanity check that the fundamental signal pulls prices toward value).

### `test_reproducibility.py`
- Same config + seed → identical `SimulationResult` arrays (exact equality).
- Different τ, same seed → identical fundamental path, agent parameters, activation sequence and
  noise array.
- `run_experiment` gives identical `runs.parquet` with `n_jobs=1` and `n_jobs=4`.

### `test_metrics.py`
- Hill estimator on a Pareto sample with known alpha recovers it within 15% (n = 100,000).
- Kurtosis of a large normal sample is near 0.
- ACF of an i.i.d. sample is within the 95% band at most lags.
- `max_drawdown` on a hand-made series gives the known answer.

### `test_analysis.py`
- Paired differences on synthetic data with a known effect recover it and its CI covers it.
- `mix_from_chartist_share` returns shares summing to 1 and the expected values.

Target coverage: ≥ 85% for `orderbook.py`, `market.py`, `metrics.py`.

---

## 17. Performance guidance

- Target: one run at the defaults (T = 50,000, N = 1,000, M = 10) in **under 30 seconds** on a
  laptop. Report the actual time at the end of Phase 3.
- Inside the hot loop, use plain Python floats and lists for per-agent parameters (convert NumPy
  arrays with `.tolist()` once). Scalar NumPy operations are slow in tight loops.
- Precompute the noise and activation arrays before the loop.
- The chartist signal is O(1): keep `logP` as an array and compute `(log p − logP[t−L]) / L`.
- Compute depth only at snapshots (every `return_interval` steps), never every step.
- Parallelize across seeds with `joblib.Parallel(n_jobs=-1, backend="loky")`, with a `tqdm`
  progress bar.
- Profile with `cProfile` before any optimization, and report the top functions.

---

## 18. Coding conventions

- Type hints on every function; NumPy-style docstrings on public functions.
- No global state; no global RNG; generators are passed explicitly.
- Metric functions are pure (arrays in, numbers out). Only `market.py` holds mutable simulation state.
- No model logic in notebooks, scripts or plotting code.
- Use `logging`, not `print`, inside the package (scripts may print).
- Constants for type indices: `FUND = 0`, `CHART = 1`, `NOISE = 2`, defined once in `agents.py`.
- Never commit `data/` or `results/`.
- Keep functions under about 60 lines; split the main loop into helpers if it grows.

---

## 19. Known limitations (for the report)

Document these in `docs/ODD.md` and `docs/results.md`:
- One asset, one exchange; no cross-market trading.
- Order size grows with expected gain up to a cap; no wealth or budget constraints; unlimited
  short selling.
- Order prices are bounded by a price band around the last trade price (Section 5.9); without it
  the model diverges when chartists are present.
- Agents see the true fundamental value.
- Volatility clustering comes partly from an imposed feedback (Section 5.10): the noise in
  forecasts scales with recent volatility. It is a modelling device, not a derived behaviour.
- Fixed agent rules (except in E3); no learning beyond switching.
- One model step has no calendar meaning, so only the **shape** of returns is calibrated.
- Tax is seller-side only, as in the PH STT; other designs are not studied.
- Results show a mechanism in a stylized model, not a prediction for the PSE.

---

## 20. References

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
