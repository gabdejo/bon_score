# 1-Year Information Ratio Distribution — project context

## Goal
Approximate the CDF of the **1-year Information Ratio** of Peruvian AFP pension funds vs fixed-weight
benchmarks, from stationary-bootstrap simulations, and test whether it fits a parametric
distribution. Typical questions: P(1y IR < 0), the 5% quantile of the 1y IR, and which closed-form
distribution can stand in for the simulated CDF.

## Pipeline (all local, run inside `.venv`)
```
weights.csv ──► build_benchmarks.py ──► benchmarks.csv ─┐
                (Yahoo ETFs + BCRP FX)                   ├─► simulate.py ──► ir_best_fit.csv, ir_fit_summary.csv,
funds.csv ───────────────────────────────────────────────┘   ir_draws.csv, ir_cdf.png
```
```
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python build_benchmarks.py          # ~10 s, needs internet; skip if you supply benchmarks.csv
.venv\Scripts\python simulate.py                  # ~20 s; --gof_mc 49 adds AD p-values (~4 min)
.venv\Scripts\python simulate.py --start 2015-01-01 --paths 1000 --block 63 --T 252 --seed 42 --out ir
```
Never install into the global Python; the user manages `.venv` (Python 3.10). Packages: numpy, pandas,
scipy >= 1.10, matplotlib, pyarrow, yfinance.

## Data
- `funds.csv`: daily NAVs (valor cuota, PEN), columns `<AFP><type>` = {HAB,INT,PRI,PRO} × {0,1,2,3},
  dates dd/mm/yyyy, Aug 1993 → Sep 2026. Type 0 funds are dropped because there is no `BMK0`.
- `weights.csv`: `benchmark,ticker,weight[,currency]`; BMK1/2/3 map to fund types 1/2/3. Current
  constituents: EMLC, EMB, LQD, HYG, ACWI, EPU (all USD ETFs, total return via Yahoo adjusted close).
- `benchmarks.csv`: daily levels in PEN, base 100, constant-mix (daily rebalanced), aligned to the
  funds.csv calendar (levels forward-filled across holiday mismatches). First valid dates:
  BMK3 2009-06-23 (EPU inception), BMK1/BMK2 2010-07-26 (EMLC inception).
- **FX**: USD→PEN uses the BCRP API, series PD04640PD (SBS selling rate). Yahoo's `PEN=X` was tried
  and rejected: 410 days off >2% from the local median, a +257% tick in Nov 2010. Buy/sell choice
  shifts levels by a constant ~0.2% and leaves returns untouched.
- `prices_raw.csv`: downloaded inputs kept for reproducibility.

## Data contract (what another dataset must satisfy to run unchanged)
1. **Levels, not returns.** Both CSVs carry price levels (NAVs, index levels); returns are derived
   with `pct_change()` inside `simulate.py`. Levels must be strictly positive.
2. **One currency.** Fund NAVs and benchmark levels must be in the same currency. The builder only
   knows USD→PEN (BCRP); another currency pair means editing `bcrp_fx` / `FX` in `build_benchmarks.py`.
3. **Mapping by last character.** A fund column `XYZk` is compared to benchmark `BMKk`; `k` is the
   last character of the column name. Funds whose `k` has no `BMKk` column are dropped. Benchmark
   names in `weights.csv` must therefore be exactly `BMK<k>`.
4. **Dates.** A `date` column, parsed day-first (dd/mm/yyyy; ISO yyyy-mm-dd also works). Business
   days only, sorted, no duplicates. Rows with any NaN inside the window are dropped across all series,
   so gaps shrink the sample rather than being filled. Benchmark levels are forward-filled onto the
   funds calendar at load time, so a hand-made `benchmarks.csv` may use its own calendar.
5. **Frequency.** Daily by default (`--ppy 252`, `--T 252`, `--block 63`). Monthly data: `--ppy 12
   --T 12 --block 3` (block length is in periods). One path = one 1-year block, so `--T` should be
   a multiple of `--ppy`.
6. **Window.** `--start` is the first date of the resampling window; all series must exist from
   then on (the join is inner). Benchmarks start at the youngest constituent's inception.
7. **weights.csv.** `benchmark,ticker,weight[,currency]`; Yahoo symbols; weights sum to 1 per
   benchmark; currency is `USD` (converted) or `PEN` (left as is). Adjusted close → total return.
8. **Ignored by git**: `funds.csv`, `weights.csv`, `benchmarks.csv`, `prices_raw.csv`, outputs. A
   new dataset is a new set of these files, not a code change.

## Simulation (`simulate.py`)
- Resampling window: common dates from `--start` (default 2015-01-01) → 3,043 daily returns, 15 series
  (12 funds + 3 benchmarks). Uniform block starts (no recency weighting).
- Stationary bootstrap (Politis–Romano): geometric block lengths, mean `--block` 63 days, circular
  wrap. **All 15 series share the same indices per path** — independent resampling of fund vs
  benchmark creates spurious tracking error.
- Default 1,000 paths × 252 days → 1,000 one-year IR draws per fund. The cube is only saved with
  `--save_cube` (`{out}_sims.npy` + `{out}_sims_cols.json`); the seed makes it reproducible.
- Per fund type k, `run_cube(R[..., type k], bench_col=BMK{k}, out=None)` returns the tables;
  simulate.py concatenates across types and writes once.

## Core modules
| File | Role |
|---|---|
| `ir_cdf_fit.py` | `one_year_ir`, `fit_candidates`, `analyze_ir`, plus a CLI for a 2-D active-return matrix |
| `ir_cube.py` | S × T × N cube → active returns (`bench_col` / `bench` / `peer` / `active_weights`) → `analyze_ir` per fund. `run_cube` returns `(best, draws, fit_summary)`; `out=None` writes nothing |
| `build_benchmarks.py` | weights + Yahoo + BCRP → `benchmarks.csv`; `--check` runs the currency self-test |
| `simulate.py` | bootstrap cube + per-type IR analysis; `--check` tests block lengths and joint indexing |

## Methodology (design decisions — keep unless there's a reason)
1. **IR per path** = mean(active) / std(active, ddof=1) × sqrt(252) over each non-overlapping 1-year block.
2. **Candidates**: `nct_theory` (iid-normal benchmark: noncentral t, df = T−1, only nc fitted; exact
   here only because block length = ppy), `nct_free`, `norm`, `t`, `skewnorm`, `johnsonsu`.
3. **Ranking** by BIC.
4. **Goodness of fit**: effect sizes first — KS distance vs the DKW 95% band and 5%/10% quantile
   errors. p-values from a parametric-bootstrap Anderson–Darling test on a 1,000-draw subsample;
   with `gof_mc=49` the floor is 0.02, so a 0.02 means "≤ 0.02", not a precise value.

## Outputs (`--out ir`, deliberately few)
| File | Content |
|---|---|
| `ir_best_fit.csv` | one row per fund: benchmark, TE, IR moments, best model (BIC) + params (JSON, scipy order: shapes…, loc, scale), KS vs DKW, P(IR<thr) fit vs empirical, q05 |
| `ir_fit_summary.csv` | every candidate model × fund (loglik, AIC/BIC, KS, quantile errors, AD p-value if `--gof_mc > 0`) |
| `ir_draws.csv` | raw 1y IR draws, one column per fund; use for joint questions and correlations |
| `ir_cdf.png` | 3 × 4 grid: empirical CDF, DKW 95% band, best-fit CDF |

The standalone `ir_cube.py` / `ir_cdf_fit.py` CLIs still write their older per-fund file sets.

## Results of the first real run (2026-10-13, seed 42, window 2015-01-05 → 2026-09-25)
- Annualized TE ≈ 7–8% (types 1–2) and ≈ 12.5% (type 3); benchmarks are ~2× as volatile as the funds.
- Mean 1y IR: type 1 ≈ 0 to +0.26, type 2 ≈ −0.2 to 0, type 3 ≈ −0.35 to −0.62. P(IR<0) ranges 0.36–0.80.
- IR sd ≈ 0.70–0.83, below the ≈1 of iid theory → negative autocorrelation in active returns
  (consistent with smoothed NAV pricing of local assets vs daily-marked ETFs). `nct_theory` rejected everywhere.
- Best by BIC: `norm` for 8 funds, `skewnorm` for 3, `t` for 1; **all inside the DKW band**
  (KS 0.018–0.035 vs half-width 0.043). Bootstrap AD p-values are mostly at the 0.02 floor for
  `norm`/`skewnorm`; `johnsonsu` and `t` are higher. Interpretation: the fits are adequate for
  P(IR<thr) and the 5% quantile (errors ≤ 0.05), but draws from a 3,043-day history heavily reuse
  the same data, so AD sees structure that no smooth family matches exactly.

## Caveats / known limitations
- Per-fund fits are **marginals**; use `_ir_draws.csv` for joint questions.
- `one_year_ir` silently drops blocks with zero std; if that ever happens the draw rows across funds
  no longer align and `_ir_corr.csv` is wrong. Not an issue with daily real data.
- `run_cube` fits each fund twice when plot=True (harmless).
- `nct_free` is slow and its GOF usually fails (NaN) — expected.
- The 2-D CLI in `ir_cdf_fit.py` also writes `{out}_samples.csv` and `{out}_fit.json`.
- Fixed 2026-10-13: `_params` used to append loc/scale twice (norm stored as 4 numbers). Params in
  older output files are not trustworthy.
- CLI flag spelling is mixed: `--bench-col`, `--peer-weights` (hyphen) vs `--gof_mc`, `--gof_n` (underscore).

## Possible next steps
- Recency-weighted block starts (exponential, ~10y half-life) if uniform resampling is not wanted.
- Pre-2010 benchmark history (needs a local-currency EM bond proxy for EMLC).
- Joint metrics (probability of ranking first, P(IR_i > IR_j)) from the raw draws.
- The user prefers CSV/JSON tables plus a sources/methodology note over written reports.
