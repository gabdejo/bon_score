"""
1-year Information Ratio distribution from stationary-bootstrap active-return paths.

Input : matrix of simulated ACTIVE returns, one row per path, one column per period
        (n_paths x n_periods). Paths longer than 1 year are cut into NON-overlapping
        1-year blocks (overlapping windows would be correlated draws).
        For an S x T x N cube of fund returns, use ir_cube.py (builds active returns
        and calls analyze_ir() per fund).

Why these candidates
--------------------
* nct_theory : if active returns were iid normal, the annualized 1y IR equals the t-stat
               of the mean, so  IR_hat ~ noncentral t(df=T-1, nc=sqrt(T)*SR_per_period).
               Only 1 free parameter -> the natural parametric benchmark.
* nct_free   : noncentral t with free df/nc/loc/scale (absorbs fat tails, autocorrelation
               and vol clustering that the bootstrap preserves but the theory ignores).
* norm       : asymptotic Lo (2002) / Mertens approximation. Fine for daily data, poor for monthly.
* t, skewnorm, johnsonsu : flexible 3-4 parameter alternatives.

Testing
-------
With thousands of simulations any formal test rejects tiny misfits, and KS/AD p-values are
invalid when parameters are estimated from the same data (Lilliefors problem). So:
  1) p-values come from a parametric bootstrap (scipy.stats.goodness_of_fit) on a subsample;
  2) decisions should lean on effect sizes: max |CDF gap| vs the DKW band, and tail
     quantile errors (5%/10%), which is what matters for P(IR < threshold).
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


# ---------------------------------------------------------------- IR computation
def one_year_ir(paths: np.ndarray, ppy: int) -> np.ndarray:
    """paths: (n_paths, n_periods) active returns -> annualized IR per 1y block."""
    n_paths, n_per = paths.shape
    n_blocks = n_per // ppy
    if n_blocks == 0:
        raise ValueError(f"Paths have {n_per} periods < 1 year ({ppy}).")
    blocks = paths[:, : n_blocks * ppy].reshape(n_paths * n_blocks, ppy)
    mu = blocks.mean(axis=1)
    sd = blocks.std(axis=1, ddof=1)
    ok = sd > 0
    return (mu[ok] / sd[ok]) * np.sqrt(ppy)


# ---------------------------------------------------------------- fitting
def fit_candidates(ir, ppy, sr_period, models=None):
    T = ppy
    want = set(models or ["norm", "t", "skewnorm", "johnsonsu", "nct_theory", "nct_free"])
    fits = {}
    if "norm" in want:
        fits["norm"] = stats.norm(*stats.norm.fit(ir))
    if "t" in want:
        fits["t"] = stats.t(*stats.t.fit(ir))
    if "skewnorm" in want:
        fits["skewnorm"] = stats.skewnorm(*stats.skewnorm.fit(ir))
    if "johnsonsu" in want:
        fits["johnsonsu"] = stats.johnsonsu(*stats.johnsonsu.fit(ir))
    nc_hat = None
    if want & {"nct_theory", "nct_free"}:
        nc0 = np.sqrt(T) * sr_period
        res = minimize_scalar(lambda nc: -stats.nct.logpdf(ir, T - 1, nc).sum(),
                              bracket=(nc0 - 1, nc0 + 1))
        nc_hat = res.x
    if "nct_theory" in want:
        fits["nct_theory"] = stats.nct(T - 1, nc_hat)
    if "nct_free" in want:
        try:
            fits["nct_free"] = stats.nct(*stats.nct.fit(ir, min(T - 1, 200), nc_hat))
        except Exception:
            pass
    return fits


N_PARAMS = {"norm": 2, "t": 3, "skewnorm": 3, "johnsonsu": 4, "nct_theory": 1, "nct_free": 4}
FAMILY = {"norm": stats.norm, "t": stats.t, "skewnorm": stats.skewnorm,
          "johnsonsu": stats.johnsonsu, "nct_free": stats.nct}


def gof_bootstrap(sample, name, dist, n_mc, rng):
    """Parametric-bootstrap AD p-value with params re-estimated each replicate."""
    if name == "nct_theory":
        r = stats.goodness_of_fit(stats.nct, sample, statistic="ad", n_mc_samples=n_mc,
                                  known_params=dict(df=dist.args[0], loc=0, scale=1),
                                  random_state=rng)
    else:
        r = stats.goodness_of_fit(FAMILY[name], sample, statistic="ad",
                                  n_mc_samples=n_mc, random_state=rng)
    return r.pvalue


def _params(d):
    """Frozen dist -> [shape args..., loc, scale] in scipy order (fits pass all of them positionally)."""
    return [float(x) for x in d.args] + [float(d.kwds[k]) for k in ("loc", "scale") if k in d.kwds]


# ---------------------------------------------------------------- core analysis
def analyze_ir(ir, ppy, sr_period, thresholds=(-0.5, 0.0, 0.5), gof_n=1000, gof_mc=199,
               seed=42, models=None, out_prefix=None, plot=True, label=""):
    """Fit candidate distributions to a vector of 1y IR draws.

    gof_mc=0 skips the (slow) bootstrap AD test.
    Returns (summary_df, quantiles_df, meta_dict). Writes files if out_prefix is given.
    """
    rng = np.random.default_rng(seed)
    ir = np.asarray(ir, float)
    ir = ir[np.isfinite(ir)]
    n = ir.size
    fits = fit_candidates(ir, ppy, sr_period, models)

    dkw = np.sqrt(np.log(2 / 0.05) / (2 * n))           # DKW 95% half-width
    xs = np.sort(ir)
    ecdf = np.arange(1, n + 1) / n
    sub = rng.choice(ir, size=min(gof_n, n), replace=False)
    q_emp = np.quantile(ir, QS)
    qtab = pd.DataFrame({"q": QS, "empirical": q_emp})

    rows = []
    for name, d in fits.items():
        cdf = d.cdf(xs)
        ks = np.max(np.maximum(np.abs(ecdf - cdf), np.abs(ecdf - 1 / n - cdf)))
        ll = d.logpdf(ir).sum()
        k = N_PARAMS[name]
        qtab[name] = d.ppf(QS)
        row = dict(label=label, model=name, k=k, loglik=ll, aic=2 * k - 2 * ll,
                   bic=k * np.log(n) - 2 * ll, ks_D=ks, dkw95=dkw, within_DKW95=ks <= dkw,
                   q05_err=d.ppf(0.05) - q_emp[1], q10_err=d.ppf(0.10) - q_emp[2])
        for t in thresholds:
            row[f"P(IR<{t})_fit"] = d.cdf(t)
            row[f"P(IR<{t})_emp"] = (ir < t).mean()
        if gof_mc > 0:
            try:
                row["AD_pvalue_boot"] = gof_bootstrap(sub, name, d, gof_mc, rng)
            except Exception:
                row["AD_pvalue_boot"] = np.nan
        rows.append(row)
    summ = pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
    qtab.insert(0, "label", label)

    meta = dict(label=label, n_ir_draws=int(n), periods_per_year=ppy, dkw95_halfwidth=dkw,
                sr_per_period_pooled=float(sr_period),
                ir_mean=float(ir.mean()), ir_sd=float(ir.std(ddof=1)),
                ir_skew=float(stats.skew(ir)), ir_exkurt=float(stats.kurtosis(ir)),
                best_by_bic=summ.iloc[0]["model"],
                params={k: _params(v) for k, v in fits.items()})

    if out_prefix:
        pd.Series(ir, name="ir_1y").to_csv(f"{out_prefix}_samples.csv", index=False)
        summ.to_csv(f"{out_prefix}_fit_summary.csv", index=False)
        qtab.to_csv(f"{out_prefix}_quantiles.csv", index=False)
        json.dump(meta, open(f"{out_prefix}_fit.json", "w"), indent=2)
        if plot:
            _plot(xs, ecdf, dkw, fits, summ.iloc[0]["model"], f"{out_prefix}_fit.png", label)
    return summ, qtab, meta


def _plot(xs, ecdf, dkw, fits, best, path, label):
    n = xs.size
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].step(xs, ecdf, where="post", color="k", lw=1.2, label="Empirical")
    ax[0].fill_between(xs, np.clip(ecdf - dkw, 0, 1), np.clip(ecdf + dkw, 0, 1),
                       step="post", color="grey", alpha=0.25, label="DKW 95%")
    grid = np.linspace(xs[0], xs[-1], 400)
    for name, d in fits.items():
        ax[0].plot(grid, d.cdf(grid), lw=1, label=name)
    ax[0].set_title(f"1y IR CDF {label}"); ax[0].legend(fontsize=7)
    pp = (np.arange(1, n + 1) - 0.5) / n
    ax[1].scatter(fits[best].ppf(pp), xs, s=3)
    lim = [xs[0], xs[-1]]; ax[1].plot(lim, lim, "r--", lw=1)
    ax[1].set_title(f"QQ vs {best}"); ax[1].set_xlabel("theoretical"); ax[1].set_ylabel("sample")
    lo = xs <= np.quantile(xs, 0.15)
    ax[2].semilogy(xs[lo], ecdf[lo], "k", label="Empirical")
    for name, d in fits.items():
        ax[2].semilogy(xs[lo], d.cdf(xs[lo]), lw=1, label=name)
    ax[2].set_title("Lower tail (log CDF)"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------- CLI (2-D active returns)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="CSV/parquet/npy, rows=paths, cols=periods (active returns)")
    ap.add_argument("--ppy", type=int, default=12, help="periods per year: 12, 52, 252")
    ap.add_argument("--transpose", action="store_true", help="if rows=periods, cols=paths")
    ap.add_argument("--gof_n", type=int, default=1000)
    ap.add_argument("--gof_mc", type=int, default=199, help="0 = skip bootstrap AD test")
    ap.add_argument("--thresholds", type=float, nargs="*", default=[-0.5, 0.0, 0.5])
    ap.add_argument("--out", default="ir")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.file.endswith(".npy"):
        X = np.load(a.file)
    else:
        df = pd.read_parquet(a.file) if a.file.endswith(".parquet") else pd.read_csv(a.file)
        X = df.select_dtypes("number").to_numpy(float)
    if a.transpose:
        X = X.T
    ir = one_year_ir(X, a.ppy)
    sr = X.mean() / X.std(ddof=1)
    summ, qtab, meta = analyze_ir(ir, a.ppy, sr, a.thresholds, a.gof_n, a.gof_mc, a.seed,
                                  out_prefix=a.out)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(json.dumps({k: v for k, v in meta.items() if k != "params"}, indent=2))
    print(summ.round(4).to_string(index=False))
    print(qtab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
