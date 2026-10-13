"""
Joint stationary bootstrap (Politis-Romano 1994) of fund + benchmark daily returns,
then the 1y IR analysis per fund type via ir_cube.run_cube.

Input : funds.csv       date, <name><k> level columns; the LAST character k links a fund to BMK<k>
        benchmarks.csv  date, BMK<k> levels (from build_benchmarks.py). Funds with no BMK<k> are dropped.
Output: {out}_best_fit.csv      one row per fund: TE, IR moments, best model + params, P(IR<thr), q05
        {out}_fit_summary.csv   every candidate model x fund
        {out}_draws.csv         raw 1y IR draws, one column per fund (joint questions, correlations)
        {out}_cdf.png           3 x 4 grid: empirical CDF, DKW band, best fit
        {out}_sims.npy          S x T x N cube, only with --save_cube (seed makes it reproducible)

All N series are resampled with the SAME block indices per path, so fund-vs-benchmark
co-movement is preserved. Blocks wrap around the end of the sample (circular), as in the paper.
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ir_cube import run_cube


def _read_levels(path):
    """CSV with a date column (dd/mm/yyyy or ISO) + level columns -> sorted DataFrame."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format="mixed", dayfirst=True)
    return df.sort_index()


def load_returns(start="2015-01-01", funds="funds.csv", bench="benchmarks.csv"):
    f, b = _read_levels(funds), _read_levels(bench)
    b = b.reindex(f.index).ffill()                            # benchmark levels on the funds calendar
    groups = {c[-1] for c in b if c.startswith("BMK")}       # BMK<k> <-> funds whose name ends in k
    f = f[[c for c in f if c[-1] in groups]]                  # funds with no benchmark are dropped
    df = f.join(b, how="inner").loc[start:].dropna()
    return df.pct_change().dropna()


def stationary_bootstrap_idx(n, T, S, mean_block, rng):
    """S x T integer indices into 0..n-1; block lengths ~ Geometric(1/mean_block)."""
    p = 1.0 / mean_block
    idx = np.empty((S, T), dtype=np.int64)
    idx[:, 0] = rng.integers(n, size=S)
    for t in range(1, T):
        restart = rng.random(S) < p
        idx[:, t] = np.where(restart, rng.integers(n, size=S), (idx[:, t - 1] + 1) % n)
    return idx


def build_cube(ret, S, T, mean_block, seed):
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(len(ret), T, S, mean_block, rng)
    return ret.to_numpy()[idx], list(ret.columns)            # S x T x N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01", help="first date of the resampling window")
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--T", type=int, default=252, help="path length in periods (1y = ppy)")
    ap.add_argument("--ppy", type=int, default=252, help="periods per year of the input data")
    ap.add_argument("--block", type=float, default=63, help="mean block length in days")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gof_mc", type=int, default=0, help="bootstrap AD replicates; 0 = skip")
    ap.add_argument("--out", default="ir")
    ap.add_argument("--save_cube", action="store_true")
    a = ap.parse_args()

    ret = load_returns(a.start)
    print(f"resampling window: {ret.index[0].date()} -> {ret.index[-1].date()}, {len(ret)} days, "
          f"{ret.shape[1]} series")
    R, cols = build_cube(ret, a.paths, a.T, a.block, a.seed)
    if a.save_cube:
        np.save(f"{a.out}_sims.npy", R)
        json.dump(cols, open(f"{a.out}_sims_cols.json", "w"))

    bests, draws, summs = [], [], []
    for k in sorted({c[-1] for c in cols if c.startswith("BMK")}):
        sel = [i for i, c in enumerate(cols) if c[-1] == k]
        names = [cols[i] for i in sel]
        b, d, s = run_cube(R[..., sel], ppy=a.ppy, names=names, out=None, plot=False,
                           bench_col=names.index(f"BMK{k}"), gof_mc=a.gof_mc, seed=a.seed)
        bests.append(b.assign(benchmark=f"BMK{k}")); draws.append(d); summs.append(s)
    best, draws = pd.concat(bests, ignore_index=True), pd.concat(draws, axis=1)
    best.to_csv(f"{a.out}_best_fit.csv", index=False)
    pd.concat(summs, ignore_index=True).to_csv(f"{a.out}_fit_summary.csv", index=False)
    draws.to_csv(f"{a.out}_draws.csv", index=False)
    plot_grid(best, draws, f"{a.out}_cdf.png")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(best.drop(columns="best_params").round(3).to_string(index=False))


def _dist(model, params):
    fam = getattr(stats, model.split("_")[0])            # nct_theory / nct_free -> nct
    return fam(*params)


def plot_grid(best, draws, path):
    """Small multiples: empirical CDF (ink), DKW 95% band (gray), best fit (accent)."""
    funds = list(best.fund)
    ncol = 4; nrow = -(-len(funds) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (_, row) in zip(axes.flat, best.iterrows()):
        xs = np.sort(draws[row.fund].dropna().to_numpy()); n = xs.size
        ecdf = np.arange(1, n + 1) / n
        dkw = np.sqrt(np.log(2 / 0.05) / (2 * n))
        ax.fill_between(xs, np.clip(ecdf - dkw, 0, 1), np.clip(ecdf + dkw, 0, 1),
                        step="post", color="#9A9A9A", alpha=0.3, lw=0, label="DKW 95%")
        ax.step(xs, ecdf, where="post", color="#1F1F1F", lw=1.2, label="Empirical")
        grid = np.linspace(xs[0], xs[-1], 300)
        d = _dist(row.best_model, json.loads(row.best_params))
        ax.plot(grid, d.cdf(grid), color="#2F6DB5", lw=1.6, label="Best fit")
        ax.axvline(0, color="#9A9A9A", lw=0.6)
        ax.set_title(f"{row.fund} vs {row.benchmark} | {row.best_model} | "
                     f"P(IR<0)={row['P(IR<0.0)_emp']:.2f}", fontsize=8.5)
        ax.grid(color="#E6E6E6", lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for ax in axes.flat[len(funds):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("1y Information Ratio", fontsize=8)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("1-year IR: empirical CDF vs best parametric fit", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97)); fig.savefig(path, dpi=130); plt.close(fig)


def _check():
    rng = np.random.default_rng(0)
    n, T, S, mb = 500, 252, 2000, 63
    idx = stationary_bootstrap_idx(n, T, S, mb, rng)
    assert idx.min() >= 0 and idx.max() < n
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % n).mean()        # P(continue block) = 1 - 1/mb
    assert abs((1 - cont) - 1 / mb) < 0.002, cont
    ret = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("abc"))
    ret["c"] = ret["a"] * 2                                      # exact linear relation...
    R, _ = build_cube(ret, 10, T, mb, 1)
    assert np.allclose(R[..., 2], 2 * R[..., 0])                 # ...survives joint resampling
    print("check ok")


if __name__ == "__main__":
    _check() if "--check" in sys.argv else main()
