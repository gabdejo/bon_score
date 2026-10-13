"""
Adapter: S x T x N cube of simulated fund returns  ->  active returns  ->  1y IR fit per fund.

    R[s, t, i] = simple return of fund i on day t in simulation s   (decimals, not %)

Active-return definitions (pick one):
  bench_col=k       active_i = r_i - r_k            (benchmark is one of the N series; it is
                                                     dropped from the output)
  bench=array       active_i = r_i - b              (b: S x T, same bootstrap draws as R,
                                                     or T for a fixed path)
  peer="loo"        active_i = r_i - weighted mean of the OTHER funds (leave-one-out)
  peer="all"        active_i = r_i - weighted mean of ALL funds (incl. itself)
  active_weights=w  single active return = R @ w   (N = assets, w = w_port - w_bench)

IMPORTANT: fund and benchmark returns must come from the SAME bootstrap indices
(same days resampled jointly), otherwise the active return gets spurious tracking error.

Outputs (prefix --out; out=None writes nothing and just returns the tables):
  {out}_ir_draws.csv          1y IR draws, one column per fund
  {out}_ir_corr.csv           cross-fund correlation of 1y IR draws
  {out}_fit_summary.csv       fit statistics, all funds stacked
  {out}_quantiles.csv         empirical vs fitted quantiles, all funds stacked
  {out}_best_fit.csv          one row per fund: best model (BIC), params, key probabilities
  {out}_{fund}_fit.png        diagnostics per fund
"""
import argparse, json
import numpy as np
import pandas as pd
from ir_cdf_fit import one_year_ir, analyze_ir


def active_from_cube(R, names=None, bench_col=None, bench=None, peer=None,
                     peer_weights=None, active_weights=None):
    """Return (A, names): A is S x T x M active returns."""
    R = np.asarray(R, float)
    if R.ndim != 3:
        raise ValueError(f"Expected S x T x N, got shape {R.shape}")
    S, T, N = R.shape
    names = list(names) if names is not None else [f"fund{i}" for i in range(N)]
    modes = sum(x is not None for x in (bench_col, bench, peer, active_weights))
    if modes != 1:
        raise ValueError("Choose exactly one of bench_col, bench, peer, active_weights.")

    if active_weights is not None:
        w = np.asarray(active_weights, float)
        return (R @ w)[..., None], ["active"]

    if bench_col is not None:
        keep = [i for i in range(N) if i != bench_col]
        A = R[..., keep] - R[..., [bench_col]]
        return A, [names[i] for i in keep]

    if bench is not None:
        b = np.asarray(bench, float)
        if b.shape == (T,):
            b = np.broadcast_to(b, (S, T))
        if b.shape != (S, T):
            raise ValueError(f"bench must be S x T or T, got {b.shape}")
        return R - b[..., None], names

    # peer average (e.g. AFP system / competitors)
    w = np.ones(N) if peer_weights is None else np.asarray(peer_weights, float)
    tot = R @ w                                   # S x T, weighted sum
    if peer == "all":
        return R - (tot / w.sum())[..., None], names
    if peer == "loo":
        others = (tot[..., None] - R * w) / (w.sum() - w)   # S x T x N
        return R - others, names
    raise ValueError("peer must be 'loo' or 'all'")


def run_cube(R, ppy=252, names=None, out="cube", thresholds=(-0.5, 0.0, 0.5),
             gof_n=1000, gof_mc=99, seed=42, models=None, plot=True, **active_kw):
    A, names = active_from_cube(R, names, **active_kw)
    S, T, M = A.shape
    ir_cols, summaries, quants, best_rows = {}, [], [], []
    for i, nm in enumerate(names):
        a = A[..., i]                             # S x T
        ir = one_year_ir(a, ppy)
        ir_cols[nm] = ir
        sr = a.mean() / a.std(ddof=1)
        summ, qtab, meta = analyze_ir(ir, ppy, sr, thresholds, gof_n, gof_mc, seed,
                                      models=models, out_prefix=None, plot=False, label=nm)
        if plot and out:  # re-run plotting only via analyze_ir's writer, without the extra files
            from ir_cdf_fit import fit_candidates, _plot
            xs = np.sort(ir); n = xs.size
            fits = fit_candidates(ir, ppy, sr, models)
            _plot(xs, np.arange(1, n + 1) / n, meta["dkw95_halfwidth"], fits,
                  meta["best_by_bic"], f"{out}_{nm}_fit.png", nm)
        summaries.append(summ); quants.append(qtab)
        b = summ.iloc[0]
        best_rows.append({
            "fund": nm, "n_ir_draws": meta["n_ir_draws"],
            "active_ret_ann": a.mean() * ppy, "TE_ann": a.std(ddof=1) * np.sqrt(ppy),
            "ir_mean": meta["ir_mean"], "ir_sd": meta["ir_sd"],
            "ir_skew": meta["ir_skew"], "ir_exkurt": meta["ir_exkurt"],
            "best_model": b["model"], "best_params": json.dumps(meta["params"][b["model"]]),
            "ks_D": b["ks_D"], "within_DKW95": b["within_DKW95"],
            **{c: b[c] for c in summ.columns if c.startswith("P(IR<")},
            "q05_emp": float(qtab.loc[qtab.q == 0.05, "empirical"].iloc[0]),
            "q05_fit": float(qtab.loc[qtab.q == 0.05, b["model"]].iloc[0]),
        })

    L = min(len(v) for v in ir_cols.values())
    draws = pd.DataFrame({k: v[:L] for k, v in ir_cols.items()})
    best, summ_all = pd.DataFrame(best_rows), pd.concat(summaries)
    if out:                                   # out=None: return tables only, write nothing
        draws.to_csv(f"{out}_ir_draws.csv", index=False)
        draws.corr().to_csv(f"{out}_ir_corr.csv")
        summ_all.to_csv(f"{out}_fit_summary.csv", index=False)
        pd.concat(quants).to_csv(f"{out}_quantiles.csv", index=False)
        best.to_csv(f"{out}_best_fit.csv", index=False)
    return best, draws, summ_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube", help=".npy file, shape S x T x N (simple returns, decimals)")
    ap.add_argument("--ppy", type=int, default=252)
    ap.add_argument("--names", nargs="*")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bench-col", type=int, help="index of benchmark series in N")
    g.add_argument("--bench-file", help=".npy benchmark, S x T or T")
    g.add_argument("--peer", choices=["loo", "all"], help="active vs peer average")
    g.add_argument("--active-weights", type=float, nargs="+", help="w_port - w_bench over N")
    ap.add_argument("--peer-weights", type=float, nargs="*", help="e.g. AUM weights for peer mean")
    ap.add_argument("--thresholds", type=float, nargs="*", default=[-0.5, 0.0, 0.5])
    ap.add_argument("--gof_n", type=int, default=1000)
    ap.add_argument("--gof_mc", type=int, default=99, help="0 = skip bootstrap AD test")
    ap.add_argument("--out", default="cube")
    a = ap.parse_args()

    R = np.load(a.cube, mmap_mode="r")
    kw = dict(bench_col=a.bench_col, peer=a.peer, peer_weights=a.peer_weights,
              bench=np.load(a.bench_file) if a.bench_file else None,
              active_weights=a.active_weights)
    best, draws, _ = run_cube(np.asarray(R), a.ppy, a.names, a.out, a.thresholds,
                              a.gof_n, a.gof_mc, **kw)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(best.drop(columns="best_params").round(4).to_string(index=False))
    print("\n1y IR cross-correlation:\n", draws.corr().round(2))


if __name__ == "__main__":
    main()
