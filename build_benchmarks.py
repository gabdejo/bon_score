"""
Build fixed-weight (daily rebalanced) benchmarks in PEN from Yahoo Finance prices.

Input : weights.csv  columns: benchmark, ticker, weight[, currency]   (currency default USD)
        USDPEN from BCRP (official SBS selling rate, series PD04640PD); Yahoo's PEN=X has bad ticks.
Output: prices_raw.csv     adjusted close per ticker (Yahoo) + USDPEN (BCRP), union calendar
        benchmarks.csv     benchmark levels in PEN (base 100), aligned to funds.csv dates

Benchmark return on day t = sum_i w_i * r_i,t  with r_i in PEN (constant mix, rebalanced daily).
Levels are forward-filled across holiday mismatches before returns are taken.
"""
import sys, json, urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

FX = "USDPEN"          # soles per dollar
BCRP_SERIES = "PD04640PD"   # TC Sistema bancario SBS - Venta
_MES = dict(Ene=1, Feb=2, Mar=3, Abr=4, May=5, Jun=6, Jul=7, Ago=8, Set=9, Oct=10, Nov=11, Dic=12)


def bcrp_fx(start="2001-01-01", end="2030-12-31"):
    url = f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{BCRP_SERIES}/json/{start}/{end}"
    d = json.load(urllib.request.urlopen(url, timeout=60))
    rows = []
    for p in d["periods"]:
        dd, mm, yy = p["name"].split(".")
        v = p["values"][0]
        if v not in ("n.d.", None):
            rows.append((pd.Timestamp((1900 if int(yy) >= 50 else 2000) + int(yy), _MES[mm], int(dd)), float(v)))
    return pd.Series(dict(rows), name=FX).sort_index()


def load_weights(path="weights.csv"):
    w = pd.read_csv(path)
    w.columns = [c.strip().lower() for c in w.columns]
    if "currency" not in w:
        w["currency"] = "USD"
    w["currency"] = w["currency"].fillna("USD").str.upper()
    w["ticker"] = w["ticker"].str.strip()
    sums = w.groupby("benchmark")["weight"].sum()
    bad = sums[(sums - 1).abs() > 1e-6]
    if len(bad):
        sys.exit(f"Weights do not sum to 1:\n{bad}")
    return w


def download(tickers, start="1990-01-01"):
    px = yf.download(sorted(set(tickers)), start=start, auto_adjust=True,
                     progress=False)["Close"]
    missing = [t for t in tickers if t not in px or px[t].dropna().empty]
    if missing:
        sys.exit(f"No Yahoo data for: {missing}")
    return px.join(bcrp_fx(start), how="outer")


def build(w, px):
    px = px.sort_index().ffill()
    fx = px[FX]
    out = {}
    for b, g in w.groupby("benchmark", sort=False):
        lv = pd.DataFrame({t: px[t] * (fx if c == "USD" else 1.0)
                           for t, c in zip(g.ticker, g.currency)})
        lv = lv.dropna()                               # start when all constituents exist
        r = lv.pct_change().dropna() @ g.weight.values
        out[b] = 100 * (1 + r).cumprod()
    return pd.DataFrame(out)


def main():
    w = load_weights()
    px = download(w.ticker.tolist())
    px.to_csv("prices_raw.csv")
    bench = build(w, px)
    funds_dates = pd.to_datetime(pd.read_csv("funds.csv", usecols=["date"])["date"], dayfirst=True)
    aligned = bench.reindex(funds_dates.sort_values()).ffill()
    aligned = aligned.loc[aligned.notna().any(axis=1)]
    aligned.index.name = "date"
    aligned.to_csv("benchmarks.csv")
    print(aligned.describe().T[["count", "min", "max"]])
    print("first valid date per benchmark:\n", aligned.apply(lambda s: s.first_valid_index()))


def _check():
    # one-day constant-mix identity: bench return == w . r  (two USD tickers + one PEN ticker)
    idx = pd.date_range("2020-01-01", periods=4)
    px = pd.DataFrame({"A": [10, 11, 11, 12], "B": [20, 20, 22, 22], "C": [5, 5, 5, 6],
                       FX: [3.0, 3.0, 3.3, 3.3]}, index=idx, dtype=float)
    w = pd.DataFrame({"benchmark": ["X"] * 3, "ticker": list("ABC"),
                      "weight": [0.5, 0.3, 0.2], "currency": ["USD", "USD", "PEN"]})
    lv = build(w, px)["X"]
    # day 3: A 0%, B +10%, C 0% in USD; fx +10%  -> A +10%, B +21%, C 0% in PEN
    assert abs(lv.iloc[1] / lv.iloc[0] - 1 - (0.5 * 0.10 + 0.3 * 0.21 + 0.2 * 0.0)) < 1e-12
    print("check ok")


if __name__ == "__main__":
    _check() if "--check" in sys.argv else main()
