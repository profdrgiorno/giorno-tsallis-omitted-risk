"""Sampling-frequency check: robust jump test on intraday k-minute returns (k = 1, 5, 15, 30) built from the
raw minute closes within each day (overnight returns excluded), full sample."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.environ.get("PB_UPLOAD", HERE)        # folder with giorno_daily_comparison/, GT_VaR_SPY/, GT_VaR_WIN/, code/
RAW = os.environ.get("PB_RAW", HERE)              # folder with SPY_2026.csv and WIN_M1.csv
import sys, numpy as np, pandas as pd
sys.path.insert(0, HERE); import realdata as R
def load(path, fmt):
    if fmt == "win":
        df = pd.read_csv(path, sep="\t"); t = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"], format="%Y.%m.%d %H:%M:%S"); p = df["<CLOSE>"].astype(float)
    else:
        df = pd.read_csv(path); t = pd.to_datetime(df["datetime"]); p = df["Close"].astype(float)
    return pd.Series(p.values, index=t).sort_index()
rng = np.random.default_rng(20261103); rows = []
for mkt, path, fmt in (("SPY", os.path.join(RAW, "SPY_2026.csv"), "spy"), ("WIN", os.path.join(RAW, "WIN_M1.csv"), "win")):
    s = load(path, fmt)
    for k in (5, 15, 30):
        bars = s.groupby([s.index.date, s.index.floor(f"{k}min")]).last()   # last close in each k-min bin, per day
        lp = np.log(bars.values); day = np.array([d for d, _ in bars.index])
        r = 100 * np.diff(lp); same = day[1:] == day[:-1]; y = r[same]
        rows.append(R.run(f"{mkt} {k}-min intraday, full", y, rng)); print(rows[-1]["series"], rows[-1]["T"], rows[-1]["t_p_down"], rows[-1]["g_p_down"], rows[-1]["gjr_p_down"], flush=True)
pd.DataFrame(rows).to_csv("realdata_freq.csv", index=False)
