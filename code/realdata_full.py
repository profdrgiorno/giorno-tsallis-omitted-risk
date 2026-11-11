"""Robust jump test on the full minute samples (raw files), using the loader of GT_VaR_StudentT_vs_HMM6.py.
Intraday 1-minute returns only (overnight and same-day non-minute returns excluded). Full sample and the
80/20 train/test split used in GT_VaR_*."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.environ.get("PB_UPLOAD", HERE)        # folder with giorno_daily_comparison/, GT_VaR_SPY/, GT_VaR_WIN/, code/
RAW = os.environ.get("PB_RAW", HERE)              # folder with SPY_2026.csv and WIN_M1.csv
import sys, importlib.util, numpy as np, pandas as pd
sys.path.insert(0, HERE)
import realdata as R
spec = importlib.util.spec_from_file_location("v", os.path.join(HERE, "GT_VaR_StudentT_vs_HMM6.py"))
src = open(os.path.join(HERE, "GT_VaR_StudentT_vs_HMM6.py")).read()
ns = {}; exec(src[src.index("def load_prices"):src.index("def log_emission")], {"pd": pd, "np": np}, ns)
rng = np.random.default_rng(20261102); rows = []
for mkt, path, fmt in (("SPY", os.path.join(RAW, "SPY_2026.csv"), "spy"), ("WIN", os.path.join(RAW, "WIN_M1.csv"), "win")):
    y, gap, times, audit, mins = ns["load_prices"](path, fmt); print(mkt, audit, flush=True)
    split = int(0.8 * len(y)); keep = (~gap) & (mins == 1)
    for lab, sl in (("full", slice(None)), ("train 80%", slice(0, split)), ("test 20%", slice(split, None))):
        yy = y[sl][keep[sl]]; t = times[sl]
        rows.append(R.run(f"{mkt} 1-min intraday, {lab} ({str(t[0])[:10]}..{str(t[-1])[:10]})", yy, rng)); print(rows[-1], flush=True)
out = pd.DataFrame(rows); out.to_csv("realdata_minute_full.csv", index=False)
