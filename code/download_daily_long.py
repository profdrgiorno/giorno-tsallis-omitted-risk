"""Download long daily histories for Paper B (items C1-C2) from Yahoo Finance.

Series (adjusted closes, auto_adjust=True):
  SPY    from 1993-01-29 (first trading day of the ETF)
  ^GSPC  from 1990-01-01 (S&P 500 index)
  ^BVSP  from 1995-01-01 (Ibovespa; starts after the Real Plan to avoid the hyperinflation years and
         index redenominations, which would look like enormous jumps)
Each file: Date, Adjusted_Close, Log_Return_Percent = 100 * log(P_t / P_{t-1}).
A short audit (rows, dates, gaps, largest moves) is printed and saved next to the data.

Usage:  pip install yfinance pandas numpy      (once)
        python download_daily_long.py
Output: SPY_daily_1993_2026.csv, GSPC_daily_1990_2026.csv, BVSP_daily_1995_2026.csv, download_audit.json
"""
import json
import numpy as np
import pandas as pd
import yfinance as yf

SERIES = [("SPY", "1993-01-29", "SPY_daily_1993_2026.csv"),
          ("^GSPC", "1990-01-01", "GSPC_daily_1990_2026.csv"),
          ("^BVSP", "1995-01-01", "BVSP_daily_1995_2026.csv")]
END = None   # None = up to today

audit = {}
for ticker, start, fname in SERIES:
    raw = yf.download(ticker, start=start, end=END, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        print(f"{ticker}: no data returned (check the internet connection or the ticker)"); continue
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance versions return a one-column frame
        close = close.iloc[:, 0]
    close = close.dropna()
    close = close[close > 0]
    df = pd.DataFrame({"Date": close.index.strftime("%Y-%m-%d"), "Adjusted_Close": close.to_numpy(dtype=float)})
    df = df.drop_duplicates("Date").reset_index(drop=True)
    df["Log_Return_Percent"] = 100 * np.log(df.Adjusted_Close).diff()
    df.to_csv(fname, index=False, float_format="%.10g")
    r = df.Log_Return_Percent.dropna(); d = pd.to_datetime(df.Date)
    gaps = d.diff().dt.days
    big = df.loc[r.abs().nlargest(5).index, ["Date", "Log_Return_Percent"]]
    audit[ticker] = dict(file=fname, rows=int(len(df)), returns=int(len(r)), first=df.Date.iloc[0], last=df.Date.iloc[-1],
                         gaps_over_5_days=int((gaps > 5).sum()), zero_returns=int((r == 0).sum()),
                         mean=float(r.mean()), sd=float(r.std()), min=float(r.min()), max=float(r.max()),
                         largest_moves=big.to_dict("records"))
    print(f"{ticker}: {len(r)} returns, {df.Date.iloc[0]} .. {df.Date.iloc[-1]}, sd {r.std():.3f}%, "
          f"min {r.min():.2f}%, max {r.max():.2f}%, gaps>5 days: {(gaps > 5).sum()}  -> {fname}")
json.dump(audit, open("download_audit.json", "w"), indent=1, default=str)
print("Audit saved to download_audit.json. Please send the three CSV files and the audit.")
