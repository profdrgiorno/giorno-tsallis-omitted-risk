"""Paper B real-data pilot: robust jump test on SPY daily (2020-2026, full sample) and on the
minute-return windows archived in GT_VaR_SPY / GT_VaR_WIN (test windows only; raw training files not supplied).

Grid: theta = g * s_hat, g in (0.3, 0.6, 0.9, 1.5, 2.5), s_hat the fitted static Student-t scale
(the simulation designs have unit-order scale, so this keeps c = theta/sigma comparable).
Minute data: overnight returns are excluded (they have their own scale); the remaining intraday returns are
tested in time order. Both directions are reported: 'down' (downward impulses) and 'up' (upward impulses).
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.environ.get("PB_UPLOAD", HERE)        # folder with giorno_daily_comparison/, GT_VaR_SPY/, GT_VaR_WIN/, code/
RAW = os.environ.get("PB_RAW", HERE)              # folder with SPY_2026.csv and WIN_M1.csv
import sys
import numpy as np, pandas as pd
sys.path.insert(0, HERE)
import hmm9_srj as H

G_REL = (0.3, 0.6, 0.9, 1.5, 2.5)

def test(y, loc, sc, nu, grid, rng, nsim=50000):
    smu = H._smu(y, loc, sc, nu); dl = 1e-4 * np.median(np.atleast_1d(sc))
    dsmu = -(H._smu(y + dl, loc, sc, nu) - H._smu(y - dl, loc, sc, nu)) / (2 * dl)
    A = []
    for th in grid:
        a = H._afun(y, loc, sc, nu, th, True)
        da = -(H._afun(y + dl, loc, sc, nu, th, True) - H._afun(y - dl, loc, sc, nu, th, True)) / (2 * dl)
        A.append(a - (da.mean() / dsmu.mean()) * smu)
    A = np.array(A); Zs = A.sum(1) / np.sqrt((A ** 2).sum(1))
    C = np.corrcoef(A); L = np.linalg.cholesky(C + 1e-10 * np.eye(len(grid)))
    sims = (L @ rng.standard_normal((len(grid), nsim))).max(0)
    return dict(Z=np.round(Zs, 2).tolist(), p_down=float((sims >= Zs.max()).mean()), p_up=float((sims >= (-Zs).max()).mean()))

def fit_gjr(y):
    """GJR-GARCH(1,1)-t with constant mean: h_t = om + (al + ga*1[e<0]) e^2_{t-1} + be h_{t-1}."""
    from scipy.optimize import minimize
    from scipy.signal import lfilter
    from scipy.stats import t as tdist
    def unpack(u):
        mu = u[0]; om = np.exp(u[1]); a, g, b = np.exp(u[2]), np.exp(u[3]), np.exp(u[4]); s = 1 + a + g / 2 + b
        return mu, om, a / s, g / s, b / s, 2.05 + np.exp(u[5])
    def hpath(mu, om, al, ga, be):
        e = y - mu; x = (al + ga * (e < 0)) * e * e
        return lfilter([1.0], [1.0, -be], np.concatenate([[y.var()], om + x[:-1]]))
    def nll(u):
        mu, om, al, ga, be, nu = unpack(u); h = hpath(mu, om, al, ga, be)
        if not np.all(np.isfinite(h)) or np.any(h <= 0): return 1e10
        sc = np.sqrt(h * (nu - 2) / nu); return -(tdist.logpdf((y - mu) / sc, nu) - np.log(sc)).sum()
    v = y.var(); best = None
    for a0, g0, b0 in ((0.03, 0.08, 0.88), (0.1, 0.1, 0.6)):
        r0 = 1 - a0 - g0 / 2 - b0
        u0 = [np.median(y), np.log(v * r0), np.log(a0 / r0), np.log(g0 / r0), np.log(b0 / r0), np.log(3.0)]
        r = minimize(nll, u0, method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-7, fatol=1e-9))
        if best is None or r.fun < best.fun: best = r
    mu, om, al, ga, be, nu = unpack(best.x); h = hpath(mu, om, al, ga, be)
    return mu, np.sqrt(h * (nu - 2) / nu), nu, (al, ga, be)

def run(name, y, rng):
    loc, s, nu = H.fit_t(y); grid = [g * s for g in G_REL]
    rt = test(y, loc, s, nu, grid, rng)
    gm, gs, gnu, gp = H.fit_garch(y); rg = test(y, gm, gs, gnu, grid, rng)
    jm, js, jnu, jp = fit_gjr(y); rj = test(y, jm, js, jnu, grid, rng)
    q = np.quantile(y, [0.001, 0.999])
    return dict(series=name, T=len(y), skew=float(pd.Series(y).skew()), t_nu=round(nu, 2), garch_nu=round(gnu, 2),
                garch_ab=(round(gp[1], 3), round(gp[2], 3)),
                q001=round(q[0], 4), q999=round(q[1], 4),
                t_Z=rt["Z"], t_p_down=rt["p_down"], t_p_up=rt["p_up"], g_Z=rg["Z"], g_p_down=rg["p_down"], g_p_up=rg["p_up"],
                gjr_agb=tuple(round(v, 3) for v in jp), gjr_Z=rj["Z"], gjr_p_down=rj["p_down"], gjr_p_up=rj["p_up"])

if __name__ == "__main__":
    up = UPLOAD; rng = np.random.default_rng(20261101); rows = []
    d = pd.read_csv(f"{up}/giorno_daily_comparison/daily_prices_and_returns.csv").dropna()
    y = d.Log_Return_Percent.to_numpy(); rows.append(run("SPY daily 2020-01-03..2026-10-28", y, rng))
    post = d[d.Date >= "2021-01-01"].Log_Return_Percent.to_numpy(); rows.append(run("SPY daily 2021..2026 (excl. 2020)", post, rng))
    for mkt in ("SPY", "WIN"):
        v = pd.read_csv(f"{up}/GT_VaR_{mkt}/var_forecasts.csv", usecols=["timestamp", "return_pct_log", "overnight", "elapsed_minutes"])
        intr = v[(~v.overnight) & (v.elapsed_minutes == 1)].return_pct_log.to_numpy()
        rows.append(run(f"{mkt} 1-min intraday, test window ({v.timestamp.iloc[0][:10]}..{v.timestamp.iloc[-1][:10]})", intr, rng))
    out = pd.DataFrame(rows); out.to_csv("realdata_pilot.csv", index=False)
    pd.set_option("display.width", 250); print(out.to_string())
