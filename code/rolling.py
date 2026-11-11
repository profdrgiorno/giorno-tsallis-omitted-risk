"""Paper B (JBF revision): rolling out-of-sample VaR.

Rolling window of W daily returns, models re-estimated every REFIT days (warm start from the previous fit),
one-day-ahead 99% and 95% VaR for each day of the evaluation period:
  GARCH(1,1)-t, GARCH(1,1) with Fernandez-Steel skew-t innovations, GARCH(1,1)-t with Bernoulli-Gamma jumps.
Usage: python -I rolling.py SERIES.csv LABEL OUT.csv [START=2010-01-01] [W=2500] [REFIT=21]   (returns in percent)
Resumable: refit blocks already in OUT.csv are skipped.
"""
import sys, os, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import t as tdist
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import procedure as P, hmm9_srj as H

LEVELS = (0.01, 0.05)

def garch_h(y, mu, om, al, be, h0):
    return lfilter([1.0], [1.0, -be], np.concatenate([[h0], om + al * (y - mu)[:-1] ** 2]))

def fit_t_garch(y, u0=None):
    """GARCH(1,1)-t in the same unconstrained parametrisation as the other two models."""
    def ll(u):
        mu, om = u[0], np.exp(u[1]); a, b = np.exp(u[2]), np.exp(u[3]); s = 1 + a + b; al, be = a / s, b / s
        nu = 2.05 + np.exp(u[4]); h = garch_h(y, mu, om, al, be, y.var())
        if not np.all(np.isfinite(h)) or np.any(h <= 0): return 1e10
        sc = np.sqrt(h * (nu - 2) / nu); return -(tdist.logpdf((y - mu) / sc, nu) - np.log(sc)).sum()
    if u0 is None:
        gm, gs, gnu, (om, al, be) = H.fit_garch(y); r = 1 - al - be
        u0 = [gm, np.log(om), np.log(al / r), np.log(be / r), np.log(gnu - 2.05)]
    r = minimize(ll, u0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    return r.x

def unpack_common(u):
    mu, om = u[0], np.exp(u[1]); a, b = np.exp(u[2]), np.exp(u[3]); s = 1 + a + b
    return mu, om, a / s, b / s, 2.05 + np.exp(u[4])

def run(f, label, outf, start="2010-01-01", W=2500, REFIT=21):
    d = pd.read_csv(f).dropna(subset=["Log_Return_Percent"]).reset_index(drop=True)
    y = d["Log_Return_Percent"].to_numpy(float); dates = d["Date"].to_numpy()
    i0 = max(int(np.searchsorted(dates, start)), W)
    done = set()
    if os.path.exists(outf):
        old = pd.read_csv(outf); done = set(old["block"].unique())
    first = not os.path.exists(outf)
    ug = us = uj = None
    for blk, i in enumerate(range(i0, len(y), REFIT)):
        tr = y[i - W:i]
        ug = fit_t_garch(tr, ug)
        mu, om, al, be, nu = unpack_common(ug)
        gs = (mu, om, al, be, nu)
        if us is None:
            us = P.fit_skew_garch(tr, gs).x
            uj = P.fit_jump_garch(tr, gs, 1.0).x
        else:
            us = minimize(lambda u: -P.skew_garch_ll(u, tr), us, method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-6, fatol=1e-8)).x
            uj = minimize(lambda u: -P.jump_garch_ll(u, tr), uj, method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-5, fatol=1e-7)).x
        if blk in done: continue
        j = min(i + REFIT, len(y)); seg = y[i - W:j]; h0 = tr.var()
        rows = dict(date=dates[i:j], ret=y[i:j], block=blk)
        # GARCH-t
        hg = garch_h(seg, mu, om, al, be, h0)[W:]; sg = np.sqrt(hg * (nu - 2) / nu)
        # skew-t
        ms, oms, als, bes, nus = unpack_common(us); gam = float(np.exp(us[5]))
        hs = garch_h(seg, ms, oms, als, bes, h0)[W:]; ss = np.sqrt(hs * (nus - 2) / nus)
        # jumps
        pj = P.natural(uj); hj = garch_h(seg, pj["mu"], pj["omega"], pj["alpha"], pj["beta"], h0)[W:]
        sj = np.sqrt(hj * (pj["nu"] - 2) / pj["nu"])
        for a in LEVELS:
            tag = f"{int(round(100 * (1 - a)))}"
            rows[f"q_t_{tag}"] = mu + sg * tdist.ppf(a, nu)
            rows[f"q_skew_{tag}"] = np.array([P.skew_quantile(a, ms, s_, nus, gam) for s_ in ss])
            rows[f"q_jump_{tag}"] = P.var_jump(a, pj["mu"], sj, pj["nu"], pj)
        rows.update(nu_t=nu, gamma_skew=gam, eps_jump=pj["eps"], theta_jump=pj["theta"], k_jump=pj["k"])
        pd.DataFrame(rows).to_csv(outf, mode="a", header=first, index=False); first = False
        print(label, dates[i], f"eps={pj['eps']:.4f} gamma={gam:.3f} nu={nu:.2f}", flush=True)

if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], a[1], a[2], *(a[3:4] or ["2010-01-01"]), *( [int(x) for x in a[4:6]] ))
