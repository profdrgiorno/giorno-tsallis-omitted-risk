"""Paper B (JBF revision): sensitivity of the jump fit to the upper bound on the jump intensity.

Full-sample GARCH(1,1)-t with Bernoulli-Gamma downward jumps, with the bound on eps set to 0.10, 0.25 and 0.50.
Reports eps, mean jump size k*theta, shape k, log-likelihood, and the Vuong statistic against the skew-t.
Usage: python -I sens_bound.py SERIES.csv LABEL OUT.json
"""
import sys, json, os, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln, roots_genlaguerre
from scipy.stats import t as tdist
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import procedure as P, hmm9_srj as H

def ll_bound(u, y, EB, per_obs=False):
    mu, om = u[0], np.exp(u[1]); a, b = np.exp(u[2]), np.exp(u[3]); s = 1 + a + b; al, be = a / s, b / s
    nu = 2.05 + np.exp(u[4]); eps = EB / (1 + np.exp(-u[5])); th = np.exp(u[6]); k = 0.3 + np.exp(u[7])
    h = lfilter([1.0], [1.0, -be], np.concatenate([[y.var()], om + al * ((y - mu) ** 2)[:-1]]))
    if not np.all(np.isfinite(h)) or np.any(h <= 0): return -1e10 if not per_obs else None
    sc = np.sqrt(h * (nu - 2) / nu)
    x, w = roots_genlaguerre(32, k - 1); w = w / np.exp(gammaln(k))
    f = tdist.pdf((y - mu) / sc, nu) / sc
    hj = sum(wi * tdist.pdf((y - mu + th * xi) / sc, nu) / sc for xi, wi in zip(x, w))
    l = np.log(np.maximum((1 - eps) * f + eps * hj, 1e-300))
    return l if per_obs else l.sum()

def run(y, label):
    gm, gs, gnu, (om, al, be) = H.fit_garch(y); r = 1 - al - be
    base = [gm, np.log(om), np.log(al / r), np.log(be / r), np.log(gnu - 2.05)]
    fs = P.fit_skew_garch(y, (gm, om, al, be, gnu)); l_skew = P.skew_garch_ll(fs.x, y, per_obs=True)
    out = dict(series=label, T=len(y), loglik_skew=float(-fs.fun), gamma=float(np.exp(fs.x[5])))
    for EB in (0.10, 0.25, 0.50):
        best = None
        for e0, t0, k0 in ((0.01, 1.0, 2.0), (min(0.3 * EB, 0.15), 0.5, 2.0), (0.6 * EB, 0.4, 1.0)):
            u0 = base + [np.log(e0 / (EB - e0)), np.log(t0), np.log(k0 - 0.3)]
            rr = minimize(lambda u: -ll_bound(u, y, EB), u0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-8))
            if best is None or rr.fun < best.fun: best = rr
        u = best.x; eps = EB / (1 + np.exp(-u[5])); th = np.exp(u[6]); k = 0.3 + np.exp(u[7])
        l_jump = ll_bound(u, y, EB, per_obs=True)
        out[f"bound_{EB}"] = dict(eps=float(eps), at_bound=bool(eps > 0.99 * EB), theta=float(th), k=float(k), mean_jump=float(k * th),
                                  loglik=float(-best.fun), vuong_skew_vs_jump=P.vuong(l_skew, l_jump, 6, 8))
        print(label, EB, out[f"bound_{EB}"], flush=True)
    return out

if __name__ == "__main__":
    f, label, outf = sys.argv[1:4]
    d = pd.read_csv(f).dropna(subset=["Log_Return_Percent"]); y = d["Log_Return_Percent"].to_numpy(float)
    json.dump(run(y, label), open(outf, "w"), indent=1)
