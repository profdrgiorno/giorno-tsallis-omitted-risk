"""Robust (symmetric) jump test of Paper B applied to the archived HMM-9 Monte Carlo design.

Paths are regenerated with the repository's simulate() logic (copied verbatim below, same SeedSequence),
training segment only (first n observations). Null models: static Student-t and GARCH(1,1)-t, both
symmetric and both misspecified for HMM data. Test exactly as in the Paper B pilot (srj_test.py):
Z ~ Gamma(2,1), theta grid (0.3,0.6,0.9,1.5,2.5), 32-node generalized Laguerre, projection on the
location score, one-sided sup with Gaussian-sup critical values. Also the naive one-sided score h^-/f-1.
Usage: python hmm9_srj.py SCENARIO N REP_START REP_END OUT.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln, roots_genlaguerre
from scipy.stats import t as tdist

# ---- generator: verbatim from giorno_tsallis_complete.py ----
QTRUE = np.array([1.12, 1.42]); NUTRUE = (3.0 - QTRUE) / (QTRUE - 1.0)
STRUE = np.array([0.80, 1.25]); PTRUE = np.array([[0.985, 0.015], [0.040, 0.960]])
NTEST = 400; SEED = 20261010

def simulate(scenario, n, rep):
    rng = np.random.default_rng(np.random.SeedSequence(
        [SEED, {"correct": 0, "gaussian_contamination": 1, "negative_gamma_jumps": 2}[scenario], n, rep]))
    z = np.empty(n + NTEST, dtype=np.int8); y = np.empty(n + NTEST); jump = np.zeros(n + NTEST, dtype=np.int8)
    z[0] = int(rng.random() > 0.7272727272727273)
    for i in range(n + NTEST):
        if i:
            z[i] = int(rng.random() >= PTRUE[z[i - 1], 0])
        yi = rng.standard_t(NUTRUE[z[i]]) * STRUE[z[i]]
        if scenario == "gaussian_contamination" and rng.random() < 0.03:
            jump[i] = 1; yi += rng.normal(0.0, 5.0)
        elif scenario == "negative_gamma_jumps" and rng.random() < 0.03:
            jump[i] = 1; yi -= rng.gamma(shape=2.0, scale=3.0)
        y[i] = yi
    return y, z, jump

# ---- test ----
XN, WN = roots_genlaguerre(32, 1); WN = WN / np.exp(gammaln(2.0))
GRID = (0.3, 0.6, 0.9, 1.5, 2.5)

def tpdf(y, loc, sc, nu):
    return tdist.pdf((y - loc) / sc, nu) / sc

def _afun(y, loc, sc, nu, th, robust):
    f = tpdf(y, loc, sc, nu)
    hm = sum(w * tpdf(y + th * q, loc, sc, nu) for q, w in zip(XN, WN))
    if robust:
        hp = sum(w * tpdf(y - th * q, loc, sc, nu) for q, w in zip(XN, WN)); return (hm - hp) / f
    return hm / f - 1

def _smu(y, loc, sc, nu):
    z = (y - loc) / sc; return ((nu + 1) * z / (nu + z * z)) / sc

def _supp(A, rng, nsim):
    Zs = A.sum(1) / np.sqrt((A ** 2).sum(1))
    C = np.corrcoef(A); L = np.linalg.cholesky(C + 1e-10 * np.eye(len(GRID)))
    sims = (L @ rng.standard_normal((len(GRID), nsim))).max(0)
    return float(Zs.max()), float((sims >= Zs.max()).mean())

def srj(y, loc, sc, nu, rng, robust=True, nsim=20000):
    """Returns (stat, p) for the pilot version (OLS projection on the location score) and
    (stat, p) for the Jacobian-corrected version: xi_t = a_t - kappa s_mu,t with
    kappa = mean(d a/d mu) / mean(d s_mu/d mu), derivatives at fixed scale (d/dmu = -d/dy)."""
    smu = _smu(y, loc, sc, nu); dl = 1e-4
    dsmu = -(_smu(y + dl, loc, sc, nu) - _smu(y - dl, loc, sc, nu)) / (2 * dl)
    A_ols, A_jac = [], []
    for th in GRID:
        a = _afun(y, loc, sc, nu, th, robust)
        da = -(_afun(y + dl, loc, sc, nu, th, robust) - _afun(y - dl, loc, sc, nu, th, robust)) / (2 * dl)
        A_ols.append(a - np.dot(a, smu) / np.dot(smu, smu) * smu)
        A_jac.append(a - (da.mean() / dsmu.mean()) * smu)
    return _supp(np.array(A_ols), rng, nsim) + _supp(np.array(A_jac), rng, nsim)

# ---- null fits ----
def fit_t(y):
    def nll(p):
        loc, ls, lnu = p; nu = 2.05 + np.exp(lnu)
        return -tdist.logpdf((y - loc) / np.exp(ls), nu).sum() + len(y) * ls
    best = None
    for nu0 in (3.0, 8.0):
        r = minimize(nll, [np.median(y), np.log(y.std() * 0.8), np.log(nu0 - 2.05)], method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9))
        if best is None or r.fun < best.fun: best = r
    loc, ls, lnu = best.x
    return loc, np.exp(ls), 2.05 + np.exp(lnu)

def garch_h(p, y):
    mu, om, al, be = p[:4]; e2 = (y - mu) ** 2
    h0 = y.var(); x = np.concatenate([[h0], e2[:-1]])
    # h_t = om + al e2_{t-1} + be h_{t-1}, h_1 = h0
    h = lfilter([1.0], [1.0, -be], np.concatenate([[h0], om + al * x[1:]]))
    return h

def fit_garch(y):
    def unpack(u):
        mu = u[0]; om = np.exp(u[1]); a, b = np.exp(u[2]), np.exp(u[3])
        s = 1 + a + b; al, be = a / s, b / s  # al+be<1
        nu = 2.05 + np.exp(u[4]); return mu, om, al, be, nu
    def nll(u):
        mu, om, al, be, nu = unpack(u); h = garch_h((mu, om, al, be), y)
        if not np.all(np.isfinite(h)) or np.any(h <= 0): return 1e10
        sc = np.sqrt(h * (nu - 2) / nu)
        return -(tdist.logpdf((y - mu) / sc, nu) - np.log(sc)).sum()
    v = y.var(); best = None
    for al0, be0 in ((0.05, 0.9), (0.15, 0.6)):
        u0 = [np.median(y), np.log(v * (1 - al0 - be0)), np.log(al0 / (1 - al0 - be0)), np.log(be0 / (1 - al0 - be0)), np.log(3.0)]
        r = minimize(nll, u0, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-7, fatol=1e-9))
        if best is None or r.fun < best.fun: best = r
    mu, om, al, be, nu = unpack(best.x); h = garch_h((mu, om, al, be), y)
    return mu, np.sqrt(h * (nu - 2) / nu), nu, (om, al, be)

if __name__ == "__main__":
    sc_name, n, r0, r1, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    rows = []
    for rep in range(r0, r1):
        y, _, jump = simulate(sc_name, n, rep); tr = y[:n]
        rng = np.random.default_rng([20261101, n, rep, len(sc_name)])
        loc, s, nu = fit_t(tr)
        rt = srj(tr, loc, s, nu, rng, True); nt = srj(tr, loc, s, nu, rng, False)
        gm, gs, gnu, gpar = fit_garch(tr)
        rg = srj(tr, gm, gs, gnu, rng, True); ng = srj(tr, gm, gs, gnu, rng, False)
        rows.append(dict(scenario=sc_name, n=n, rep=rep, njumps=int(jump[:n].sum()), t_loc=loc, t_scale=s, t_nu=nu,
                         p_rob_t=rt[1], p_robJ_t=rt[3], p_naive_t=nt[1], p_naiveJ_t=nt[3],
                         p_rob_g=rg[1], p_robJ_g=rg[3], p_naive_g=ng[1], p_naiveJ_g=ng[3],
                         stat_rob_t=rt[0], stat_robJ_t=rt[2], stat_rob_g=rg[0], stat_robJ_g=rg[2],
                         g_mu=gm, g_nu=gnu, g_alpha=gpar[1], g_beta=gpar[2]))
        p_r, pg_r = rt[3], rg[3]
        print(sc_name, n, rep, round(p_r, 3), round(pg_r, 3), flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
