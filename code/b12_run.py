"""Paper B, roadmap items B1 (asymmetry versus jumps) and B2 (competitor tests), on the 4,000-path HMM-6 manifest.

For each manifest path (regenerated exactly; first n_train = 4,800 observations used) four arms:
  sym    : the HMM path itself (conditionally symmetric)                       -> size
  jump   : downward jumps, eps = 0.02, size 0.9*Gamma(2,1) (as in the full run) -> power against jumps
  jumpbig: downward jumps, eps = 0.01, size 3*Gamma(2,1) (mean 6, as in HMM-9)     -> separation when jumps are large
  skew90 : Fernandez-Steel skewed emissions, gamma = 0.90, regime-centred, no jumps
  skew95 : same with gamma = 0.95 (milder)                                       -> power against skewness
Skew arms keep |X_t| and the regime path of the manifest path and redraw only the sign (stream [seed, 20261103]),
then centre each regime's emission at zero, so the conditional mean is unchanged and only the shape is asymmetric.

Tests on each arm (5% level, one-sided for downward asymmetry unless stated):
  rob_t, rob_g : robust jump test (orthogonalised), static Student-t and GARCH(1,1)-t nulls
  naive_g      : naive downward-only score, GARCH-t null
  ks_g         : symmetry test on GARCH-t standardised residuals, sup_x |F(x) - (1 - F(-x))|, two-sided,
                 sign-flip (symmetrisation) critical values, 199 flips; a Bai-Ng (2001)-type statistic without
                 their martingale transformation
  skew_g       : skewness test on GARCH standardised residuals, z^3 - 3z with Newey-West variance (Bai-Ng 2005
                 type); one-sided (negative skewness). Needs E z^6 < inf, which fails for nu < 6.
Follow-up (B1): with the GARCH(1,1)-t volatility filter held fixed, Fernandez-Steel skew-t innovations versus
Student-t innovations plus Bernoulli-Gamma(2) downward jumps; Vuong (1989) statistic with Newey-West variance and
Schwarz correction. V > 0 favours skew-t, V < 0 favours jumps.

Usage: python -I b12_run.py WORKER NWORKERS OUT.csv      (resumable; HMM6_REPO = folder with DGP_MANIFEST.csv)
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln, roots_genlaguerre
from scipy.stats import t as tdist
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hmm9_srj as H

REPO = os.environ.get("HMM6_REPO", HERE)
OPTS = json.load(open(os.path.join(HERE, "scenarios.json")))
EPS, THETA = 0.02, 0.9            # moderate jumps (mean size 1.8), as in the full run
EPS_BIG, THETA_BIG = 0.01, 3.0     # large jumps (mean size 6), as in the HMM-9 design
XN, WN = roots_genlaguerre(32, 1); WN = WN / np.exp(gammaln(2.0))   # Z ~ Gamma(2,1)

# ---------------- data ----------------
def gen(n, burn, pars, seed):
    """Same draws as giorno_models.simulate_hmm6; also returns the state path."""
    rng = np.random.default_rng(int(seed))
    p00, p11, nu0, nu1, s0, s1 = [float(pars[k]) for k in ("p00", "p11", "nu0", "nu1", "s0", "s1")]
    total = int(n + burn); st = np.empty(total, dtype=np.int8); u = rng.random(total)
    st[0] = 0 if u[0] < (1 - p11) / (2 - p00 - p11) else 1
    for t in range(1, total):
        st[t] = 0 if (u[t] < (p00 if st[t - 1] == 0 else 1 - p11)) else 1
    z = rng.standard_t(np.where(st == 0, nu0, nu1)) * np.where(st == 0, s0, s1)
    return z[burn:], st[burn:]

def e_abs_t(nu):
    return 2 * np.sqrt(nu) * np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / (np.sqrt(np.pi) * (nu - 1))

def skew_arm(x, st, pars, g, uni):
    """Fernandez-Steel skewing with parameter g (<1: negative skew), regime-centred."""
    pos = uni < g * g / (1 + g * g); a = np.abs(x)
    y = np.where(pos, g * a, -a / g)
    nu = np.where(st == 0, pars["nu0"], pars["nu1"]); s = np.where(st == 0, pars["s0"], pars["s1"])
    return y - s * e_abs_t(nu) * (g - 1 / g)

# ---------------- competitor tests ----------------
def ks_symmetry(z, rng, B=199):
    z = z - np.median(z)
    grid = np.quantile(np.abs(z), np.linspace(0.02, 0.98, 49))
    def stat(v):
        return np.max(np.abs((v[:, None] <= -grid).mean(0) - (v[:, None] >= grid).mean(0))) * np.sqrt(len(v))
    s0 = stat(z); flips = np.array([stat(z * rng.choice([-1.0, 1.0], len(z))) for _ in range(B)])
    return float((1 + (flips >= s0).sum()) / (B + 1))

def nw_var(v, L=None):
    v = v - v.mean(); T = len(v); L = int(np.floor(4 * (T / 100) ** (2 / 9))) if L is None else L
    s = v @ v / T
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (v[l:] @ v[:-l]) / T
    return s

def skew_test(z):
    z = (z - z.mean()) / z.std(); w = z ** 3 - 3 * z
    stat = np.sqrt(len(z)) * w.mean() / np.sqrt(nw_var(w))
    from scipy.stats import norm
    return float(norm.cdf(stat)), float(stat)       # one-sided p for negative skewness

# ---------------- follow-up: skew-t versus jumps (two-step, on GARCH-standardised residuals) ----------------
# The GARCH(1,1)-t volatility filter is held fixed at its estimate; on the standardised residuals u_t two
# innovation laws are fitted: Fernandez-Steel skew-t (location m, scale c, nu, gamma) and Student-t plus
# Bernoulli-Gamma(2) downward jumps (m, c, nu, eps, theta). The Jacobian of the standardisation is common to
# both and cancels in the Vuong statistic.
def t_logpdf(x, nu):
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(x * x / nu)

def ll_skew(p, u):
    m, lc, lnu, lg = p; c = np.exp(lc); nu = 2.05 + np.exp(lnu); g = np.exp(lg)
    e = (u - m) / c
    return np.log(2.0 / (g + 1.0 / g)) + np.where(e >= 0, t_logpdf(e / g, nu), t_logpdf(e * g, nu)) - lc

def ll_jump(p, u):
    m, lc, lnu, le, lth = p; c = np.exp(lc); nu = 2.05 + np.exp(lnu); eps = 0.2 / (1 + np.exp(-le)); th = np.exp(lth)
    e = (u - m) / c
    f = np.exp(t_logpdf(e, nu)); hj = sum(w * np.exp(t_logpdf(e + th * q / c, nu)) for q, w in zip(XN, WN))
    return np.log(np.maximum((1 - eps) * f + eps * hj, 1e-300)) - lc

def fit_ll(fun, starts, u):
    best = None
    for p0 in starts:
        r = minimize(lambda p: -np.sum(fun(p, u)), p0, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun: best = r
    return best, fun(best.x, u)

def vuong(u, nu0):
    ln0 = np.log(max(nu0 - 2.05, 0.05))
    rs, ls = fit_ll(ll_skew, [[0.0, 0.0, ln0, 0.0], [0.0, 0.0, ln0, np.log(0.9)]], u)
    rj, lj = fit_ll(ll_jump, [[0.0, 0.0, ln0, -1.5, np.log(1.0)], [0.0, 0.0, ln0, -3.0, np.log(2.0)]], u)
    d = ls - lj; T = len(u)
    sd = np.sqrt(T * nw_var(d))
    V = (d.sum() + 0.5 * np.log(T)) / sd                     # Schwarz correction: 4 vs 5 parameters
    return float(V), float(d.sum()), float(sd), float(np.exp(rs.x[3])), float(0.2 / (1 + np.exp(-rj.x[3]))), float(np.exp(rj.x[4]))

# ---------------- one path ----------------
def one(e, rng):
    pars = OPTS[e.scenario]; n = int(e.n); ntr = int(e.n_train)
    x, st = gen(n, 600, pars, int(e.seed))
    rj = np.random.default_rng([int(e.seed), 20261031]); ub = rj.random(n); B = ub < EPS; Z = rj.gamma(2.0, 1.0, n)
    Bbig = ub < EPS_BIG                                                     # a subset of the same jump dates
    uni = np.random.default_rng([int(e.seed), 20261103]).random(n)
    arms = {"sym": x, "jump": x - B * THETA * Z, "jumpbig": x - Bbig * THETA_BIG * Z, "skew90": skew_arm(x, st, pars, 0.90, uni), "skew95": skew_arm(x, st, pars, 0.95, uni)}
    out = []
    for arm, y in arms.items():
        tr = y[:ntr]
        loc, s, nu = H.fit_t(tr); rt = H.srj(tr, loc, s, nu, rng, True)
        gm, gs, gnu, _ = H.fit_garch(tr); rg = H.srj(tr, gm, gs, gnu, rng, True); ng = H.srj(tr, gm, gs, gnu, rng, False)
        ures = (tr - gm) / gs                                       # t-scale GARCH residuals
        p_ks = ks_symmetry(ures, rng); p_sk, sk = skew_test(ures)
        V, LR, sdLR, g_hat, eps_hat, th_hat = vuong(ures, gnu)
        out.append(dict(id=e.id, scenario=e.scenario, rep=int(e.rep), arm=arm,
                        p_rob_t=rt[3], p_rob_g=rg[3], p_naive_g=ng[1], p_ks_g=p_ks, p_skew_g=p_sk, skew_stat=sk,
                        vuong=V, lr_skew_minus_jump=LR, sd_lr=sdLR, gamma_hat=g_hat, eps_hat=eps_hat, theta_hat=th_hat, garch_nu=gnu))
    return out

if __name__ == "__main__":
    w, nw, outf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    man = pd.read_csv(os.path.join(REPO, "DGP_MANIFEST.csv")).sort_values(["rep", "scenario"]).reset_index(drop=True)
    man = man.iloc[w::nw]
    done = set(pd.read_csv(outf).id) if os.path.exists(outf) else set()
    for e in man.itertuples():
        if e.id in done: continue
        rows = one(e, np.random.default_rng([int(e.seed), 11]))
        pd.DataFrame(rows).to_csv(outf, mode="a", header=not os.path.exists(outf), index=False)
        print(e.id, " ".join(f"{r['arm']}:{r['p_rob_g']:.3f}" for r in rows), flush=True)
