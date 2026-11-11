"""Paper B, roadmap item B3: full value-at-risk arm on the 4,000-path HMM-6 manifest.

For each manifest path (regenerated exactly; estimation on the first 4,800 observations, evaluation on the last
1,200) three arms share the same regime path and jump stream:
  nojump : the HMM path
  jump01 : downward jumps with eps = 0.01, size 0.9*Gamma(2,1)   (a subset of the jump02 dates)
  jump02 : downward jumps with eps = 0.02, size 0.9*Gamma(2,1)   (identical to the full jump arm)
Four models are fitted on the estimation sample without a jump component and filtered through the test sample
with fixed parameters: static Student-t, GARCH(1,1)-t, APARCH(1,1)-t, HMM-6 (repository fit_hmm6).
The true conditional law is the HMM with its own jump component; its predictive state probabilities come from
the exact forward filter with jump-contaminated emissions (1 - eps) f_j + eps h_j.

Outputs, per path x arm x model x level (alpha = 0.05, 0.01, 0.005): breach rate; Kupiec p-value; relative VaR
shortfall (true VaR - model VaR)/true VaR; expected pinball regret given the true conditional law,
int_{q*}^{q_m} (F_t(x) - alpha) dx (Corollary 3); realised pinball difference. Per path x arm x model at 99%:
Basel exceptions in the last 250 days, zone, plus factor, capital charge relative to the true-VaR capital charge;
relative ES shortfall at 97.5%.

Usage: python -I b3_var.py WORKER NWORKERS OUT.csv      (resumable; HMM6_REPO = folder with DGP_MANIFEST.csv
and giorno_models.py; numba recommended - without it HMM-6 fits are about 10 times slower)
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln, roots_genlaguerre, roots_legendre
from scipy.stats import t as tdist, chi2
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.environ.get("HMM6_REPO", HERE); sys.path.insert(1, REPO)
import hmm9_srj as H
try:
    import giorno_models as GM
except ModuleNotFoundError:            # numba missing: use a pure-Python stand-in for njit
    import types; sys.modules["numba"] = types.SimpleNamespace(njit=lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f)))
    import giorno_models as GM

OPTS = json.load(open(os.path.join(HERE, "scenarios.json")))
THETA = 0.9
LEVELS = (0.05, 0.01, 0.005)
XN, WN = roots_genlaguerre(32, 1); WN = WN / np.exp(gammaln(2.0))     # Z ~ Gamma(2,1)
GL_X, GL_W = roots_legendre(16)

# ---------------- generator (identical draws to simulate_hmm6; also returns the state path) ----------------
def gen(n, burn, pars, seed):
    rng = np.random.default_rng(int(seed))
    p00, p11, nu0, nu1, s0, s1 = [float(pars[k]) for k in ("p00", "p11", "nu0", "nu1", "s0", "s1")]
    total = int(n + burn); st = np.empty(total, dtype=np.int8); u = rng.random(total)
    st[0] = 0 if u[0] < (1 - p11) / (2 - p00 - p11) else 1
    for t in range(1, total):
        st[t] = 0 if (u[t] < (p00 if st[t - 1] == 0 else 1 - p11)) else 1
    z = rng.standard_t(np.where(st == 0, nu0, nu1)) * np.where(st == 0, s0, s1)
    return z[burn:]

# ---------------- distribution helpers ----------------
def t_cdf(x, nu, s): return tdist.cdf(x / s, nu)
def t_pe(q, nu, s):
    """E[X 1{X <= q}] for X = s * T_nu (nu > 1)."""
    z = q / s; return -s * (nu + z * z) / (nu - 1) * tdist.pdf(z, nu)

def state_cdf(q, nu, s, eps):
    F = t_cdf(q, nu, s)
    if eps == 0: return F
    return (1 - eps) * F + eps * sum(w * t_cdf(q + THETA * x, nu, s) for x, w in zip(XN, WN))

def state_pe(q, nu, s, eps):
    pe = t_pe(q, nu, s)
    if eps == 0: return pe
    jb = sum(w * (t_pe(q + THETA * x, nu, s) - THETA * x * t_cdf(q + THETA * x, nu, s)) for x, w in zip(XN, WN))
    return (1 - eps) * pe + eps * jb

def mix_cdf(q, w0, P, eps):
    return w0 * state_cdf(q, P["nu0"], P["s0"], eps) + (1 - w0) * state_cdf(q, P["nu1"], P["s1"], eps)

def mix_pe(q, w0, P, eps):
    return w0 * state_pe(q, P["nu0"], P["s0"], eps) + (1 - w0) * state_pe(q, P["nu1"], P["s1"], eps)

def bisect(cdf, alpha, lo, hi, it=60):
    lo = np.full_like(hi, lo) if np.isscalar(lo) else lo.copy(); hi = hi.copy()
    for _ in range(it):
        mid = 0.5 * (lo + hi); below = cdf(mid) < alpha
        lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)

# ---------------- true conditional law ----------------
def true_law(y, P, eps, ntr):
    """Predictive probability of state 0 over the test sample, from the exact forward filter with
    jump-contaminated emissions, run over the whole path."""
    def loge(nu, s):
        f = tdist.pdf(y / s, nu) / s
        if eps > 0:
            f = (1 - eps) * f + eps * sum(w * tdist.pdf((y + THETA * x) / s, nu) / s for x, w in zip(XN, WN))
        return np.log(np.maximum(f, 1e-300))
    pi0 = (1 - P["p11"]) / (2 - P["p00"] - P["p11"])
    _, pred, _ = GM._forward(loge(P["nu0"], P["s0"]), loge(P["nu1"], P["s1"]), P["p00"], P["p11"], pi0)
    return pred[ntr:]

# ---------------- models ----------------
def garch_filter(y, mu, om, al, be, nu):
    e2 = (y - mu) ** 2
    h = lfilter([1.0], [1.0, -be], np.concatenate([[y[:4800].var()], om + al * e2[:-1]]))
    return np.sqrt(h * (nu - 2) / nu)            # t-scale

def fit_aparch(y):
    """APARCH(1,1)-t: sigma_t^d = om + al (|e| - ga e)^d + be sigma_{t-1}^d, e = y - mu."""
    def unpack(u):
        a, b = np.exp(u[2]), np.exp(u[4]); s = 1 + a + b
        return u[0], np.exp(u[1]), a / s, np.tanh(u[3]), b / s, 0.5 + 2.5 / (1 + np.exp(-u[5])), 2.05 + np.exp(u[6])
    def sig(p, yy):
        mu, om, al, ga, be, d, nu = p; e = yy - mu
        x = al * (np.abs(e) - ga * e) ** d
        s = lfilter([1.0], [1.0, -be], np.concatenate([[yy[:4800].std() ** d], om + x[:-1]]))
        return np.maximum(s, 1e-12) ** (1 / d)
    def nll(u):
        p = unpack(u); sd = sig(p, y); nu = p[6]
        if not np.all(np.isfinite(sd)): return 1e10
        sc = sd * np.sqrt((nu - 2) / nu)
        return -(tdist.logpdf((y - p[0]) / sc, nu) - np.log(sc)).sum()
    v = y.std(); best = None
    for d0, g0 in ((2.0, 0.0), (1.2, 0.2)):
        r0 = 0.05
        u0 = [np.median(y), np.log(r0 * v ** d0), np.log(0.05 / 0.05), np.arctanh(g0), np.log(0.9 / 0.05),
              np.log((d0 - 0.5) / (3.0 - d0)), np.log(3.0)]
        r = minimize(nll, u0, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun: best = r
    p = unpack(best.x)
    return p, (lambda yy: sig(p, yy) * np.sqrt((p[6] - 2) / p[6]))   # t-scale filter for any series

# ---------------- evaluation ----------------
def kupiec(x, n, a):
    if x == 0: lr = -2 * n * np.log(1 - a)
    elif x == n: lr = -2 * n * np.log(a)
    else:
        ph = x / n; lr = -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return float(chi2.sf(lr, 1))

PLUS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
def capital(var99, breaches):
    k = int(breaches[-250:].sum()); pf = 0.0 if k <= 4 else PLUS.get(k, 1.0)
    return max(var99[-1], (3 + pf) * var99[-60:].mean()), k, ("green" if k <= 4 else "yellow" if k <= 9 else "red")

def expected_regret(qm, qs, w0, P, eps, a):
    """int_{qs}^{qm} (F_t(x) - alpha) dx under the true conditional law (Gauss-Legendre, 16 nodes)."""
    mid, half = 0.5 * (qm + qs), 0.5 * (qm - qs); tot = 0.0
    for x, w in zip(GL_X, GL_W):
        tot = tot + w * (mix_cdf(mid + half * x, w0, P, eps) - a)
    return half * tot

def pinball(y, q, a): return (a - (y < q)) * (y - q)

def one(e):
    P = OPTS[e.scenario]; n = int(e.n); ntr = int(e.n_train)
    x = gen(n, 600, P, int(e.seed))
    rj = np.random.default_rng([int(e.seed), 20261031]); ub = rj.random(n); Z = rj.gamma(2.0, 1.0, n)
    arms = {"nojump": (x, 0.0), "jump01": (x - (ub < 0.01) * THETA * Z, 0.01), "jump02": (x - (ub < 0.02) * THETA * Z, 0.02)}
    rows = []
    for arm, (y, eps) in arms.items():
        tr, te = y[:ntr], y[ntr:]
        w0 = true_law(y, P, eps, ntr)
        smax = max(P["s0"], P["s1"]) * 60
        qtrue = {a: bisect(lambda q: mix_cdf(q, w0, P, eps), a, -smax, np.zeros(len(te))) for a in LEVELS + (0.025,)}
        # fitted models -> dict name -> (loc, scale array, nu) or ("hmm", params)
        loc, s, nu = H.fit_t(tr)
        gm, _, gnu, (om, al, be) = H.fit_garch(tr); gs = garch_filter(y, gm, om, al, be, gnu)[ntr:]
        ap, apf = fit_aparch(tr); asc = apf(y)[ntr:]
        fh = GM.fit_hmm6(tr, starts=4, maxiter=260, seed=(int(e.seed) + 1) % (2 ** 32))
        hp = fh["params"]; _, hpred, _ = GM.hmm_test_predict(tr, te, [hp[k] for k in ("p00", "p11", "nu0", "nu1", "s0", "s1")])
        models = {"Student-t": ("t", loc, np.full(len(te), s), nu), "GARCH-t": ("t", gm, gs, gnu),
                  "APARCH-t": ("t", ap[0], asc, ap[6]), "HMM-6": ("hmm", hp, hpred)}
        for name, M in models.items():
            def q_of(a):
                if M[0] == "t": return M[1] + M[2] * tdist.ppf(a, M[3])
                return bisect(lambda q: mix_cdf(q, M[2], M[1], 0.0), a, -smax, np.zeros(len(te)))
            def pe_of(q):
                if M[0] == "t": return M[1] * tdist.cdf((q - M[1]) / M[2], M[3]) + t_pe(q - M[1], M[3], M[2])
                return mix_pe(q, M[2], M[1], 0.0)
            q975 = q_of(0.025); es_m = -pe_of(q975) / 0.025; es_t = -mix_pe(qtrue[0.025], w0, P, eps) / 0.025
            q99 = None
            for a in LEVELS:
                qm, qs = q_of(a), qtrue[a]; br = te < qm
                if a == 0.01: q99, br99 = qm, br
                rows.append(dict(id=e.id, scenario=e.scenario, rep=int(e.rep), arm=arm, model=name, level=a,
                                 breach_rate=float(br.mean()), kupiec_p=kupiec(int(br.sum()), len(te), a),
                                 rel_shortfall=float(np.mean((qs - qm) / qs)),      # (true VaR - model VaR)/true VaR, VaR = -q
                                 exp_regret=float(np.mean(expected_regret(qm, qs, w0, P, eps, a))),
                                 pin_diff=float(np.mean(pinball(te, qm, a) - pinball(te, qs, a))),
                                 hmm_success=bool(fh["success"]) if name == "HMM-6" else None))
            cc_m, k_m, zone_m = capital(-q99, br99); cc_t, k_t, _ = capital(-qtrue[0.01], te < qtrue[0.01])
            for r in rows[-3:]:
                r.update(basel_exceptions=k_m, basel_zone=zone_m, capital_ratio=float(cc_m / cc_t),
                         es975_rel_shortfall=float(np.mean((es_t - es_m) / es_t)))
    return rows

if __name__ == "__main__":
    w, nw, outf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    man = pd.read_csv(os.path.join(REPO, "DGP_MANIFEST.csv")).sort_values(["rep", "scenario"]).reset_index(drop=True).iloc[w::nw]
    done = set(pd.read_csv(outf).id) if os.path.exists(outf) else set()
    for e in man.itertuples():
        if e.id in done: continue
        rows = one(e)
        pd.DataFrame(rows).to_csv(outf, mode="a", header=not os.path.exists(outf), index=False)
        r99 = [r for r in rows if r["level"] == 0.01 and r["model"] == "GARCH-t"]
        print(e.id, " ".join(f"{r['arm']}:{r['breach_rate']*100:.2f}%" for r in r99), flush=True)
