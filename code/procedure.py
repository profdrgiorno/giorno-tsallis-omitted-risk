"""Paper B, items C1-C2: the practical procedure of Section 5.1 applied to a daily return series.

Step 1 (Paper A): i.i.d. two-component Student-t mixture -> contrast d, kappa = pi0 pi1 E[d^2], T kappa^2,
         and Paper A's persistence test Z = sqrt(T-1) corr(d_t, d_{t+1}). (kappa-hat is biased upward when there is
         only one regime, so a large T kappa^2 needs a significant Z.)
Step 2 : robust jump test (orthogonalised), nulls GARCH(1,1)-t and GJR-GARCH(1,1)-t; both directions.
Step 3 : pre-check of the jump information: theta~ = argmax Z(theta); eps~ = mean(xi)/lambda-hat(theta~), with
         lambda-hat the single-impulse response of the test function; n_J = T * I_theta * theta^2 (Proposition 3),
         averaged over the volatility path in standardised units.
Step 4 : GARCH(1,1)-t with Bernoulli-Gamma downward jumps (-B_t G_t, G = theta Z, Z ~ Gamma(k,1)), joint MLE;
         standard errors from the numerical Hessian; n_J confirmed at the estimates.
Step 5 : out-of-sample VaR (last 20%): GARCH-t versus GARCH-t + jumps, breach rates, Kupiec, pinball-loss
         difference with a Newey-West t-statistic, 97.5% expected shortfall.
Usage: python -I procedure.py SERIES.csv DATECOL RETCOL LABEL OUT.json     (returns in percent)
"""
import sys, json, os
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln, roots_genlaguerre, digamma
from scipy.stats import t as tdist, norm, chi2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hmm9_srj as H, realdata as R

# ---------------- step 1: Paper A contrast ----------------
def t_pdf(y, nu, s, m=0.0): return tdist.pdf((y - m) / s, nu) / s

def fit_mixture(y):
    """i.i.d. two-component Student-t mixture with a common location."""
    def unpack(u):
        return 1 / (1 + np.exp(-u[0])), u[1], 2.05 + np.exp(u[2]), 2.05 + np.exp(u[3]), np.exp(u[4]), np.exp(u[5])
    def nll(u):
        p0, m, n0, n1, s0, s1 = unpack(u)
        return -np.log(p0 * t_pdf(y, n0, s0, m) + (1 - p0) * t_pdf(y, n1, s1, m) + 1e-300).sum()
    sd = y.std(); best = None
    for a, b in ((0.6, 1.8), (0.8, 2.5), (0.5, 1.3)):
        u0 = [0.5, np.median(y), np.log(8), np.log(4), np.log(sd * a), np.log(sd * b)]
        r = minimize(nll, u0, method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-7, fatol=1e-9))
        if best is None or r.fun < best.fun: best = r
    p0, m, n0, n1, s0, s1 = unpack(best.x)
    if s0 > s1: p0, n0, n1, s0, s1 = 1 - p0, n1, n0, s1, s0                # state 0 = calm
    return dict(pi0=p0, mu=m, nu0=n0, nu1=n1, s0=s0, s1=s1)

def paperA(y, mix):
    f0, f1 = t_pdf(y, mix["nu0"], mix["s0"], mix["mu"]), t_pdf(y, mix["nu1"], mix["s1"], mix["mu"])
    fb = mix["pi0"] * f0 + (1 - mix["pi0"]) * f1; d = (f0 - f1) / fb
    kappa = mix["pi0"] * (1 - mix["pi0"]) * np.mean(d ** 2)
    Z = np.sqrt(len(y) - 1) * np.corrcoef(d[:-1], d[1:])[0, 1]
    return dict(kappa=float(kappa), T_kappa2=float(len(y) * kappa ** 2), Z_persistence=float(Z), p_persistence=float(norm.sf(Z)))

# ---------------- step 3: pre-check ----------------
XN2, WN2 = roots_genlaguerre(32, 1); WN2 = WN2 / np.exp(gammaln(2.0))

def xi_series(y, loc, sc, nu, th):
    """Orthogonalised test function (Jacobian kappa), as in realdata.test, for one theta."""
    smu = H._smu(y, loc, sc, nu); dl = 1e-4 * np.median(np.atleast_1d(sc))
    dsmu = -(H._smu(y + dl, loc, sc, nu) - H._smu(y - dl, loc, sc, nu)) / (2 * dl)
    a = H._afun(y, loc, sc, nu, th, True)
    da = -(H._afun(y + dl, loc, sc, nu, th, True) - H._afun(y - dl, loc, sc, nu, th, True)) / (2 * dl)
    k = da.mean() / dsmu.mean()
    return lambda yy: H._afun(yy, loc, sc, nu, th, True) - k * H._smu(yy, loc, sc, nu)

def info_theta(eps, c, nu, k=2.0, lo=-200, hi=100, n=60001):
    """Proposition 3: Fisher information about the standardised jump scale c (base t_nu, unit scale)."""
    x, w = roots_genlaguerre(48, k - 1); w = w / np.exp(gammaln(k))
    y = np.linspace(lo, hi, n); dy = y[1] - y[0]
    f = tdist.pdf(y, nu); h = sum(wi * tdist.pdf(y + c * xi, nu) for xi, wi in zip(x, w))
    dfun = lambda v: -(nu + 1) * v / (nu + v * v) * tdist.pdf(v, nu)
    dh = sum(wi * xi * dfun(y + c * xi) for xi, wi in zip(x, w))
    return float(eps ** 2 * np.sum(dh * dh / ((1 - eps) * f + eps * h)) * dy)

def n_jump(eps, theta, sig, nu, k=2.0, bins=12):
    """n_J = theta^2 * sum_t I_c(eps, theta/sigma_t) / sigma_t^2 with sigma_t grouped in quantile bins."""
    edges = np.quantile(sig, np.linspace(0, 1, bins + 1)); tot = 0.0
    for i in range(bins):
        m = (sig >= edges[i]) & (sig <= edges[i + 1]) if i == bins - 1 else (sig >= edges[i]) & (sig < edges[i + 1])
        if m.sum() == 0: continue
        sb = np.median(sig[m]); tot += m.sum() * info_theta(eps, theta / sb, nu, k) / sb ** 2
    return float(theta ** 2 * tot)

# ---------------- step 4: GARCH-t + Bernoulli-Gamma jumps ----------------
def jump_garch_ll(u, y, per_obs=False):
    mu, om = u[0], np.exp(u[1]); a, b = np.exp(u[2]), np.exp(u[3]); s = 1 + a + b; al, be = a / s, b / s
    nu = 2.05 + np.exp(u[4]); eps = 0.10 / (1 + np.exp(-u[5])); th = np.exp(u[6]); k = 0.3 + np.exp(u[7])
    e2 = (y - mu) ** 2
    h = lfilter([1.0], [1.0, -be], np.concatenate([[y.var()], om + al * e2[:-1]]))
    if not np.all(np.isfinite(h)) or np.any(h <= 0): return -1e10 if not per_obs else None
    sc = np.sqrt(h * (nu - 2) / nu)
    x, w = roots_genlaguerre(32, k - 1); w = w / np.exp(gammaln(k))
    f = tdist.pdf((y - mu) / sc, nu) / sc
    hj = sum(wi * tdist.pdf((y - mu + th * xi) / sc, nu) / sc for xi, wi in zip(x, w))
    l = np.log(np.maximum((1 - eps) * f + eps * hj, 1e-300))
    return l if per_obs else l.sum()

def natural(u):
    a, b = np.exp(u[2]), np.exp(u[3]); s = 1 + a + b
    return dict(mu=u[0], omega=np.exp(u[1]), alpha=a / s, beta=b / s, nu=2.05 + np.exp(u[4]),
                eps=0.10 / (1 + np.exp(-u[5])), theta=np.exp(u[6]), k=0.3 + np.exp(u[7]))

def fit_jump_garch(y, g_start, th0):
    mu, om, al, be, nu = g_start; r = 1 - al - be
    base = [mu, np.log(om), np.log(al / r), np.log(be / r), np.log(nu - 2.05)]
    best = None
    for e0, t0, k0 in ((0.01, th0, 2.0), (0.03, th0 / 2, 2.0), (0.005, 2 * th0, 1.0)):
        u0 = base + [np.log(e0 / (0.10 - e0)), np.log(t0), np.log(k0 - 0.3)]
        rr = minimize(lambda u: -jump_garch_ll(u, y), u0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-8))
        if best is None or rr.fun < best.fun: best = rr
    return best

def se_natural(u, y, keys=("eps", "theta", "k")):
    """Delta-method standard errors of selected natural parameters from the numerical Hessian (outer product
    as fallback if the Hessian is not negative definite)."""
    f = lambda v: jump_garch_ll(v, y); n = len(u); hstep = 1e-4 * np.maximum(1, np.abs(u))
    Hm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ej = np.zeros(n); ei[i] = hstep[i]; ej[j] = hstep[j]
            Hm[i, j] = Hm[j, i] = (f(u + ei + ej) - f(u + ei - ej) - f(u - ei + ej) + f(u - ei - ej)) / (4 * hstep[i] * hstep[j])
    ev = np.linalg.eigvalsh(-Hm); ok = bool(np.all(ev > 1e-8 * max(abs(ev).max(), 1)))
    try:
        V = np.linalg.inv(-Hm) if ok else np.linalg.pinv(-Hm)
    except np.linalg.LinAlgError:
        ok = False; V = np.linalg.pinv(-Hm)
    out = {}
    for kname in keys:
        g = np.array([(natural(u + np.eye(n)[i] * hstep[i])[kname] - natural(u - np.eye(n)[i] * hstep[i])[kname]) / (2 * hstep[i]) for i in range(n)])
        out[kname] = float(np.sqrt(max(g @ V @ g, 0)))
    return out, ok

# ---------------- step 4b: GARCH(1,1) with Fernandez-Steel skew-t innovations ----------------
def fs_logpdf(e, nu, g):
    return np.log(2.0 / (g + 1.0 / g)) + np.where(e >= 0, tdist.logpdf(e / g, nu), tdist.logpdf(e * g, nu))

def skew_garch_ll(u, y, per_obs=False):
    mu, om = u[0], np.exp(u[1]); a, b = np.exp(u[2]), np.exp(u[3]); s = 1 + a + b; al, be = a / s, b / s
    nu = 2.05 + np.exp(u[4]); g = np.exp(u[5])
    h = lfilter([1.0], [1.0, -be], np.concatenate([[y.var()], om + al * ((y - mu) ** 2)[:-1]]))
    if not np.all(np.isfinite(h)) or np.any(h <= 0): return -1e10 if not per_obs else None
    sc = np.sqrt(h * (nu - 2) / nu); l = fs_logpdf((y - mu) / sc, nu, g) - np.log(sc)
    return l if per_obs else l.sum()

def fit_skew_garch(y, g_start):
    mu, om, al, be, nu = g_start; r = 1 - al - be
    base = [mu, np.log(om), np.log(al / r), np.log(be / r), np.log(nu - 2.05)]; best = None
    for g0 in (0.9, 1.0):
        rr = minimize(lambda u: -skew_garch_ll(u, y), base + [np.log(g0)], method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-8))
        if best is None or rr.fun < best.fun: best = rr
    return best

def skew_quantile(a, mu, sc, nu, g):
    """Lower a-quantile of mu + sc * FS(nu, g)."""
    p_neg = 1.0 / (1.0 + g * g)                       # P(e < 0)
    if a <= p_neg: return mu + sc * tdist.ppf(a * (1 + g * g) / 2.0, nu) / g
    return mu + sc * g * tdist.ppf((a - p_neg) * (1 + g * g) / (2 * g * g) + 0.5, nu)

def vuong(l1, l2, k1, k2):
    d = l1 - l2; T = len(d); v = d - d.mean(); L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    s = v @ v / T + sum(2 * (1 - l / (L + 1)) * (v[l:] @ v[:-l]) / T for l in range(1, L + 1))
    return float((d.sum() - 0.5 * (k1 - k2) * np.log(T)) / np.sqrt(T * s))

# ---------------- step 5: VaR ----------------
def kupiec(x, n, a):
    if x == 0: lr = -2 * n * np.log(1 - a)
    else:
        ph = x / n; lr = -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return float(chi2.sf(lr, 1))

def nw_t(dv, L=None):
    v = dv - dv.mean(); T = len(v); L = int(np.floor(4 * (T / 100) ** (2 / 9))) if L is None else L
    s = v @ v / T + sum(2 * (1 - l / (L + 1)) * (v[l:] @ v[:-l]) / T for l in range(1, L + 1))
    return float(dv.mean() / np.sqrt(s / T))

def var_jump(qlevel, mu, sc, nu, p):
    """Lower quantile of (1-eps) t + eps (t - G) per t, by bisection on the CDF."""
    x, w = roots_genlaguerre(32, p["k"] - 1); w = w / np.exp(gammaln(p["k"]))
    cdf = lambda q: (1 - p["eps"]) * tdist.cdf((q - mu) / sc, nu) + p["eps"] * sum(wi * tdist.cdf((q - mu + p["theta"] * xi) / sc, nu) for xi, wi in zip(x, w))
    lo = mu - 80 * sc; hi = mu + 0 * sc
    for _ in range(60):
        mid = 0.5 * (lo + hi); below = cdf(mid) < qlevel; lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)

def run(y, label):
    rng = np.random.default_rng(20261104); T = len(y); out = dict(series=label, T=T)
    # step 1
    gm, gs, gnu, (om, al, be) = H.fit_garch(y)
    mix = fit_mixture(y); out["step1_raw"] = dict(mixture=mix, **paperA(y, mix))
    z = (y - gm) / gs; mixz = fit_mixture(z); out["step1_garch_std"] = dict(mixture=mixz, **paperA(z, mixz))
    # step 2
    loc, s, nu = H.fit_t(y); grid = [g * s for g in R.G_REL]
    rg = R.test(y, gm, gs, gnu, grid, rng)
    jm, js, jnu, jp = R.fit_gjr(y); rj = R.test(y, jm, js, jnu, grid, rng)
    out["step2"] = dict(grid=[float(g) for g in grid], garch_Z=rg["Z"], garch_p_down=rg["p_down"], garch_p_up=rg["p_up"],
                        gjr_Z=rj["Z"], gjr_p_down=rj["p_down"], gjr_p_up=rj["p_up"])
    # step 3 (pre-check, GARCH-t null)
    i = int(np.argmax(rg["Z"])); th = grid[i]; xi = xi_series(y, gm, gs, gnu, th)
    x, w = roots_genlaguerre(32, 1); w = w / np.exp(gammaln(2.0))
    lam = float(np.mean(sum(wi * xi(y - th * xx) for xx, wi in zip(x, w)) - xi(y)))
    eps_t = float(max(np.mean(xi(y)) / lam, 1e-6)) if lam > 0 else float("nan")
    sig_t = gs                                  # t-scale of the GARCH-t fit
    nJ_pre = n_jump(eps_t, th, sig_t, gnu) if np.isfinite(eps_t) else float("nan")
    Zp = np.array(rg["Z"]); profile = "decreasing (asymmetry in the body: skewness-like)" if Zp[0] == Zp.max() and Zp[-1] < Zp[0] else ("increasing (asymmetry in the tail: jump-like)" if Zp[-1] == Zp.max() else "interior maximum")
    flag = (not np.isfinite(eps_t)) or eps_t >= 0.10 or i == 0
    out["step3"] = dict(theta_tilde=float(th), lambda_hat=lam, eps_tilde=eps_t, n_J=nJ_pre if not flag else float("nan"),
                        rel_error=float(1 / np.sqrt(nJ_pre)) if (nJ_pre > 0 and not flag) else float("nan"),
                        Z_profile=profile, precheck_flag=("pre-check not interpretable as jumps: " + ("eps~ >= 0.10" if np.isfinite(eps_t) and eps_t >= 0.10 else "") + (" theta~ at the smallest grid value" if i == 0 else "")) if flag else "ok")
    # step 4
    fj = fit_jump_garch(y, (gm, om, al, be, gnu), th); pj = natural(fj.x)
    ses, hess_ok = se_natural(fj.x, y)
    llg = -(-(tdist.logpdf((y - gm) / gs, gnu) - np.log(gs)).sum())
    h_j = lfilter([1.0], [1.0, -pj["beta"]], np.concatenate([[y.var()], pj["omega"] + pj["alpha"] * (y - pj["mu"])[:-1] ** 2]))
    sig_j = np.sqrt(h_j * (pj["nu"] - 2) / pj["nu"])
    nJ_fit = n_jump(pj["eps"], pj["theta"], sig_j, pj["nu"], k=pj["k"])
    fs = fit_skew_garch(y, (gm, om, al, be, gnu)); us = fs.x
    l_skew = skew_garch_ll(us, y, per_obs=True); l_jump = jump_garch_ll(fj.x, y, per_obs=True)
    out["step4_skew"] = dict(gamma=float(np.exp(us[5])), nu=float(2.05 + np.exp(us[4])), loglik=float(-fs.fun),
                             LR_vs_garch_t=float(2 * (-fs.fun - llg)),
                             vuong_skew_vs_jump=vuong(l_skew, l_jump, 6, 8),
                             aic_skew=float(2 * 6 + 2 * fs.fun), aic_jump=float(2 * 8 + 2 * fj.fun), aic_garch_t=float(2 * 5 - 2 * llg))
    out["step4"] = dict(params={k: float(v) for k, v in pj.items()}, se=ses, hessian_ok=hess_ok, eps_at_bound=bool(pj["eps"] > 0.099),
                        loglik_jump=float(-fj.fun), loglik_garch_t=float(llg), LR=float(2 * (-fj.fun - llg)),
                        n_J=nJ_fit, rel_error_theta=float(1 / np.sqrt(nJ_fit)) if nJ_fit > 0 else float("nan"),
                        expected_jumps=float(T * pj["eps"]))
    # step 5: out-of-sample VaR, parameters estimated on the first 80%
    ntr = int(0.8 * T); tr = y[:ntr]
    gm2, _, gnu2, (om2, al2, be2) = H.fit_garch(tr)
    hg = lfilter([1.0], [1.0, -be2], np.concatenate([[tr.var()], om2 + al2 * (y - gm2)[:-1] ** 2]))
    sg = np.sqrt(hg * (gnu2 - 2) / gnu2)[ntr:]
    fj2 = fit_jump_garch(tr, (gm2, om2, al2, be2, gnu2), th); p2 = natural(fj2.x)
    hj2 = lfilter([1.0], [1.0, -p2["beta"]], np.concatenate([[tr.var()], p2["omega"] + p2["alpha"] * (y - p2["mu"])[:-1] ** 2]))
    sj = np.sqrt(hj2 * (p2["nu"] - 2) / p2["nu"])[ntr:]; te = y[ntr:]; res = {}
    fs2 = fit_skew_garch(tr, (gm2, om2, al2, be2, gnu2)); u2 = fs2.x
    mu_s = u2[0]; a_, b_ = np.exp(u2[2]), np.exp(u2[3]); ss_ = 1 + a_ + b_; nu_s = 2.05 + np.exp(u2[4]); g_s = np.exp(u2[5])
    hs = lfilter([1.0], [1.0, -b_ / ss_], np.concatenate([[tr.var()], np.exp(u2[1]) + a_ / ss_ * (y - mu_s)[:-1] ** 2]))
    s_s = np.sqrt(hs * (nu_s - 2) / nu_s)[ntr:]
    for a in (0.05, 0.01, 0.005):
        qg = gm2 + sg * tdist.ppf(a, gnu2); qj = var_jump(a, p2["mu"], sj, p2["nu"], p2); qs = skew_quantile(a, mu_s, s_s, nu_s, g_s)
        pin = lambda q: (a - (te < q)) * (te - q)
        res[str(a)] = dict(breach_garch=float((te < qg).mean()), breach_jump=float((te < qj).mean()),
                           kupiec_garch=kupiec(int((te < qg).sum()), len(te), a), kupiec_jump=kupiec(int((te < qj).sum()), len(te), a),
                           pinball_gain_pct=float(100 * (pin(qg).mean() - pin(qj).mean()) / pin(qg).mean()),
                           nw_t_gain=nw_t(pin(qg) - pin(qj)), mean_var_garch=float(-qg.mean()), mean_var_jump=float(-qj.mean()),
                           breach_skew=float((te < qs).mean()), kupiec_skew=kupiec(int((te < qs).sum()), len(te), a),
                           pinball_gain_skew_pct=float(100 * (pin(qg).mean() - pin(qs).mean()) / pin(qg).mean()),
                           nw_t_gain_skew=nw_t(pin(qg) - pin(qs)), mean_var_skew=float(-qs.mean()))
    out["step5"] = dict(n_test=len(te), jump_params_train={k: float(v) for k, v in p2.items()}, var=res)
    return out

if __name__ == "__main__":
    f, dcol, rcol, label, outf = sys.argv[1:6]
    d = pd.read_csv(f).dropna(subset=[rcol]); y = d[rcol].to_numpy(dtype=float)
    out = run(y, f"{label} ({d[dcol].iloc[0]}..{d[dcol].iloc[-1]})")
    json.dump(out, open(outf, "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
