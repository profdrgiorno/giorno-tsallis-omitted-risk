"""Corrected Bai-Ng-type symmetry test for the B2 comparison (replaces p_ks_g of b12_run.py). Version 2.

Version 2 (3 Nov, evening): each sign-flip replicate re-estimates the location, as the original fit did.
Without this, the estimated location makes the residuals look more symmetric than the replicates and the test
never rejects under the null (version 1: null p-values had median 0.86). This is the estimation effect that the
martingale transformation of Bai and Ng (2001) removes; re-estimation inside the bootstrap removes it here.

The earlier version centred the residuals at their median before computing the statistic but not inside the
sign-flip replicates, which made the test far too conservative (null p-values had median 0.76). Here the
residuals are centred at the fitted GARCH location for both the statistic and every replicate, so under
conditional symmetry the sign-flipped series has the same law as the original (up to estimation of the location).
Statistic: sqrt(T) * sup_x |F_T(-x) - (1 - F_T(x-))| over 60 quantiles of |z| from the 2nd to the 99.9th percentile; two-sided; 499 sign flips.
Usage: python -I ks_rerun.py WORKER NWORKERS OUT.csv   (HMM6_REPO = folder with DGP_MANIFEST.csv)
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import b12_run as R, hmm9_srj as H

def ks_stat(z, grid):
    return np.sqrt(len(z)) * np.max(np.abs((z[:, None] <= -grid).mean(0) - (z[:, None] >= grid).mean(0)))

def recentre(z, nu, it=4):
    """Re-estimate the location of a residual series by Newton steps on the Student-t location score,
    mimicking the estimation of the location in the original fit."""
    m = 0.0
    for _ in range(it):
        u = z - m; m += ((nu + 1) * u / (nu + u * u)).sum() / ((nu + 1) * (nu - u * u) / (nu + u * u) ** 2).sum()
    return z - m

def ks_flip(z, nu, rng, B=499):
    grid = np.quantile(np.abs(z), np.linspace(0.02, 0.999, 60)); s0 = ks_stat(z, grid)
    sims = np.array([ks_stat(recentre(z * rng.choice([-1.0, 1.0], len(z)), nu), grid) for _ in range(B)])
    return float((1 + (sims >= s0).sum()) / (B + 1))

def one(e, rng):
    pars = R.OPTS[e.scenario]; n = int(e.n); ntr = int(e.n_train)
    x, st = R.gen(n, 600, pars, int(e.seed))
    rj = np.random.default_rng([int(e.seed), 20261031]); ub = rj.random(n); Z = rj.gamma(2.0, 1.0, n)
    uni = np.random.default_rng([int(e.seed), 20261103]).random(n)
    arms = {"sym": x, "jump": x - (ub < R.EPS) * R.THETA * Z, "jumpbig": x - (ub < R.EPS_BIG) * R.THETA_BIG * Z,
            "skew90": R.skew_arm(x, st, pars, 0.90, uni), "skew95": R.skew_arm(x, st, pars, 0.95, uni)}
    out = []
    for arm, y in arms.items():
        tr = y[:ntr]; gm, gs, gnu, _ = H.fit_garch(tr)
        out.append(dict(id=e.id, scenario=e.scenario, rep=int(e.rep), arm=arm, p_ks2_g=ks_flip((tr - gm) / gs, gnu, rng)))
    return out

if __name__ == "__main__":
    w, nw, outf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    man = pd.read_csv(os.path.join(os.environ.get("HMM6_REPO", HERE), "DGP_MANIFEST.csv")).sort_values(["rep", "scenario"]).reset_index(drop=True).iloc[w::nw]
    done = set(pd.read_csv(outf).id) if os.path.exists(outf) else set()
    for e in man.itertuples():
        if e.id in done: continue
        rows = one(e, np.random.default_rng([int(e.seed), 13]))
        pd.DataFrame(rows).to_csv(outf, mode="a", header=not os.path.exists(outf), index=False)
        print(e.id, " ".join(f"{r['arm']}:{r['p_ks2_g']:.3f}" for r in rows), flush=True)
