"""Numerical check of the local-power formula (Theorem B4) and of the null limit (Theorem B3).

DGP under H0: two-state HMM of the archived HMM-9 design (clean arm), centred, reflection invariant.
Local alternative: Y_t = X_t - B_t theta0 Z_t, B_t ~ Bernoulli(c/sqrt(T)), Z_t ~ Gamma(2,1), theta0 = 0.9.
Fitted null model: static Student-t (misspecified). Orthogonalised statistic Z(theta) (Jacobian kappa).
Predicted mean of Z(theta): c * lambda(theta) / sqrt(Omega(theta,theta)), with
lambda(theta) = E_0 E_G[ xi(X_t - G) - xi(X_t) ] for the static model (single-impulse functional).
lambda and Omega are computed at the pseudo-true value from one long clean path.
Usage: python localpower_check.py T C NPATH OUT.json
"""
import json, sys
import numpy as np
from scipy.special import gammaln, roots_genlaguerre
sys.path.insert(0, "/home/claude/paperB_work")
import hmm9_srj as H

def hmm_path(T, rng):
    z = np.empty(T, np.int8); z[0] = rng.random() > 0.7272727272727273
    u = rng.random(T)
    for t in range(1, T):
        z[t] = u[t] >= H.PTRUE[z[t - 1], 0]
    return rng.standard_t(H.NUTRUE[z]) * H.STRUE[z]

def xi_matrix(y, loc, sc, nu, kappa=None):
    smu = H._smu(y, loc, sc, nu); dl = 1e-4
    dsmu = -(H._smu(y + dl, loc, sc, nu) - H._smu(y - dl, loc, sc, nu)) / (2 * dl)
    X, K = [], []
    for i, th in enumerate(H.GRID):
        a = H._afun(y, loc, sc, nu, th, True)
        if kappa is None:
            da = -(H._afun(y + dl, loc, sc, nu, th, True) - H._afun(y - dl, loc, sc, nu, th, True)) / (2 * dl)
            k = da.mean() / dsmu.mean()
        else:
            k = kappa[i]
        K.append(k); X.append(a - k * smu)
    return np.array(X), np.array(K)

if __name__ == "__main__":
    T, c, npath, out = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    th0 = 0.9
    rng = np.random.default_rng(777)
    # pseudo-true value and population quantities from a long clean path
    xl = hmm_path(400_000, rng); loc, sc, nu = H.fit_t(xl)
    Xi, kap = xi_matrix(xl, loc, sc, nu)
    Omega = (Xi ** 2).mean(1)
    G = th0 * rng.gamma(2.0, 1.0, xl.size)
    Xi_j, _ = xi_matrix(xl - G, loc, sc, nu, kappa=kap)
    lam = (Xi_j - Xi).mean(1); lam_se = (Xi_j - Xi).std(1) / np.sqrt(xl.size)
    pred = c * lam / np.sqrt(Omega)
    eps = c / np.sqrt(T); Zs = []
    for p in range(npath):
        x = hmm_path(T, rng); B = rng.random(T) < eps; y = x - B * th0 * rng.gamma(2.0, 1.0, T)
        l, s, v = H.fit_t(y); A, _ = xi_matrix(y, l, s, v)
        Zs.append((A.sum(1) / np.sqrt((A ** 2).sum(1))).tolist())
    Zs = np.array(Zs)
    res = dict(T=T, c=c, npath=npath, pseudo_true=[loc, sc, nu], kappa=kap.tolist(), lam=lam.tolist(),
               lam_se=lam_se.tolist(), Omega=Omega.tolist(), predicted_meanZ=pred.tolist(),
               sim_meanZ=Zs.mean(0).tolist(), sim_sdZ=Zs.std(0).tolist(),
               sim_se_meanZ=(Zs.std(0) / np.sqrt(npath)).tolist())
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
