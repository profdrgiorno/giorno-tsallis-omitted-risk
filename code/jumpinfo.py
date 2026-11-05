"""A3: per-observation Fisher information about the jump size theta (and shape k) in
p(y) = (1-eps) f(y) + eps h(y),  h(y) = E f(y + theta Z),  Z ~ Gamma(k, 1),  f = Student-t(nu, scale s).
Checks: C_f = int (d h)^2 / f < inf, the bounds eps^2 C_{f+h} <= I <= min(eps C_h, eps^2 C_f/(1-eps)),
and the implied Cramer-Rao standard deviation of theta-hat, 1/sqrt(T I), for the designs used in the paper."""
import numpy as np
from scipy.stats import t as tdist, gamma as gdist
from scipy.special import digamma, roots_genlaguerre, gammaln

def pieces(nu, s, k, th, y):
    # generalized Laguerre quadrature for Z ~ Gamma(k,1)
    x, w = roots_genlaguerre(80, k - 1); w = w / np.exp(gammaln(k))
    F = lambda v: tdist.pdf(v / s, nu) / s
    dF = lambda v: -(nu + 1) * (v / s) / (nu + (v / s) ** 2) * F(v) / s
    f = F(y)
    h = sum(wi * F(y + th * xi) for xi, wi in zip(x, w))
    dh_th = sum(wi * xi * dF(y + th * xi) for xi, wi in zip(x, w))
    dh_k = sum(wi * F(y + th * xi) * (np.log(xi) - digamma(k)) for xi, wi in zip(x, w))
    return f, h, dh_th, dh_k

def info(nu, s, k, th, eps, lo=-400, hi=200, n=600001):
    y = np.linspace(lo, hi, n); dy = y[1] - y[0]
    f, h, a, b = pieces(nu, s, k, th, y); p = (1 - eps) * f + eps * h
    I = lambda d, q: float(np.sum(d * d / q) * dy)
    return dict(I_theta=eps ** 2 * I(a, p), I_k=eps ** 2 * I(b, p),
                Cf_theta=I(a, f), Ch_theta=I(a, h), Cfh_theta=I(a, f + h), Cf_k=I(b, f))

if __name__ == "__main__":
    print("tail check: C_f over growing ranges (finite => converges)")
    for lo in (-100, -400, -1600):
        r = info(4.0, 1.0, 2.0, 3.0, 0.03, lo=lo, hi=-lo / 2, n=int(-lo * 1500))
        print(f"  range [{lo},{-lo//2}]: C_f(theta)={r['Cf_theta']:.4f}  C_f(k)={r['Cf_k']:.4f}")
    designs = [("HMM-9 design: eps=.03, k=2, theta=3 (mean 6)", 0.03, 2.0, 3.0, (1000, 2000, 3000)),
               ("Paper B large jumps: eps=.01, theta=3", 0.01, 2.0, 3.0, (4800,)),
               ("Paper B moderate jumps: eps=.02, theta=.9 (mean 1.8)", 0.02, 2.0, 0.9, (4800,))]
    for nu in (4.0, 8.0):
        print(f"\nbase Student-t nu={nu}, scale 1")
        for lab, eps, k, th, Ts in designs:
            r = info(nu, 1.0, k, th, eps)
            lb, ub = eps ** 2 * r["Cfh_theta"], min(eps * r["Ch_theta"], eps ** 2 * r["Cf_theta"] / (1 - eps))
            sds = ", ".join(f"T={T}: sd(theta)>={1/np.sqrt(T*r['I_theta']):.2f}, sd(k)>={1/np.sqrt(T*r['I_k']):.2f}" for T in Ts)
            print(f"  {lab}\n    I_theta={r['I_theta']:.2e}  bounds [{lb:.2e}, {ub:.2e}]   eps*C_h={eps*r['Ch_theta']:.2e}  eps^2*C_f={eps**2*r['Cf_theta']:.2e}\n    {sds}")
