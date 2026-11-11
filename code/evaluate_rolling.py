"""Paper B (JBF revision): evaluation of the rolling one-day VaR forecasts written by rolling.py.

For each series, model (GARCH-t, skew-t, jumps) and level (99%, 95%):
  breach rate; Kupiec unconditional coverage; Christoffersen independence and conditional coverage;
  Christoffersen-Pelletier (2004) Weibull duration test; pinball loss and Diebold-Mariano t-statistic
  (Newey-West) against GARCH-t.
Basel (99%): exceptions in the trailing 250 days -> green (0-4), yellow (5-9), red (10+) share of days;
  capital proxy max(VaR_t, (3 + plus factor) x mean VaR of the last 60 days), relative to GARCH-t.
Usage: python -I evaluate_rolling.py OUT.json roll_SPY.csv roll_GSPC.csv roll_BVSP.csv
"""
import sys, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.special import gammaln

def kupiec(I, a):
    n, x = len(I), int(I.sum())
    if x == 0: lr = -2 * n * np.log(1 - a)
    else:
        p = x / n; lr = -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - p) - x * np.log(p))
    return float(lr)

def christoffersen(I, a):
    I = I.astype(int); prev, cur = I[:-1], I[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum()); n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum()); n11 = int(((prev == 1) & (cur == 1)).sum())
    def xl(x, p): return x * np.log(p) if x > 0 else 0.0
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = xl(n00 + n10, 1 - p) + xl(n01 + n11, p)
    l1 = xl(n00, 1 - p01) + xl(n01, p01) + xl(n10, 1 - p11) + xl(n11, p11)
    lr_ind = -2 * (l0 - l1); lr_uc = kupiec(I, a)
    return float(chi2.sf(lr_ind, 1)), float(chi2.sf(lr_uc + lr_ind, 2))

def duration_test(I):
    """Christoffersen-Pelletier Weibull test of b = 1 (no duration dependence), censored first/last spells."""
    t = np.flatnonzero(I)
    if len(t) < 3: return float("nan")
    D = np.diff(t).astype(float); C = [0] * len(D)
    D = np.concatenate([[t[0] + 1], D, [len(I) - t[-1]]]); C = np.array([1] + C + [1])
    def nll(u, fix_b=False):
        a, b = np.exp(u[0]), (1.0 if fix_b else np.exp(u[1]))
        logf = np.log(b) + b * np.log(a) + (b - 1) * np.log(D) - (a * D) ** b
        logS = -(a * D) ** b
        return -np.sum(np.where(C == 1, logS, logf))
    r1 = minimize(lambda u: nll(u), [np.log(1 / D.mean()), 0.0], method="Nelder-Mead")
    r0 = minimize(lambda u: nll([u[0], 0.0], True), [np.log(1 / D.mean())], method="Nelder-Mead")
    return float(chi2.sf(max(2 * (r0.fun - r1.fun), 0), 1))

def nw_t(dv):
    v = dv - dv.mean(); T = len(v); L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    s = v @ v / T + sum(2 * (1 - l / (L + 1)) * (v[l:] @ v[:-l]) / T for l in range(1, L + 1))
    return float(dv.mean() / np.sqrt(s / T))

PLUS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}

def basel(ret, q):
    """Trailing-250-day exception counts for 99% VaR, zone shares and capital proxy series."""
    I = (ret < q).astype(int); n = len(I); var = -q
    cnt = np.array([I[max(0, i - 250):i].sum() for i in range(250, n)])
    zone = np.where(cnt <= 4, "green", np.where(cnt <= 9, "yellow", "red"))
    k = np.array([0.0 if c <= 4 else (1.0 if c >= 10 else PLUS[int(c)]) for c in cnt])
    m60 = np.array([var[i - 60:i].mean() for i in range(250, n)])
    cap = np.maximum(var[250:], (3 + k) * m60)
    return dict(green=float((zone == "green").mean()), yellow=float((zone == "yellow").mean()), red=float((zone == "red").mean())), cap

def evaluate(f):
    d = pd.read_csv(f); out = dict(n=len(d), start=str(d["date"].iloc[0]), end=str(d["date"].iloc[-1]),
                                   median_eps_jump=float(d["eps_jump"].median()), share_eps_at_bound=float((d["eps_jump"] > 0.099).mean()),
                                   median_gamma_skew=float(d["gamma_skew"].median()))
    r = d["ret"].to_numpy()
    for tag, a in (("99", 0.01), ("95", 0.05)):
        res = {}
        pin = {}
        for m in ("t", "skew", "jump"):
            q = d[f"q_{m}_{tag}"].to_numpy(); I = (r < q)
            p_ind, p_cc = christoffersen(I, a)
            pin[m] = (a - I) * (r - q)
            res[m] = dict(breach=float(I.mean()), kupiec_p=float(chi2.sf(kupiec(I, a), 1)), ind_p=p_ind, cc_p=p_cc,
                          duration_p=duration_test(I), mean_var=float(-q.mean()), pinball=float(pin[m].mean()))
        for m in ("skew", "jump"):
            res[m]["pinball_gain_pct"] = float(100 * (pin["t"].mean() - pin[m].mean()) / pin["t"].mean())
            res[m]["dm_t"] = nw_t(pin["t"] - pin[m])
        if tag == "99":
            caps = {}
            for m in ("t", "skew", "jump"):
                z, caps[m] = basel(r, d[f"q_{m}_{tag}"].to_numpy()); res[m]["basel"] = z
            for m in ("skew", "jump"):
                res[m]["capital_vs_t_pct"] = float(100 * (caps[m].mean() / caps["t"].mean() - 1))
        # subperiods
        sub = {}
        dates = d["date"].astype(str).to_numpy()
        for lab, msk in (("2010-2019", dates < "2020-01-01"), ("2020-2026", dates >= "2020-01-01")):
            sub[lab] = {m: dict(breach=float((r[msk] < d[f"q_{m}_{tag}"].to_numpy()[msk]).mean())) for m in ("t", "skew", "jump")}
            for m in ("skew", "jump"):
                sub[lab][m]["dm_t"] = nw_t(pin["t"][msk] - pin[m][msk])
                sub[lab][m]["pinball_gain_pct"] = float(100 * (pin["t"][msk].mean() - pin[m][msk].mean()) / pin["t"][msk].mean())
        res["subperiods"] = sub
        out[tag] = res
    return out

if __name__ == "__main__":
    outf = sys.argv[1]; allres = {}
    for f in sys.argv[2:]:
        lab = f.split("roll_")[-1].replace(".csv", ""); allres[lab] = evaluate(f)
    json.dump(allres, open(outf, "w"), indent=1); print(json.dumps(allres, indent=1))
