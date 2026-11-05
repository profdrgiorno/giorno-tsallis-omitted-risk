"""Paper B jump arm on the 4,000-path HMM-6 manifest: size and power of the robust jump test.

Paths: regenerated exactly from DGP_MANIFEST.csv (simulate_hmm6, n=6000, burn=600; seed derivation verified).
Jump arm: identical to the Paper B pilot: rj = default_rng([seed, 20261031]); B = rj.random(n) < 0.02;
Z = rj.gamma(2, 1, n); y_jump = x - B * 0.9 * Z (mean jump 1.8).
Test on the n_train = 4800 training observations, nulls: static Student-t and GARCH(1,1)-t.
Usage: python -I full_srj.py WORKER NWORKERS OUT.csv   (resumable: skips ids already in OUT.csv)
"""
import os, sys, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("HMM6_REPO", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)                         # hmm9_srj.py (test + null fits)
import json
import hmm9_srj as H
OPTS = json.load(open(os.path.join(HERE, "scenarios.json")))   # = make_scenario_table() of script 01 (checked)
EPS, THETA = 0.02, 0.9

def gen_returns(n, burn, pars, seed):
    """Same draws as giorno_models.simulate_hmm6 (returns only; skips its filtering pass)."""
    rng = np.random.default_rng(int(seed))
    p00, p11, nu0, nu1, s0, s1 = [float(pars[k]) for k in ("p00", "p11", "nu0", "nu1", "s0", "s1")]
    total = int(n + burn); st = np.empty(total, dtype=np.int8); u = rng.random(total)
    st[0] = 0 if u[0] < (1 - p11) / (2 - p00 - p11) else 1
    for t in range(1, total):
        st[t] = 0 if (u[t] < (p00 if st[t - 1] == 0 else 1 - p11)) else 1
    z = rng.standard_t(np.where(st == 0, nu0, nu1)) * np.where(st == 0, s0, s1)
    return z[burn:]

def one(e, rng):
    x = gen_returns(int(e.n), 600, OPTS[e.scenario], int(e.seed))
    rj = np.random.default_rng([int(e.seed), 20261031]); B = rj.random(len(x)) < EPS; Z = rj.gamma(2.0, 1.0, len(x))
    out = []
    for arm, y in (("nojump", x), ("jump", x - B * THETA * Z)):
        tr = y[: int(e.n_train)]
        loc, s, nu = H.fit_t(tr); rt = H.srj(tr, loc, s, nu, rng, True); nt = H.srj(tr, loc, s, nu, rng, False)
        gm, gs, gnu, gp = H.fit_garch(tr); rg = H.srj(tr, gm, gs, gnu, rng, True); ng = H.srj(tr, gm, gs, gnu, rng, False)
        out.append(dict(id=e.id, scenario=e.scenario, rep=int(e.rep), arm=arm, njumps=int(B[: int(e.n_train)].sum()) if arm == "jump" else 0,
                        p_robJ_t=rt[3], p_rob_t=rt[1], p_naive_t=nt[1], p_robJ_g=rg[3], p_rob_g=rg[1], p_naive_g=ng[1],
                        stat_robJ_t=rt[2], stat_robJ_g=rg[2], t_loc=loc, t_nu=nu, g_nu=gnu, g_alpha=gp[1], g_beta=gp[2]))
    return out

if __name__ == "__main__":
    w, nw, outf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    man = pd.read_csv(os.path.join(REPO, "DGP_MANIFEST.csv"))
    # interleave scenarios so partial runs are balanced: order by rep, then scenario
    man = man.sort_values(["rep", "scenario"]).reset_index(drop=True)
    man = man.iloc[w::nw]
    done = set(pd.read_csv(outf).id) if os.path.exists(outf) else set()
    for e in man.itertuples():
        if e.id in done: continue
        rows = one(e, np.random.default_rng([int(e.seed), 7]))
        pd.DataFrame(rows).to_csv(outf, mode="a", header=not os.path.exists(outf), index=False)
        print(e.id, round(rows[0]["p_robJ_t"], 3), round(rows[1]["p_robJ_t"], 3), flush=True)
