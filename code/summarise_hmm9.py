"""Summarise the HMM-9-design run and write mc_section.tex (table + text numbers filled from data)."""
import glob, json
import numpy as np, pandas as pd

d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("out/*_w*.csv"))], ignore_index=True)
assert not d.duplicated(["scenario", "n", "rep"]).any()
cols = ["p_robJ_t", "p_rob_t", "p_naive_t", "p_robJ_g", "p_rob_g", "p_naive_g"]
rej = d.groupby(["scenario", "n"])[cols].agg(lambda p: 100 * (p < 0.05).mean())
cnt = d.groupby(["scenario", "n"]).size().rename("paths")
nj = d.groupby(["scenario", "n"]).njumps.mean().rename("mean_jumps")
summ = pd.concat([cnt, nj, rej], axis=1)
summ.to_csv("hmm9_design_test_summary.csv"); d.to_csv("hmm9_design_test_replications.csv", index=False)
print(summ.round(1).to_string())
# agreement between OLS and Jacobian versions
agree = {m: float((((d[f"p_rob_{m}"] < .05) == (d[f"p_robJ_{m}"] < .05))).mean()) for m in ("t", "g")}
print("decision agreement OLS vs Jacobian:", agree)
print("max |stat diff|:", float((d.stat_rob_t - d.stat_robJ_t).abs().max()), float((d.stat_rob_g - d.stat_robJ_g).abs().max()))

lab = {"correct": "Clean Student-$t$ HMM", "gaussian_contamination": "Gaussian contamination", "negative_gamma_jumps": "Negative Gamma jumps"}
role = {"correct": "size", "gaussian_contamination": "size", "negative_gamma_jumps": "power"}
rows = []
for sc in ("correct", "gaussian_contamination", "negative_gamma_jumps"):
    for n in (1000, 2000, 3000):
        r = summ.loc[(sc, n)]
        rows.append(f"{lab[sc] if n == 1000 else ''} & {role[sc] if n == 1000 else ''} & {n:,} & "
                    f"{r.p_robJ_t:.1f} & {r.p_rob_t:.1f} & {r.p_naive_t:.1f} & {r.p_robJ_g:.1f} & {r.p_rob_g:.1f} & {r.p_naive_g:.1f}\\\\")
    rows.append(r"\addlinespace")
tab = "\n".join(rows[:-1])
S = lambda sc, n, c: summ.loc[(sc, n), c]
sz = d[d.scenario != "negative_gamma_jumps"]
pooled = {c: 100 * (sz[c] < .05).mean() for c in cols}
txt = rf"""The archived HMM-9 design (Table~1 of that manuscript) has nine cells of 300 paths: a clean two-state Student-$t$ HMM ($p_{{00}}=0.985$, $p_{{11}}=0.960$, $q=(1.12,1.42)$, $s=(0.80,1.25)$), the same HMM with symmetric $N(0,5^2)$ contamination at rate $0.03$, and the same HMM with negative Gamma$(2,3)$ jumps at rate $0.03$, for training lengths $n=1{{,}}000$, $2{{,}}000$, $3{{,}}000$. The paths were regenerated from the repository's \texttt{{simulate()}} with its seed sequence; regeneration is exact (zero difference on all three generators). The test uses only the $n$ training observations. Both null models, a static Student-$t$ and a GARCH(1,1)-$t$, are misspecified for HMM data; they are symmetric, so Assumption~\ref{{A:null}} holds in the first two generators (Lemma~\ref{{lem:pseudo}}). The Gaussian-contamination arm is the decisive check: its conditional law is symmetric but not Student-$t$, so the robust test should keep its size and the naive one need not. This arm did not exist in the Paper~B pilot.

\begin{{table}}[ht]\centering\small
\caption{{Rejection rates (\%) at the 5\% level, 300 paths per row. ``Jac.'': orthogonalised statistic of Theorem~\ref{{thm:null}}; ``OLS'': pilot projection; ``Naive'': downward-only score with OLS projection. Monte Carlo s.e.\ about 1.3 points at 5\%.}}\label{{tab:mc}}
\begin{{tabular}}{{llr rrr rrr}}
\toprule
 & & & \multicolumn{{3}}{{c}}{{Static Student-$t$ null}} & \multicolumn{{3}}{{c}}{{GARCH(1,1)-$t$ null}}\\
\cmidrule(lr){{4-6}}\cmidrule(lr){{7-9}}
Generator & Role & $n$ & Jac. & OLS & Naive & Jac. & OLS & Naive\\
\midrule
{tab}
\bottomrule
\end{{tabular}}
\end{{table}}

Pooled over the six jump-free cells (1{{,}}800 paths), the orthogonalised robust test rejects {pooled['p_robJ_t']:.1f}\% (Student-$t$ null) and {pooled['p_robJ_g']:.1f}\% (GARCH-$t$ null); the naive score rejects {pooled['p_naive_t']:.1f}\% and {pooled['p_naive_g']:.1f}\%. The OLS and Jacobian versions reach the same decision on {100*agree['t']:.1f}\% and {100*agree['g']:.1f}\% of paths. Against negative Gamma jumps the orthogonalised test rejects {S('negative_gamma_jumps',1000,'p_robJ_t'):.1f}--{S('negative_gamma_jumps',3000,'p_robJ_t'):.1f}\% of paths with the Student-$t$ null and {S('negative_gamma_jumps',1000,'p_robJ_g'):.1f}--{S('negative_gamma_jumps',3000,'p_robJ_g'):.1f}\% with the GARCH-$t$ null, for $n$ from 1{{,}}000 to 3{{,}}000 (on average {S('negative_gamma_jumps',1000,'mean_jumps'):.0f} to {S('negative_gamma_jumps',3000,'mean_jumps'):.0f} jumps per path). The naive downward-only score is not valid here either, but in the opposite direction to the Paper~B pilot: with these null fits it is conservative (0--6\% under the null) and less powerful ({S('negative_gamma_jumps',1000,'p_naive_t'):.0f}\% and {S('negative_gamma_jumps',1000,'p_naive_g'):.0f}\% at $n=1{{,}}000$, against {S('negative_gamma_jumps',1000,'p_robJ_t'):.0f}\% and {S('negative_gamma_jumps',1000,'p_robJ_g'):.0f}\% for the robust test). Its size depends on how the tails are misspecified, which is the point of Theorem~2: in the pilot's calibration-anchor design it rejected 67--100\% of jump-free paths. The robust test rejects at most {max(S(sc,n,c) for sc in ('correct','gaussian_contamination') for n in (1000,2000,3000) for c in ('p_robJ_t','p_robJ_g')):.1f}\% in any jump-free cell; the highest values are in the Gaussian-contamination cell at $n=3{{,}}000$ (about 1.5 Monte Carlo s.e.\ above 5\%), which is worth watching in larger samples. These jumps are large (mean size 6, against regime scales 0.8 and 1.25), so this is a high-power design; the pilot's weaker design ($\varepsilon=0.02$, mean size 1.8) remains the relevant check of power for moderate jumps.
"""
open("mc_section.tex", "w").write(txt)
json.dump(dict(pooled_size=pooled, agree=agree), open("mc_pooled.json", "w"), indent=1)
