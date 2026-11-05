"""Merge b3_w*.csv from both computers and summarise B3."""
import glob, pandas as pd, numpy as np
d = pd.concat([pd.read_csv(f) for f in glob.glob("b3_w*.csv")], ignore_index=True).drop_duplicates(["id", "arm", "model", "level"])
print("paths:", d.id.nunique())
d["kupiec_rej"] = d.kupiec_p < 0.05
g = d.groupby(["level", "model", "arm"]).agg(breach_pct=("breach_rate", lambda x: 100 * x.mean()), kupiec_rej_pct=("kupiec_rej", lambda x: 100 * x.mean()),
        shortfall_pct=("rel_shortfall", lambda x: 100 * x.mean()), exp_regret=("exp_regret", "mean"), pin_diff=("pin_diff", "mean")).round(4)
print(g.to_string())
b = d[d.level == 0.01].copy()
bb = b.groupby(["model", "arm"]).agg(yellow_pct=("basel_zone", lambda z: 100 * (z == "yellow").mean()), red_pct=("basel_zone", lambda z: 100 * (z == "red").mean()),
        capital_ratio=("capital_ratio", "mean"), es975_shortfall_pct=("es975_rel_shortfall", lambda x: 100 * x.mean())).round(3)
print("\n99% Basel / capital / ES:\n", bb.to_string())
# Corollary 3 check: regret increase with eps; ratio of increases (eps=.02 vs .01) is 4 under eps^2 scaling
r = d.pivot_table(index=["id", "model", "level"], columns="arm", values="exp_regret").reset_index()
r["inc01"] = r.jump01 - r.nojump; r["inc02"] = r.jump02 - r.nojump
c = r.groupby(["model", "level"]).agg(inc01=("inc01", "mean"), inc02=("inc02", "mean")); c["ratio_02_over_01"] = c.inc02 / c.inc01
print("\nRegret increase (Corollary 3; ratio 4 = second order, 2 = first order):\n", c.round(6).to_string())
bydesign = d[d.level == 0.01].groupby(["scenario", "model", "arm"]).agg(shortfall_pct=("rel_shortfall", lambda x: 100 * x.mean()), breach_pct=("breach_rate", lambda x: 100 * x.mean())).round(2)
g.to_csv("b3_summary.csv"); bb.to_csv("b3_basel_capital_es.csv"); c.to_csv("b3_regret_scaling.csv"); bydesign.to_csv("b3_by_design_99.csv"); d.to_csv("b3_all_rows.csv", index=False)
print("\nWritten: b3_summary.csv, b3_basel_capital_es.csv, b3_regret_scaling.csv, b3_by_design_99.csv, b3_all_rows.csv")
