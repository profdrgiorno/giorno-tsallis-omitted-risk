"""Merge b12_w*.csv (copy the files from both computers into one folder first) and summarise B1 + B2."""
import glob, pandas as pd, numpy as np
d = pd.concat([pd.read_csv(f) for f in glob.glob("b12_w*.csv")], ignore_index=True).drop_duplicates(["id", "arm"])
print("paths:", d.id.nunique(), "rows:", len(d))
tests = ["p_rob_t", "p_rob_g", "p_naive_g", "p_ks_g", "p_skew_g"]
rej = d.groupby(["arm", "scenario"])[tests].agg(lambda p: 100 * (p < 0.05).mean()).round(1)
rej["paths"] = d.groupby(["arm", "scenario"]).size()
pooled = d.groupby("arm")[tests].agg(lambda p: 100 * (p < 0.05).mean()).round(1)
print(rej.to_string()); print("\nPooled:\n", pooled.to_string())
# Follow-up among paths where the robust GARCH-t test rejects
r = d[d.p_rob_g < 0.05].copy()
r["choice"] = np.select([r.vuong > 1.96, r.vuong < -1.96], ["skew-t", "jumps"], "undecided")
sel = pd.crosstab([r.arm], r.choice, normalize="index").mul(100).round(1)
sel["rejected_paths"] = r.groupby("arm").size()
print("\nAfter rejection, Vuong choice (% of rejected paths):\n", sel.to_string())
sel2 = pd.crosstab([r.arm, r.scenario], r.choice, normalize="index").mul(100).round(1)
rej.to_csv("b12_rejection_rates.csv"); pooled.to_csv("b12_pooled.csv"); sel.to_csv("b12_vuong_choice.csv"); sel2.to_csv("b12_vuong_choice_by_design.csv")
d.to_csv("b12_all_replications.csv", index=False)
print("\nWritten: b12_rejection_rates.csv, b12_pooled.csv, b12_vuong_choice.csv, b12_vuong_choice_by_design.csv, b12_all_replications.csv")
