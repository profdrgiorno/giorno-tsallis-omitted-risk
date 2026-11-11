import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("full_w*.csv")], ignore_index=True).drop_duplicates(["id", "arm"])
c = ["p_robJ_t", "p_robJ_g", "p_rob_t", "p_rob_g", "p_naive_t", "p_naive_g"]
t = d.groupby(["arm", "scenario"])[c].agg(lambda p: 100 * (p < 0.05).mean()).round(1)
t["paths"] = d.groupby(["arm", "scenario"]).size()
print(t.to_string()); print(d.groupby("arm")[c].agg(lambda p: 100 * (p < 0.05).mean()).round(1))
t.to_csv("full_jump_arm_summary.csv"); d.to_csv("full_jump_arm_replications.csv", index=False)
