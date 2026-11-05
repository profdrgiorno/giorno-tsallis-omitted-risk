import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("ks_w*.csv")], ignore_index=True).drop_duplicates(["id", "arm"])
print("paths:", d.id.nunique())
t = d.groupby(["arm", "scenario"]).p_ks2_g.agg(lambda p: 100 * (p < 0.05).mean()).round(1).unstack(0)
print(t.to_string()); print("\nPooled:\n", d.groupby("arm").p_ks2_g.agg(lambda p: 100 * (p < 0.05).mean()).round(1).to_string())
t.to_csv("ks_rejection_rates.csv"); d.to_csv("ks_all_replications.csv", index=False)
