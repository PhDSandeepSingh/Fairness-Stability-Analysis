import pandas as pd, numpy as np, json
from scipy import stats

T = pd.read_csv("results/table_paired_deltas.csv")
H = T[T.metric.isin(["dpd", "eod"])]
t_crit = stats.t.ppf(1 - (0.05 / 44) / 2, 19)
t_80 = stats.t.ppf(0.80, 19)
out = {}
for tag, ratio in (("525", 0.3 / 0.525), ("07", 0.3 / 0.7)):
    fac = np.sqrt(1 / 20 + ratio)
    out[f"crit_{tag}"] = float(t_crit * fac)
    out[f"mde80_{tag}"] = float((t_crit + t_80) * fac)

# Per-dataset metric-unit equivalents (median sd_delta * crit_525 and * mde80_525)
rows = []
for (ds, met), g in H.groupby(["dataset", "metric"]):
    med_sd = float(g.sd_delta.median())
    rows.append(dict(dataset=ds, metric=met, med_sd=med_sd,
                     crit_525_units=med_sd * out["crit_525"],
                     mde80_525_units=med_sd * out["mde80_525"]))
pd.DataFrame(rows).to_csv("results/table_mde_units.csv", index=False)
json.dump(out, open("results/mde_summary.json", "w"), indent=2)
print("MDE summary:", json.dumps(out, indent=2))
print(pd.DataFrame(rows).round(3).to_string(index=False))
