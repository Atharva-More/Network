import pandas as pd
from scipy.stats import wilcoxon

PAPER = {"Adiac": .143, "SwedishLeaf": .034, "FaceFour": .068, "ECGFiveDays": .015,
         "Beef": .25, "OliveOil": .167, "Gun_Point": 0.0}
df = pd.read_csv("results/fcn_runs.csv")
df["variant"] = df.apply(lambda r: "base" if (not r.se and r.ls == 0) else
                         ("+SE" if (r.se and r.ls == 0) else ("+LS" if not r.se else "+SE+LS")), axis=1)

for col in ["err_trainloss_epoch", "err_last"]:
    g = (df.groupby(["dataset", "variant"])
           .agg(n=("seed", "size"), mean=(col, "mean"), std=(col, "std"),
                minutes=("seconds", lambda s: round(s.mean() / 60, 1)), params=("params", "first"))
           .round(4).reset_index())
    g["paper"] = g.dataset.map(PAPER)
    print(f"\n=== test error at: {col} ===")
    print(g.to_string(index=False))

    piv = df.pivot_table(index=["dataset", "seed"], columns="variant", values=col)
    print("\npaired difference vs base, mean over seeds (negative = better):")
    rows = []
    for ds, sub in piv.groupby(level=0):
        r = {"dataset": ds}
        for v in ["+SE", "+LS", "+SE+LS"]:
            if v in sub and "base" in sub:
                d = (sub[v] - sub["base"]).dropna()
                r[v] = round(d.mean(), 4) if len(d) else None
                r[v + " wins"] = f"{int((d < 0).sum())}/{len(d)}"
        rows.append(r)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    for v in ["+SE", "+LS", "+SE+LS"]:
        if v in t and t[v].notna().sum() >= 5:
            print(f"Wilcoxon over datasets, {v}: p =", round(wilcoxon(t[v].dropna()).pvalue, 4))
