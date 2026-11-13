import os
import numpy as np
import pandas as pd

df = pd.read_csv("results/resnet_runs.csv")
df = df[(df.epochs == 300) & (~df.fixed_labels)]

def n_test(ds):
    with open(f"data/UCR_TS_Archive_2015/{ds}/{ds}_TEST") as f:
        return sum(1 for _ in f)

rows = []
for ds, g in df.groupby("dataset"):
    paper = g.paper_error.iloc[0]
    row = {"dataset": ds, "test_series": n_test(ds), "paper": paper}
    for name, flag in [("base", False), ("1d", True)]:
        s = g[g.kernel1d == flag].test_error
        row[f"{name}_n"] = len(s)
        row[f"{name}_mean"] = s.mean() if len(s) else np.nan
        row[f"{name}_std"] = s.std() if len(s) > 1 else np.nan
    row["gap_series"] = (row["base_mean"] - paper) * row["test_series"]
    rows.append(row)

t = pd.DataFrame(rows).round(4)
t.to_csv("results/resnet_paper_vs_ours.csv", index=False)

def cell(m, s, n):
    if n == 0 or np.isnan(m):
        return "-"
    return f"{m:.3f}" + (f" ± {s:.3f}" if n > 1 else "") + f" (n={n})"

lines = ["| Dataset | Test series | Paper | Ours k×k (300 ep) | Ours k×1 (300 ep) | Gap k×k vs paper (series) |",
         "|---|---|---|---|---|---|"]
for _, r in t.iterrows():
    lines.append(f"| {r.dataset} | {int(r.test_series)} | {r.paper:.3f} | "
                 f"{cell(r.base_mean, r.base_std, int(r.base_n))} | "
                 f"{cell(r['1d_mean'], r['1d_std'], int(r['1d_n']))} | {r.gap_series:+.0f} |")
open("results/resnet_paper_vs_ours.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
