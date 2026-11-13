import numpy as np
import pandas as pd

runs = pd.read_csv("results/resnet_runs.csv")
runs = runs[(runs.epochs == 300) & (~runs.fixed_labels)]
first = pd.read_csv("results/first_run.csv").set_index("dataset")

def n_test(ds):
    with open(f"data/UCR_TS_Archive_2015/{ds}/{ds}_TEST") as f:
        return sum(1 for _ in f)

rows = []
for ds, g in runs.groupby("dataset"):
    paper = g.paper_error.iloc[0]
    n = n_test(ds)
    r = {"dataset": ds, "test_series": n, "paper": paper,
         "first_run": first.test_error.get(ds, np.nan)}
    for name, flag in [("base", False), ("1d", True)]:
        s = g[g.kernel1d == flag].test_error
        r[f"{name}_n"] = len(s)
        r[f"{name}_mean"] = s.mean() if len(s) else np.nan
        r[f"{name}_std"] = s.std() if len(s) > 1 else np.nan
    # a dataset "matches" if any of our results is within 2 test series of the paper
    cands = [v for v in (r["first_run"], r["base_mean"]) if not np.isnan(v)]
    r["gap_series"] = (r["base_mean"] - paper) * n
    r["matched_within_2_series"] = any(abs(v - paper) * n <= 2 for v in cands)
    rows.append(r)

t = pd.DataFrame(rows).round(4)
t.to_csv("results/resnet_paper_vs_ours.csv", index=False)

def cell(m, s, n):
    if n == 0 or np.isnan(m):
        return "-"
    return f"{m:.3f}" + (f" ± {s:.3f}" if n > 1 else "") + f" (n={n})"

lines = ["| Dataset | Test series | Paper | First run (1500 ep) | Ours k×k (300 ep) | Ours k×1 (300 ep) | Gap k×k vs paper (series) | Within 2 series |",
         "|---|---|---|---|---|---|---|---|"]
for _, r in t.iterrows():
    fr = "-" if np.isnan(r.first_run) else f"{r.first_run:.4f}"
    lines.append(f"| {r.dataset} | {int(r.test_series)} | {r.paper:.3f} | {fr} | "
                 f"{cell(r.base_mean, r.base_std, int(r.base_n))} | "
                 f"{cell(r['1d_mean'], r['1d_std'], int(r['1d_n']))} | {r.gap_series:+.0f} | "
                 f"{'yes' if r.matched_within_2_series else 'no'} |")
open("results/resnet_paper_vs_ours.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nmatched within 2 test series:", ", ".join(t[t.matched_within_2_series].dataset))
