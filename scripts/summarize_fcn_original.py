import numpy as np
import pandas as pd

o = pd.read_csv("results/fcn_original_runs.csv")
o = o[~o.fixed_labels]
m = pd.read_csv("results/fcn_runs.csv").drop_duplicates(subset=["dataset", "seed", "epochs", "se", "ls"])
m = m[(m.epochs == 2000) & (~m.se) & (m.ls == 0)]

def n_test(ds):
    with open(f"data/UCR_TS_Archive_2015/{ds}/{ds}_TEST") as f:
        return sum(1 for _ in f)

def stat(s):
    return len(s), (s.mean() if len(s) else np.nan), (s.std() if len(s) > 1 else np.nan)

rows = []
for ds in sorted(set(o.dataset) | set(m.dataset)):
    r = {"dataset": ds, "test_series": n_test(ds), "paper": o[o.dataset == ds].paper_error.iloc[0] if (o.dataset == ds).any() else np.nan}
    r["orig_kxk_n"], r["orig_kxk_mean"], r["orig_kxk_std"] = stat(o[(o.dataset == ds) & (~o.kernel1d)].err_trainloss_epoch)
    r["orig_kx1_n"], r["orig_kx1_mean"], r["orig_kx1_std"] = stat(o[(o.dataset == ds) & (o.kernel1d)].err_trainloss_epoch)
    r["mine_kx1_n"], r["mine_kx1_mean"], r["mine_kx1_std"] = stat(m[m.dataset == ds].err_trainloss_epoch)
    r["gap_kxk_series"] = (r["orig_kxk_mean"] - r["paper"]) * r["test_series"]
    rows.append(r)

t = pd.DataFrame(rows).round(4)
print(t.to_string(index=False))
t.to_csv("results/fcn_paper_vs_ours.csv", index=False)
