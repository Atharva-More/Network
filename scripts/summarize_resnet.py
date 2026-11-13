import pandas as pd

df = pd.read_csv("results/resnet_runs.csv")
g = (df.groupby(["dataset", "epochs", "fixed_labels", "kernel1d"])
       .agg(n=("test_error", "size"), mean=("test_error", "mean"),
            std=("test_error", "std"), paper=("paper_error", "first"),
            minutes=("seconds", lambda s: round(s.mean() / 60, 1)))
       .reset_index())
g["diff_vs_paper"] = (g["mean"] - g["paper"]).round(4)
g["mean"] = g["mean"].round(4)
g["std"] = g["std"].round(4)
print(g.to_string(index=False))
g.to_csv("results/resnet_summary.csv", index=False)
