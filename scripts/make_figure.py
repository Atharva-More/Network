import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = pd.read_csv("results/resnet_paper_vs_ours.csv").sort_values("paper").reset_index(drop=True)
pos = {d: i for i, d in enumerate(t.dataset)}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.4, 1]})

b = t.dropna(subset=["base_mean"])
ax[0].errorbar([pos[d] - 0.2 for d in b.dataset], b.base_mean, yerr=b.base_std.fillna(0),
               fmt="o", capsize=3, label="ours k×k, 300 ep")
k = t.dropna(subset=["1d_mean"])
ax[0].errorbar([pos[d] + 0.2 for d in k.dataset], k["1d_mean"], yerr=k["1d_std"].fillna(0),
               fmt="s", capsize=3, label="ours k×1, 300 ep")
f = t.dropna(subset=["first_run"])
ax[0].scatter([pos[d] for d in f.dataset], f.first_run, marker="*", s=130, color="tab:green",
              zorder=6, label="first run, 1500 ep")
ax[0].scatter(range(len(t)), t.paper, marker="x", color="k", zorder=5, label="paper")
ax[0].set_xticks(range(len(t)))
ax[0].set_xticklabels(t.dataset, rotation=45, ha="right", fontsize=8)
ax[0].set_ylabel("test error")
ax[0].legend(fontsize=7)
ax[0].set_title("ResNet test error: paper vs ours (mean ± std over seeds)", fontsize=9)

ax[1].bar(["k×k", "k×1"], [2.866, 0.528], color=["tab:blue", "tab:orange"])
ax[1].set_ylabel("parameters (millions)")
ax[1].set_title("Model size", fontsize=9)
plt.tight_layout()
plt.savefig("results/resnet_figure.png", dpi=200)
print("saved results/resnet_figure.png")
