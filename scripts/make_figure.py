import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = pd.read_csv("results/resnet_paper_vs_ours.csv").dropna(subset=["base_mean"])
t = t[t.base_n >= 1].sort_values("paper")
x = np.arange(len(t))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.3, 1]})

ax[0].errorbar(x - 0.15, t.base_mean, yerr=t.base_std.fillna(0), fmt="o", capsize=3, label="ours k×k")
d = t.dropna(subset=["1d_mean"])
ax[0].errorbar(np.searchsorted(t.dataset.values, d.dataset.values) * 0 + [list(t.dataset).index(n) + 0.15 for n in d.dataset],
               d["1d_mean"], yerr=d["1d_std"].fillna(0), fmt="s", capsize=3, label="ours k×1")
ax[0].scatter(x, t.paper, marker="x", color="k", zorder=5, label="paper")
ax[0].set_xticks(x)
ax[0].set_xticklabels(t.dataset, rotation=45, ha="right", fontsize=8)
ax[0].set_ylabel("test error")
ax[0].legend(fontsize=8)
ax[0].set_title("ResNet: test error, paper vs ours (mean ± std over seeds)", fontsize=9)

ax[1].bar(["k×k", "k×1"], [2.866, 0.528], color=["tab:blue", "tab:orange"])
ax[1].set_ylabel("parameters (millions)")
ax[1].set_title("Model size", fontsize=9)
plt.tight_layout()
plt.savefig("results/resnet_figure.png", dpi=200)
print("saved results/resnet_figure.png")
