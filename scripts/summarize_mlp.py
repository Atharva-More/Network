#!/usr/bin/env python3
"""Build the MLP paper-vs-ours table, statistics and figure from the saved logs.

Run from the repository root:  python scripts/summarize_mlp.py
Outputs: results/mlp_paper_vs_ours.csv/.md, results/mlp_stats.txt, results/mlp_figure.png
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

R = "results"
PAPER_ACC = 0.752          # Wang et al. 2016, Table I, Adiac MLP error 0.248

# ---- 1. parse the Keras progress logs (authors' script, epoch-by-epoch) -------------
EPOCH = re.compile(r"Epoch (\d+)/5000\n(.*?)(?=\nEpoch \d+/5000|\Z)", re.S)
METR = re.compile(r"- loss: ([\d.e+-]+) - (?:acc|accuracy): ([\d.e+-]+) - "
                  r"val_loss: ([\d.e+-]+) - val_(?:acc|accuracy): ([\d.e+-]+)")


def parse_keras_log(path):
    txt = open(path, errors="ignore").read()
    rows = []
    for m in EPOCH.finditer(txt):
        hit = METR.findall(m.group(2))
        if hit:
            rows.append((int(m.group(1)),) + tuple(float(v) for v in hit[-1]))
    df = pd.DataFrame(rows, columns=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
    assert len(df) == 5000, f"{path}: parsed {len(df)} epochs, expected 5000"
    return df


def paper_style(df):
    return df.loc[df.train_loss.idxmin()]


def seed_of(path):
    return int(re.search(r"seed_(\d+)", path).group(1))


base = {}
for p in sorted(glob.glob(f"{R}/logs/keras2017_seed_*.log")):
    base[seed_of(p)] = parse_keras_log(p)
orig = parse_keras_log(f"{R}/logs/MLP_original_tf2_default.log")

b_rows = []
for s, d in base.items():
    sel, best = paper_style(d), d.loc[d.test_acc.idxmax()]
    b_rows.append(dict(seed=s, epoch=int(sel.epoch), train_loss=sel.train_loss,
                       train_acc=sel.train_acc, selected=sel.test_acc,
                       final=d.test_acc.iloc[-1], oracle=best.test_acc))
B = pd.DataFrame(b_rows)
B.to_csv(f"{R}/mlp_baseline_keras2017_summary.csv", index=False)

a1 = pd.read_csv(f"{R}/A1_official_test_results.csv")
o5 = pd.read_csv(f"{R}/O5_summary.csv")
so = paper_style(orig)

# ---- 2. paper-vs-ours table -------------------------------------------------------------
def ms(x):
    return f"{100*np.mean(x):.2f} ± {100*np.std(x, ddof=1):.2f}" if len(x) > 1 else f"{100*x[0]:.2f}"

table = pd.DataFrame([
    ["Paper (Wang et al. 2016), 3x500 MLP, Adadelta", "1 (not stated)", "5000", "75.20"],
    ["Repo as published, TF 2.x (Adadelta default lr=0.001)", "1", "5000", f"{100*so.test_acc:.2f}"],
    ["Repo as published, 2017 Keras/TF1 (Adadelta lr=1.0), min train loss", "5", "5000", ms(B.selected.values)],
    ["Ours A1: 128-unit MLP, Adam, final epoch", "5", "186", ms(a1.test_accuracy.values)],
    ["Ours O5: 128-unit MLP, Adam, min train loss", "5", "5000", ms(o5.selected_test_accuracy_history.values)],
], columns=["Setting", "Runs", "Epochs", "Adiac test accuracy (%)"])
table.to_csv(f"{R}/mlp_paper_vs_ours.csv", index=False)
with open(f"{R}/mlp_paper_vs_ours.md", "w") as f:
    f.write("| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n")
    for r in table.itertuples(index=False):
        f.write("| " + " | ".join(r) + " |\n")

# ---- 3. statistics --------------------------------------------------------------------
sel_o5 = o5.selected_test_accuracy_history.values
fin_o5 = o5.final_epoch_test_accuracy.values
orc_o5 = o5.max_test_accuracy_DIAGNOSTIC_ONLY.values
t_vs_paper = stats.ttest_1samp(sel_o5, PAPER_ACC)
t_vs_base = stats.ttest_ind(sel_o5, B.selected.values, equal_var=False)
t_a1_o5 = stats.ttest_rel(sel_o5, a1.test_accuracy.values)
lines = [
    f"Keras-2017 baseline (5 seeds): mean {100*B.selected.mean():.2f}  std {100*B.selected.std(ddof=1):.2f}  "
    f"min {100*B.selected.min():.2f}  max {100*B.selected.max():.2f}",
    f"  final-epoch mean {100*B.final.mean():.2f}; oracle best-test-epoch mean {100*B.oracle.mean():.2f} (diagnostic)",
    f"  training accuracy at selected epoch (dropout active): {100*B.train_acc.mean():.1f}%  "
    f"(paper states training accuracy 'almost all 100%')",
    f"TF2 default-lr run: selected epoch {int(so.epoch)}, train loss {so.train_loss:.3f}, test acc {100*so.test_acc:.2f}",
    f"A1 (186 ep): mean {100*a1.test_accuracy.mean():.2f}  std {100*a1.test_accuracy.std(ddof=1):.2f}",
    f"O5 (5000 ep, min-train-loss): mean {100*sel_o5.mean():.2f}  std {100*sel_o5.std(ddof=1):.2f}  "
    f"median {100*np.median(sel_o5):.2f}  min {100*sel_o5.min():.2f}  max {100*sel_o5.max():.2f}",
    f"  final-epoch mean {100*fin_o5.mean():.2f}; oracle best-test-epoch mean {100*orc_o5.mean():.2f} (diagnostic)",
    f"  seeds above paper 75.2%: {(sel_o5 > PAPER_ACC).sum()} of {len(sel_o5)}",
    f"  selected epochs: {o5.selected_epoch_min_train_loss.tolist()} (of 5000)",
    f"  test loss at selected epoch: mean {o5.selected_test_loss_reevaluated.mean():.2f} "
    f"(A1 mean {a1.test_loss.mean():.2f})",
    f"O5 vs paper (one-sample t): t={t_vs_paper.statistic:.2f}, p={t_vs_paper.pvalue:.3f}",
    f"O5 vs Keras-2017 baseline (Welch t): t={t_vs_base.statistic:.2f}, p={t_vs_base.pvalue:.2e}",
    f"O5 vs A1 same seeds (paired t): t={t_a1_o5.statistic:.2f}, p={t_a1_o5.pvalue:.4f}",
    f"Parameters: original 3x500 MLP = {176*500+500 + 2*(500*500+500) + 500*37+37:,}; ours = {176*128+128 + 128*37+37:,}",
]
open(f"{R}/mlp_stats.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print()
print(open(f"{R}/mlp_paper_vs_ours.md").read())

# ---- 4. figure: (a) accuracy by setting, (b) test accuracy vs epoch ------------------------
c_base, c_ours, c_ref = "#c0504d", "#1f6fb2", "#444444"
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})

labels = ["Repo\nTF2", "Repo\n2017", "A1\n186 ep", "O5\n5000 ep"]
vals = [[so.test_acc], B.selected.values, a1.test_accuracy.values, sel_o5]
cols = [c_base, c_base, c_ours, c_ours]
for i, (v, c) in enumerate(zip(vals, cols)):
    ax[0].bar(i, 100 * np.mean(v), color=c, alpha=.35, width=.65)
    ax[0].scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), 100 * np.asarray(v),
                  s=11, color=c, zorder=3)
ax[0].axhline(75.2, color=c_ref, ls="--", lw=1)
ax[0].text(-0.45, 77, "paper 75.2", ha="left", fontsize=7, color=c_ref)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labels, fontsize=7)
ax[0].set_ylabel("Adiac test accuracy (%)", fontsize=8); ax[0].set_ylim(0, 85)
ax[0].tick_params(axis="y", labelsize=7); ax[0].set_title("(a) final numbers", fontsize=8)

# (b) baseline curves (all epochs, smoothed) vs O5 checkpoints printed every 250 epochs
def smooth(x, w=50):
    return pd.Series(x).rolling(w, min_periods=1).mean().values

for s, d in base.items():
    ax[1].plot(d.epoch, 100 * smooth(d.test_acc), color=c_base, lw=.8, alpha=.8,
               label="Repo 2017 (5 seeds)" if s == min(base) else None)
log = open(f"{R}/logs/O5_training.log").read()
for k, blk in enumerate(re.split(r"Starting seed", log)[1:]):
    pts = [(int(e), float(a)) for e, a in re.findall(r"Epoch\s+(\d+) \| train_loss [\d.]+ \| train_acc [\d.]+ \| test_acc ([\d.]+)", blk)]
    pts = [p for p in pts if p[0] > 1]
    ax[1].plot([e for e,_ in pts], [100*a for _,a in pts], color=c_ours, lw=.9, alpha=.85, marker="o", ms=1.8,
               label="Ours O5 (5 seeds)" if k == 0 else None)
ax[1].axhline(75.2, color=c_ref, ls="--", lw=1)
ax[1].set_xlabel("epoch", fontsize=8); ax[1].set_ylabel("test accuracy (%)", fontsize=8)
ax[1].tick_params(labelsize=7); ax[1].set_ylim(0, 85); ax[1].legend(fontsize=6.5, loc="lower right")
ax[1].set_title("(b) test accuracy during training", fontsize=8)
plt.tight_layout()
plt.savefig(f"{R}/mlp_figure.png", dpi=300)
print("saved", f"{R}/mlp_figure.png")
