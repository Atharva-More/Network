#!/usr/bin/env python3
"""Simplified MLP for UCR Adiac (our improvement over the authors' 3x500 MLP).

Architecture : 176 -> 128 (ReLU) -> 37 (softmax), 27,429 parameters
Optimiser    : Adam, lr = 0.001, batch size 16, no dropout, no L2
Selection    : the paper's rule - report the test accuracy of the epoch with the
               LOWEST TRAINING LOSS (the test set is never used to choose anything).
               The best-test-epoch is stored as a DIAGNOSTIC column only.

Examples (run from the repository root)
    # A1: 186 epochs, final epoch (first look at the official test set)
    python src/mlp_adiac.py --epochs 186  --tag A1
    # O5: 5000 epochs, lowest-training-loss epoch (paper-style selection)
    python src/mlp_adiac.py --epochs 5000 --tag O5
"""
import argparse
import os
import random

import numpy as np
import pandas as pd

SEEDS = [813306, 813307, 813308, 813309, 813310]


def read_ucr(path):
    data = np.loadtxt(path, delimiter=",")
    return data[:, 1:], data[:, 0]


def load_dataset(root, name):
    """Load a UCR-2015 dataset.  Labels are mapped with np.unique (exact integers),
    which avoids the float-truncation label bug of the authors' rescaling line."""
    xtr, ytr = read_ucr(os.path.join(root, name, f"{name}_TRAIN"))
    xte, yte = read_ucr(os.path.join(root, name, f"{name}_TEST"))
    classes = np.unique(np.concatenate([ytr, yte]))
    lut = {c: i for i, c in enumerate(classes)}
    ytr = np.array([lut[c] for c in ytr])
    yte = np.array([lut[c] for c in yte])
    # z-normalisation with the mean / std of the TRAINING part (as in the paper)
    mu, sd = xtr.mean(), xtr.std()
    return (xtr - mu) / sd, ytr, (xte - mu) / sd, yte, len(classes)


def set_seed(seed):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build(input_dim, n_classes, hidden=128, lr=1e-3):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input((input_dim,)),
        tf.keras.layers.Dense(hidden, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def run_seed(seed, args, data):
    xtr, ytr, xte, yte, k = data
    set_seed(seed)
    model = build(xtr.shape[1], k, args.hidden, args.lr)
    bs = min(len(xtr) // 10, 16)
    h = model.fit(xtr, ytr, batch_size=bs, epochs=args.epochs, verbose=0,
                  validation_data=(xte, yte), shuffle=True).history
    hist = pd.DataFrame({"epoch": np.arange(1, args.epochs + 1),
                         "train_loss": h["loss"], "train_acc": h["accuracy"],
                         "test_loss": h["val_loss"], "test_acc": h["val_accuracy"]})
    hist.to_csv(os.path.join(args.out, f"{args.tag}_seed{seed}_history.csv"), index=False)
    sel = hist.loc[hist["train_loss"].idxmin()]          # paper rule
    best = hist.loc[hist["test_acc"].idxmax()]           # diagnostic only
    return dict(seed=seed, params=model.count_params(), epochs=args.epochs,
                selected_epoch_min_train_loss=int(sel["epoch"]),
                min_train_loss=sel["train_loss"],
                selected_test_acc=sel["test_acc"],
                final_epoch_test_acc=hist["test_acc"].iloc[-1],
                max_test_acc_DIAGNOSTIC_ONLY=best["test_acc"],
                max_test_acc_epoch_DIAGNOSTIC_ONLY=int(best["epoch"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/UCR_TS_Archive_2015")
    ap.add_argument("--dataset", default="Adiac")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--tag", default="O5")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = load_dataset(args.data_root, args.dataset)
    print(f"{args.dataset}: train {data[0].shape}, test {data[2].shape}, classes {data[4]}")
    rows = []
    for s in args.seeds:
        r = run_seed(s, args, data)
        rows.append(r)
        print(f"seed {s} | epoch {r['selected_epoch_min_train_loss']} | "
              f"selected test {r['selected_test_acc']:.4f} | final {r['final_epoch_test_acc']:.4f} | "
              f"diagnostic max {r['max_test_acc_DIAGNOSTIC_ONLY']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"{args.tag}_summary.csv"), index=False)
    print(f"\n{args.tag}: mean selected test acc {df.selected_test_acc.mean():.4f} "
          f"(std {df.selected_test_acc.std():.4f}), "
          f"mean final-epoch test acc {df.final_epoch_test_acc.mean():.4f}")


if __name__ == "__main__":
    main()
