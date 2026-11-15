
"""O5: 128-unit MLP, 5000 epochs, original repository's minimum-training-loss rule.

The official test set is monitored every epoch for comparability with the
published repository, but never determines the selected epoch. All max-test
accuracy figures are exploratory diagnostics, not paper-style results.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "optimization" / "results" / "official_test" / "O5_5000"
OUT.mkdir(parents=True, exist_ok=True)

EPOCHS = 5000
BATCH_SIZE = 16
LR = 0.001
SEEDS = [int(s.strip()) for s in os.environ.get(
    "UCR_SEEDS", "813306,813307,813308,813309,813310"
).split(",") if s.strip()]

train = np.loadtxt(ROOT / "Adiac" / "Adiac_TRAIN", delimiter=",")
test = np.loadtxt(ROOT / "Adiac" / "Adiac_TEST", delimiter=",")

X_train = train[:, 1:].astype(np.float32)
X_test = test[:, 1:].astype(np.float32)
classes, y_train = np.unique(train[:, 0], return_inverse=True)
assert np.all(np.isin(test[:, 0], classes))
y_test = np.searchsorted(classes, test[:, 0])
assert X_train.shape == (390, 176)
assert X_test.shape == (391, 176)
assert len(classes) == 37

# Identical preprocessing to our 186-epoch A1 evaluation.
train_mean = X_train.mean()
train_std = X_train.std()
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std


class BestTrainingLossWeights(tf.keras.callbacks.Callback):
    """Remember end-of-epoch weights whenever epoch-average train loss improves."""

    def __init__(self):
        super().__init__()
        self.best_loss = float("inf")
        self.best_epoch = None
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        loss = float((logs or {})["loss"])
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch + 1
            self.best_weights = [w.copy() for w in self.model.get_weights()]


class Progress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % 250 == 0:
            logs = logs or {}
            print(
                "  Epoch %4d | train_loss %.4f | train_acc %.4f | "
                "test_acc %.4f" % (
                    epoch + 1,
                    float(logs["loss"]),
                    float(logs["accuracy"]),
                    float(logs["val_accuracy"]),
                ),
                flush=True,
            )


results = []
print("TensorFlow version:", tf.__version__, flush=True)
print("Epochs:", EPOCHS, "| Seeds:", SEEDS, flush=True)
print("Train/test:", X_train.shape, X_test.shape, flush=True)

for seed in SEEDS:
    print("\nStarting seed", seed, flush=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(176,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(37, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    tracker = BestTrainingLossWeights()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        verbose=0,
        callbacks=[tracker, Progress()],
    )

    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(OUT / f"seed_{seed}_history.csv", index=False)

    # PAPER-STYLE RESULT: select epoch by MINIMUM TRAINING LOSS only.
    chosen = int(hist["loss"].idxmin())
    chosen_epoch = int(hist.loc[chosen, "epoch"])
    assert tracker.best_epoch == chosen_epoch
    chosen_test_acc = float(hist.loc[chosen, "val_accuracy"])

    # Exploratory diagnostic: highest test accuracy over all epochs. NEVER
    # substitute this for the selected paper-style result.
    diagnostic = int(hist["val_accuracy"].idxmax())

    # Restore and retain the minimum-training-loss checkpoint.
    model.set_weights(tracker.best_weights)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    model.save(str(OUT / f"seed_{seed}_best_train_loss.keras"))

    results.append({
        "seed": seed,
        "epochs_ran": len(hist),
        "selected_epoch_min_train_loss": chosen_epoch,
        "min_epoch_training_loss": float(hist.loc[chosen, "loss"]),
        "selected_test_accuracy_history": chosen_test_acc,
        "selected_test_accuracy_reevaluated": float(test_acc),
        "selected_test_loss_reevaluated": float(test_loss),
        "selected_train_accuracy_reevaluated": float(train_acc),
        "max_test_accuracy_DIAGNOSTIC_ONLY": float(hist.loc[diagnostic, "val_accuracy"]),
        "max_test_accuracy_epoch_DIAGNOSTIC_ONLY": int(hist.loc[diagnostic, "epoch"]),
        "final_epoch_test_accuracy": float(hist.iloc[-1]["val_accuracy"]),
    })

    pd.DataFrame(results).to_csv(OUT / "O5_summary.csv", index=False)
    print(
        "Seed %d | paper-style epoch %d | selected test %.2f%% | "
        "diagnostic max test %.2f%% at epoch %d" % (
            seed, chosen_epoch, chosen_test_acc * 100,
            float(hist.loc[diagnostic, "val_accuracy"]) * 100,
            int(hist.loc[diagnostic, "epoch"]),
        ),
        flush=True,
    )

summary = pd.DataFrame(results)
print("\nO5 PAPER-STYLE RESULTS (lowest TRAINING loss)")
print(summary[["seed", "selected_epoch_min_train_loss",
               "selected_test_accuracy_history",
               "max_test_accuracy_DIAGNOSTIC_ONLY"]].to_string(index=False))
print("Mean selected test accuracy: %.2f%%" %
      (100 * summary["selected_test_accuracy_history"].mean()))
print("Median selected test accuracy: %.2f%%" %
      (100 * summary["selected_test_accuracy_history"].median()))
print("Published Adiac MLP test accuracy: 75.20%")
print("186-epoch A1 mean test accuracy: 59.34%")
print("All per-epoch histories and selected checkpoint models saved in:", OUT)
