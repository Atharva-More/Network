
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "optimization" / "results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

data = np.loadtxt(
    ROOT / "Adiac" / "Adiac_TRAIN",
    delimiter=","
)

X = data[:, 1:].astype(np.float32)
_, y = np.unique(data[:, 0], return_inverse=True)

configs = {
    "A1_single_128": [128],
    "A2_128_64": [128, 64],
    "A3_256_128": [256, 128],
    "A4_500_500_500": [500, 500, 500]
}


def build_model(widths):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(X.shape[1],)))

    for units in widths:
        model.add(
            tf.keras.layers.Dense(
                units,
                activation="relu"
            )
        )

    model.add(
        tf.keras.layers.Dense(
            len(np.unique(y)),
            activation="softmax"
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=813306
)

folds = list(cv.split(X, y))
records = []

for fold, (tr, va) in enumerate(folds, 1):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X[tr], y[tr])

    records.append({
        "model": "1NN_reference",
        "fold": fold,
        "parameters": 0,
        "best_epoch": 0,
        "epochs_ran": 0,
        "train_accuracy": np.nan,
        "val_accuracy": knn.score(X[va], y[va]),
        "val_loss": np.nan
    })


for name, widths in configs.items():
    print("\nStarting:", name, flush=True)

    for fold, (tr, va) in enumerate(folds, 1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(813306 + fold)

        mean = X[tr].mean()
        std = X[tr].std()

        X_train = (X[tr] - mean) / std
        X_val = (X[va] - mean) / std

        model = build_model(widths)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=60,
            restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y[tr],
            validation_data=(X_val, y[va]),
            epochs=500,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )

        hist = pd.DataFrame(history.history)

        best_epoch = int(hist["val_loss"].idxmin()) + 1

        _, train_acc = model.evaluate(
            X_train, y[tr], verbose=0
        )

        val_loss, val_acc = model.evaluate(
            X_val, y[va], verbose=0
        )

        hist.to_csv(
            OUT / f"{name}_fold{fold}.csv",
            index=False
        )

        records.append({
            "model": name,
            "fold": fold,
            "parameters": model.count_params(),
            "best_epoch": best_epoch,
            "epochs_ran": len(hist),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_loss": val_loss
        })

        print(
            f"Fold {fold} | "
            f"Epoch {best_epoch} | "
            f"Train acc {train_acc:.4f} | "
            f"Val acc {val_acc:.4f} | "
            f"Val loss {val_loss:.4f}",
            flush=True
        )


results = pd.DataFrame(records)

results.to_csv(
    OUT / "phase3_all_results.csv",
    index=False
)

summary = (
    results.groupby("model")["val_accuracy"]
    .agg(["mean", "std", "min", "max"])
    .sort_values("mean", ascending=False)
)

summary.to_csv(
    OUT / "phase3_summary.csv"
)

print("\nPHASE 3 RESULTS")
print(summary.to_string())

print("\nOfficial test set was not evaluated.")
