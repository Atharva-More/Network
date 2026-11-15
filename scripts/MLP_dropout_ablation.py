
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "optimization/results/phase2"
OUT.mkdir(parents=True, exist_ok=True)

data = np.loadtxt(
    ROOT / "Adiac/Adiac_TRAIN",
    delimiter=","
)

X = data[:, 1:].astype(np.float32)
_, y = np.unique(data[:, 0], return_inverse=True)

configs = {
    "D0_no_dropout": {
        "dropout": (0.0, 0.0, 0.0),
        "l2": 0.0
    },
    "D1_light_dropout": {
        "dropout": (0.05, 0.10, 0.15),
        "l2": 0.0
    },
    "D2_previous_dropout": {
        "dropout": (0.10, 0.20, 0.30),
        "l2": 0.0
    },
    "D3_l2_only": {
        "dropout": (0.0, 0.0, 0.0),
        "l2": 0.0001
    }
}


def build_model(dropout, l2_strength):
    regularizer = (
        tf.keras.regularizers.l2(l2_strength)
        if l2_strength > 0 else None
    )

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(176,)),

        tf.keras.layers.Dropout(dropout[0]),

        tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizer
        ),
        tf.keras.layers.Dropout(dropout[1]),

        tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizer
        ),
        tf.keras.layers.Dropout(dropout[2]),

        tf.keras.layers.Dense(
            37,
            activation="softmax"
        )
    ])

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

# Classical reference, recalculated on four folds
for fold, (tr, va) in enumerate(folds, start=1):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X[tr], y[tr])

    records.append({
        "model": "1NN_reference",
        "fold": fold,
        "best_epoch": 0,
        "train_accuracy": np.nan,
        "val_accuracy": clf.score(X[va], y[va]),
        "val_loss": np.nan
    })


for name, config in configs.items():

    print("\nStarting:", name, flush=True)

    for fold, (tr, va) in enumerate(folds, start=1):

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(813306 + fold)

        mean = X[tr].mean()
        std = X[tr].std()

        X_train = (X[tr] - mean) / std
        X_val = (X[va] - mean) / std

        model = build_model(
            config["dropout"],
            config["l2"]
        )

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

        best_index = int(hist["val_loss"].idxmin())
        best_epoch = best_index + 1

        val_loss, val_accuracy = model.evaluate(
            X_val,
            y[va],
            verbose=0
        )

        train_accuracy = float(
            hist.loc[best_index, "accuracy"]
        )

        hist.to_csv(
            OUT / f"{name}_fold{fold}.csv",
            index=False
        )

        records.append({
            "model": name,
            "fold": fold,
            "best_epoch": best_epoch,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss
        })

        print(
            f"Fold {fold} | "
            f"Epoch {best_epoch} | "
            f"Train acc {train_accuracy:.4f} | "
            f"Val acc {val_accuracy:.4f} | "
            f"Val loss {val_loss:.4f}",
            flush=True
        )


results = pd.DataFrame(records)

results.to_csv(
    OUT / "phase2_all_results.csv",
    index=False
)

summary = (
    results.groupby("model")["val_accuracy"]
    .agg(["mean", "std", "min", "max"])
    .sort_values("mean", ascending=False)
)

summary.to_csv(
    OUT / "phase2_summary.csv"
)

print("\nPHASE 2 RESULTS")
print(summary.to_string())

print("\nOfficial test set was not evaluated.")
