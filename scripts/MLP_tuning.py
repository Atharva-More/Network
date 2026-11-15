
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "optimization" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

data = np.loadtxt(
    ROOT / "Adiac" / "Adiac_TRAIN",
    delimiter=","
)

X = data[:, 1:].astype("float32")
labels = data[:, 0]
classes, y = np.unique(labels, return_inverse=True)

assert X.shape == (390, 176)
assert len(classes) == 37

configs = {
    "O1_adadelta_500": ([500, 500, 500], "adadelta"),
    "O2_adam_500": ([500, 500, 500], "adam"),
    "O3_adam_256": ([256, 128], "adam"),
    "O4_adam_128": ([128, 64], "adam"),
}

def create_model(widths, optimizer_name):
    inputs = tf.keras.Input(shape=(X.shape[1],))
    x = tf.keras.layers.Dropout(0.1)(inputs)

    for i, units in enumerate(widths):
        x = tf.keras.layers.Dense(
            units,
            activation="relu"
        )(x)

        rate = 0.3 if i == len(widths) - 1 else 0.2
        x = tf.keras.layers.Dropout(rate)(x)

    outputs = tf.keras.layers.Dense(
        len(classes),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    if optimizer_name == "adadelta":
        optimizer = tf.keras.optimizers.Adadelta(
            learning_rate=0.1,
            rho=0.95,
            epsilon=1e-8
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001
        )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=813306
)

folds = list(cv.split(X, y))
records = []

# Classical reference: 1-nearest-neighbor
for fold, (tr, va) in enumerate(folds, 1):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X[tr], y[tr])

    accuracy = clf.score(X[va], y[va])

    records.append({
        "model": "1NN_Euclidean",
        "fold": fold,
        "best_epoch": 0,
        "val_loss": np.nan,
        "val_accuracy": accuracy
    })

# MLP experiments
for name, (widths, optimizer_name) in configs.items():
    print("\nStarting:", name, flush=True)

    for fold, (tr, va) in enumerate(folds, 1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(813306 + fold)

        mean = X[tr].mean()
        std = X[tr].std()

        if std < 1e-8:
            raise ValueError("Invalid normalization standard deviation")

        X_train = (X[tr] - mean) / std
        X_val = (X[va] - mean) / std

        model = create_model(widths, optimizer_name)

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=40,
            restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y[tr],
            validation_data=(X_val, y[va]),
            batch_size=16,
            epochs=500,
            callbacks=[callback],
            verbose=0
        )

        history_df = pd.DataFrame(history.history)

        best_epoch = int(
            history_df["val_loss"].idxmin()
        ) + 1

        val_loss, val_accuracy = model.evaluate(
            X_val,
            y[va],
            verbose=0
        )

        history_df.to_csv(
            RESULTS / f"{name}_fold{fold}.csv",
            index=False
        )

        records.append({
            "model": name,
            "fold": fold,
            "best_epoch": best_epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Fold {fold} | "
            f"Best epoch: {best_epoch} | "
            f"Validation accuracy: {val_accuracy:.4f} | "
            f"Validation loss: {val_loss:.4f}",
            flush=True
        )

results = pd.DataFrame(records)

results.to_csv(
    RESULTS / "cv_all_results.csv",
    index=False
)

summary = (
    results.groupby("model")["val_accuracy"]
    .agg(["mean", "std", "min", "max"])
    .sort_values("mean", ascending=False)
)

summary.to_csv(RESULTS / "cv_summary.csv")

print("\nFINAL CROSS-VALIDATION SUMMARY")
print(summary.to_string())

print("\nHighest mean CV accuracy:", summary.index[0])
print("Official test set was not used.")
