
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "optimization/results/official_test"
OUT.mkdir(parents=True, exist_ok=True)

train = np.loadtxt(
    ROOT / "Adiac/Adiac_TRAIN",
    delimiter=","
)

test = np.loadtxt(
    ROOT / "Adiac/Adiac_TEST",
    delimiter=","
)

X_train = train[:, 1:].astype(np.float32)
X_test = test[:, 1:].astype(np.float32)

classes, y_train = np.unique(
    train[:, 0], return_inverse=True
)

assert np.all(np.isin(test[:, 0], classes))
y_test = np.searchsorted(classes, test[:, 0])

mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

assert X_train.shape == (390, 176)
assert X_test.shape == (391, 176)
assert len(classes) == 37

seeds = [813306, 813307, 813308, 813309, 813310]
epochs = 186
records = []

for seed in seeds:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(176,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(37, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=16,
        verbose=0
    )

    train_loss, train_acc = model.evaluate(
        X_train, y_train, verbose=0
    )

    test_loss, test_acc = model.evaluate(
        X_test, y_test, verbose=0
    )

    pd.DataFrame(history.history).to_csv(
        OUT / f"A1_seed_{seed}_history.csv",
        index=False
    )

    records.append({
        "seed": seed,
        "epochs": epochs,
        "train_accuracy": train_acc,
        "train_loss": train_loss,
        "test_accuracy": test_acc,
        "test_loss": test_loss
    })

    print(
        f"Seed {seed} | "
        f"Train {train_acc:.4f} | "
        f"Test {test_acc:.4f} | "
        f"Test loss {test_loss:.4f}",
        flush=True
    )

results = pd.DataFrame(records)
results.to_csv(
    OUT / "A1_official_test_results.csv",
    index=False
)

print("\nOFFICIAL ADIAC TEST RESULTS")
print(results.to_string(index=False))

print("\nMean test accuracy:",
      results["test_accuracy"].mean())

print("Median test accuracy:",
      results["test_accuracy"].median())

print("Std test accuracy:",
      results["test_accuracy"].std())

print("\nPublished reference accuracy: 0.752")
