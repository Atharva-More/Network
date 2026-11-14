import argparse, csv, os, tempfile, time
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

ROOT = "data/UCR_TS_Archive_2015"


def load(name):
    tr = np.loadtxt(f"{ROOT}/{name}/{name}_TRAIN", delimiter=",")
    te = np.loadtxt(f"{ROOT}/{name}/{name}_TEST", delimiter=",")
    classes = np.unique(np.concatenate([tr[:, 0], te[:, 0]]))
    return (tr[:, 1:], np.searchsorted(classes, tr[:, 0]),
            te[:, 1:], np.searchsorted(classes, te[:, 0]), len(classes))


def se_block(x, ratio=8):
    f = int(x.shape[-1])
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Reshape((1, f))(s)
    s = layers.Dense(f // ratio, activation="relu", kernel_initializer="he_normal", use_bias=False)(s)
    s = layers.Dense(f, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(s)
    return layers.Multiply()([x, s])


def build(length, n_classes, se):
    inp = layers.Input((length, 1))
    x = inp
    for f, k in [(128, 8), (256, 5), (128, 3)]:       # paper's FCN: 128/256/128 filters, kernels 8/5/3
        x = layers.Conv1D(f, k, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if se:
            x = se_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(inp, layers.Dense(n_classes, activation="softmax")(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--se", action="store_true")
    ap.add_argument("--ls", type=float, default=0.0, help="label smoothing")
    ap.add_argument("--out", default="results/fcn_runs.csv")
    a = ap.parse_args()

    keras.utils.set_random_seed(a.seed)
    Xtr, ytr, Xte, yte, k = load(a.dataset)
    mu, sd = Xtr.mean(), Xtr.std()                     # train statistics only (paper)
    Xtr = ((Xtr - mu) / sd)[..., None].astype("float32")
    Xte = ((Xte - mu) / sd)[..., None].astype("float32")
    Ytr = keras.utils.to_categorical(ytr, k)
    bs = int(max(1, min(len(Xtr) // 10, 16)))

    model = build(Xtr.shape[1], k, a.se)
    model.compile(keras.optimizers.Adam(),
                  keras.losses.CategoricalCrossentropy(label_smoothing=a.ls),
                  metrics=["accuracy"])

    ckpt = os.path.join(tempfile.mkdtemp(), "best.weights.h5")
    cbs = [keras.callbacks.ModelCheckpoint(ckpt, monitor="loss", save_best_only=True,
                                           save_weights_only=True),     # paper: lowest TRAINING loss
           keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=1e-4)]

    t0 = time.time()
    hist = model.fit(Xtr, Ytr, batch_size=bs, epochs=a.epochs, callbacks=cbs, verbose=0)  # no test data here
    secs = round(time.time() - t0)

    def test_err():
        return float(1 - (model.predict(Xte, verbose=0).argmax(1) == yte).mean())

    err_last = test_err()
    model.load_weights(ckpt)
    err_sel = test_err()
    row = dict(dataset=a.dataset, seed=a.seed, se=a.se, ls=a.ls,
               err_trainloss_epoch=round(err_sel, 4), err_last=round(err_last, 4),
               best_epoch=int(np.argmin(hist.history["loss"])) + 1,
               params=model.count_params(), seconds=secs)
    print(row)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
