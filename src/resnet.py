import time
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def conv(x, filters, k, mode):
    if mode == "2d":   # published layout: Conv2D(f, k, 1) -> k x k kernel
        return layers.Conv2D(filters, k, 1, padding="same")(x)
    return layers.Conv1D(filters, k, padding="same")(x)


def conv_bn_relu(x, filters, k, mode):
    x = layers.BatchNormalization()(conv(x, filters, k, mode))
    return layers.Activation("relu")(x)


def res_block(x, filters, mode):
    h = conv_bn_relu(x, filters, 8, mode)
    h = conv_bn_relu(h, filters, 5, mode)
    h = layers.BatchNormalization()(conv(h, filters, 3, mode))
    s = layers.BatchNormalization()(conv(x, filters, 1, mode))
    return layers.Activation("relu")(layers.Add()([s, h]))


def build_resnet(length, n_classes, mode="1d", n=64):
    shape = (length, 1, 1) if mode == "2d" else (length, 1)
    inp = layers.Input(shape)
    x = res_block(inp, n, mode)
    x = res_block(x, n * 2, mode)
    x = res_block(x, n * 2, mode)
    pool = layers.GlobalAveragePooling2D if mode == "2d" else layers.GlobalAveragePooling1D
    return keras.Model(inp, layers.Dense(n_classes, activation="softmax")(pool()(x)))


if __name__ == "__main__":
    length, n_classes, n = 176, 37, 390   # Adiac
    X = np.random.randn(n, length, 1, 1).astype("float32")
    y = np.random.randint(0, n_classes, n)
    for mode in ["2d", "1d"]:
        m = build_resnet(length, n_classes, mode)
        Xm = X if mode == "2d" else X[..., 0]
        m.compile("adam", "sparse_categorical_crossentropy")
        m.fit(Xm, y, batch_size=16, epochs=1, verbose=0)
        t = time.time()
        m.fit(Xm, y, batch_size=16, epochs=3, verbose=0)
        print(mode, "params:", m.count_params(), "sec/epoch:", round((time.time() - t) / 3, 2))
