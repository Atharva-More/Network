import numpy as np
from tensorflow import keras

root = "data/UCR_TS_Archive_2015/50words"
tr = np.loadtxt(f"{root}/50words_TRAIN", delimiter=",", usecols=0)
te = np.loadtxt(f"{root}/50words_TEST", delimiter=",", usecols=0)
k = len(np.unique(te))  # the original code takes nb_classes from the test labels
all_labels = np.unique(np.concatenate([tr, te]))

def original_rescale(y):  # the exact formula from ResNet.py
    return (y - y.min()) / (y.max() - y.min()) * (k - 1)

for name, y in [("train", tr), ("test", te)]:
    used = keras.utils.to_categorical(original_rescale(y), k).argmax(1)  # what the code really trains on
    true = np.searchsorted(all_labels, y)                                # the correct label
    wrong = used != true
    print(f"{name}: {len(y)} series | true classes: {len(np.unique(true))}"
          f" | classes after original code: {len(np.unique(used))}"
          f" | wrong labels: {int(wrong.sum())}")
    if name == "test":
        print("(true class -> class actually used):",
              sorted(set(zip(true[wrong].tolist(), used[wrong].tolist())))[:15])