import os
import numpy as np

root = "data/UCR_TS_Archive_2015"
for ds in sorted(os.listdir(root)):
    f_tr = os.path.join(root, ds, ds + "_TRAIN")
    f_te = os.path.join(root, ds, ds + "_TEST")
    if not (os.path.exists(f_tr) and os.path.exists(f_te)):
        continue
    tr = np.loadtxt(f_tr, delimiter=",", usecols=0)
    te = np.loadtxt(f_te, delimiter=",", usecols=0)
    k = len(np.unique(te))                      # the original code takes nb_classes from test labels
    classes = np.unique(np.concatenate([tr, te]))
    for name, y in [("train", tr), ("test", te)]:
        used = ((y - y.min()) / (y.max() - y.min()) * (k - 1)).astype("int64")  # to_categorical truncates
        true = np.searchsorted(classes, y)
        bad = int((used != true).sum())
        if bad:
            print(f"{ds:35s} {name}: {bad}/{len(y)} labels differ | "
                  f"classes {len(np.unique(true))} -> {len(np.unique(used))}")
print("done")