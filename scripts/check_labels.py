import os
import numpy as np

root = "data/UCR_TS_Archive_2015"
for ds in sorted(os.listdir(root)):
    p = os.path.join(root, ds, ds + "_TEST")
    if not os.path.exists(p):
        continue
    y = np.loadtxt(p, delimiter=",", usecols=0)
    k = len(np.unique(y))
    z = (y - y.min()) / (y.max() - y.min()) * (k - 1)
    bad = int((z.astype(int) != np.round(z).astype(int)).sum())
    if bad:
        print(ds, "-> truncation would mislabel", bad, "test series")
print("done")