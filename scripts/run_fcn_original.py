import argparse, csv, os, re, subprocess, sys, time

# Paper (Table I) FCN test errors, keyed by UCR 2015 folder names
PAPER = {"Adiac": .143, "Beef": .25, "Coffee": 0, "ECGFiveDays": .015, "FaceFour": .068,
         "Gun_Point": 0, "OliveOil": .167, "SwedishLeaf": .034, "50words": .321,
         "Trace": 0, "TwoLeadECG": 0, "ItalyPowerDemand": .03,
         "SonyAIBORobotSurface": .032, "CBF": 0, "wafer": .003, "yoga": .155}

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("--epochs", type=int, default=2000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--fix-labels", action="store_true")
p.add_argument("--kernel1d", action="store_true")
a = p.parse_args()

def sub1(pat, rep, text):
    new, n = re.subn(pat, lambda m: rep, text, flags=re.M)
    assert n == 1, f"pattern {pat!r} matched {n} times"
    return new

src = open("patched_original/FCN.py").read()
src = sub1(r"^nb_epochs\s*=\s*\d+", f"nb_epochs = {a.epochs}", src)
src = sub1(r"^flist\s*=\s*\['Adiac'\]", f"flist = ['{a.dataset}']", src)
src = sub1(r"^import pandas as pd", f"import pandas as pd\nkeras.utils.set_random_seed({a.seed})", src)
src = sub1(r"^\s*batch_size = min\(x_train\.shape\[0\]/10, 16\)",
           "    batch_size = int(max(1, min(x_train.shape[0]/10, 16)))", src)
if a.fix_labels:
    src = sub1(r"^\s*y_train = \(y_train - .*$",
               "    classes = np.unique(np.concatenate([y_train, y_test]))\n"
               "    y_train = np.searchsorted(classes, y_train)", src)
    src = sub1(r"^\s*y_test = \(y_test - .*$", "    y_test = np.searchsorted(classes, y_test)", src)
if a.kernel1d:   # k x k kernels -> k x 1 kernels, as in Keras 1
    src, n = re.subn(r"Conv2D\((\d+), (8|5|3), 1, ?padding", r"Conv2D(\1, (\2, 1), 1, padding", src)
    assert n == 3, f"expected 3 conv layers, matched {n}"

tmp = os.path.abspath("scripts/_run_fcn_tmp.py")
open(tmp, "w").write(src)
tag = f"{a.dataset}_e{a.epochs}_s{a.seed}{'_fixed' if a.fix_labels else ''}{'_k1d' if a.kernel1d else ''}"
os.makedirs("results/logs", exist_ok=True)
log = os.path.abspath(f"results/logs/fcn_{tag}.log")

t0 = time.time()
with open(log, "w") as f:   # run inside the UCR folder so 'Dataset/Dataset_TRAIN' resolves
    subprocess.run([sys.executable, tmp], cwd="data/UCR_TS_Archive_2015",
                   stdout=f, stderr=subprocess.STDOUT)
secs = round(time.time() - t0)

text = open(log).read()
lines = [l for l in text.splitlines() if l.strip()]
try:
    last = lines[-1].split()
    loss, acc = float(last[0]), float(last[1])
    err_last = round(1 - float(re.findall(r"val_accuracy: ([0-9.]+)", text)[-1]), 4)
except Exception:
    sys.exit(f"Run failed. See the end of {log}")
err = round(1 - acc, 4)
paper = PAPER.get(a.dataset, "")
print(f"{tag}: error {err} at lowest train loss, {err_last} at last epoch (paper {paper}), {secs}s")

path = "results/fcn_original_runs.csv"
new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["dataset", "epochs", "seed", "fixed_labels", "kernel1d",
                    "err_trainloss_epoch", "err_last", "paper_error", "seconds"])
    w.writerow([a.dataset, a.epochs, a.seed, a.fix_labels, a.kernel1d, err, err_last, paper, secs])
