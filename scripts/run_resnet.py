import argparse, csv, os, re, subprocess, sys, time

# Paper (Table I) ResNet test errors, keyed by the folder names in the UCR 2015 archive
PAPER = {
    "Adiac": .174, "Beef": .233, "CBF": .006, "ChlorineConcentration": .172,
    "CinC_ECG_torso": .229, "Coffee": 0, "Cricket_X": .179, "Cricket_Y": .195,
    "Cricket_Z": .187, "DiatomSizeReduction": .069, "ECGFiveDays": .045,
    "FaceAll": .166, "FaceFour": .068, "FacesUCR": .042, "50words": .273,
    "FISH": .011, "Gun_Point": .007, "Haptics": .495, "InlineSkate": .635,
    "ItalyPowerDemand": .04, "Lighting2": .246, "Lighting7": .164, "MALLAT": .021,
    "MedicalImages": .228, "MoteStrain": .105, "NonInvasiveFatalECG_Thorax1": .052,
    "NonInvasiveFatalECG_Thorax2": .049, "OliveOil": .133, "OSULeaf": .021,
    "SonyAIBORobotSurface": .015, "SonyAIBORobotSurfaceII": .038,
    "StarLightCurves": .029, "SwedishLeaf": .042, "Symbols": .128,
    "synthetic_control": 0, "Trace": 0, "TwoLeadECG": 0, "Two_Patterns": 0,
    "uWaveGestureLibrary_X": .213, "uWaveGestureLibrary_Y": .332,
    "uWaveGestureLibrary_Z": .245, "wafer": .003, "WordsSynonyms": .368, "yoga": .142,
}

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--fix-labels", action="store_true")
p.add_argument("--kernel1d", action="store_true")
a = p.parse_args()

def sub1(pat, rep, text):
    new, n = re.subn(pat, lambda m: rep, text, flags=re.M)
    assert n == 1, f"pattern {pat!r} matched {n} times"
    return new

src = open("patched_original/ResNet.py").read()
src = sub1(r"^nb_epochs\s*=\s*\d+", f"nb_epochs = {a.epochs}", src)
src = sub1(r"^flist\s*=\s*\['Adiac'\]", f"flist = ['{a.dataset}']", src)
src = sub1(r"^np\.random\.seed\(813306\)",
           f"np.random.seed(813306)\nkeras.utils.set_random_seed({a.seed})", src)
src = sub1(r"^\s*batch_size = min\(x_train\.shape\[0\]/10, 16\)",
           "    batch_size = int(max(1, min(x_train.shape[0]/10, 16)))", src)
if a.fix_labels:
    src = sub1(r"^\s*y_train = \(y_train - .*$",
               "    classes = np.unique(np.concatenate([y_train, y_test]))\n"
               "    y_train = np.searchsorted(classes, y_train)", src)
    src = sub1(r"^\s*y_test = \(y_test - .*$",
               "    y_test = np.searchsorted(classes, y_test)", src)

if a.kernel1d:   # k x k kernels -> k x 1 kernels, as in Keras 1
    src, n = re.subn(r"Conv2D\(([^,]+), (8|5|3), 1, ?padding", r"Conv2D(\1, (\2, 1), 1, padding", src)
    assert n == 9, f"expected 9 conv layers, matched {n}"
tmp = os.path.abspath("scripts/_run_resnet_tmp.py")
open(tmp, "w").write(src)
tag = f"{a.dataset}_e{a.epochs}_s{a.seed}{'_fixed' if a.fix_labels else ''}{'_k1d' if a.kernel1d else ''}"
os.makedirs("results/logs", exist_ok=True)
log = os.path.abspath(f"results/logs/resnet_{tag}.log")

t0 = time.time()
with open(log, "w") as f:   # run inside the UCR folder so 'Dataset/Dataset_TRAIN' resolves
    subprocess.run([sys.executable, tmp], cwd="data/UCR_TS_Archive_2015",
                   stdout=f, stderr=subprocess.STDOUT)
secs = round(time.time() - t0)

lines = [l for l in open(log).read().splitlines() if l.strip()]
try:
    last = lines[-1].split()
    loss, acc = float(last[0]), float(last[1])
except Exception:
    sys.exit(f"Run failed. See the end of {log}")
err = round(1 - acc, 4)
paper = PAPER.get(a.dataset, "")
print(f"{tag}: error {err} (paper {paper}), lowest train loss {loss:.2e}, {secs}s")

path = "results/resnet_runs.csv"
new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["dataset", "epochs", "seed", "fixed_labels", "kernel1d", "test_error", "paper_error", "seconds"])
    w.writerow([a.dataset, a.epochs, a.seed, a.fix_labels, a.kernel1d, err, paper, secs])
