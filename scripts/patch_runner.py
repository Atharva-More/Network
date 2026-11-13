import pandas as pd

p = "scripts/run_resnet.py"
s = open(p).read()

def rep(old, new):
    global s
    assert s.count(old) == 1, f"expected 1 match for: {old[:50]}"
    s = s.replace(old, new)

rep('p.add_argument("--fix-labels", action="store_true")\n',
    'p.add_argument("--fix-labels", action="store_true")\n'
    'p.add_argument("--kernel1d", action="store_true")\n')

rep('tmp = os.path.abspath("scripts/_run_resnet_tmp.py")',
    'if a.kernel1d:   # k x k kernels -> k x 1 kernels, as in Keras 1\n'
    '    src, n = re.subn(r"Conv2D\\(([^,]+), (8|5|3), 1, ?padding", r"Conv2D(\\1, (\\2, 1), 1, padding", src)\n'
    '    assert n == 9, f"expected 9 conv layers, matched {n}"\n'
    'tmp = os.path.abspath("scripts/_run_resnet_tmp.py")')

rep("""{'_fixed' if a.fix_labels else ''}\"""",
    """{'_fixed' if a.fix_labels else ''}{'_k1d' if a.kernel1d else ''}\"""")

rep('w.writerow(["dataset", "epochs", "seed", "fixed_labels", "test_error", "paper_error", "seconds"])',
    'w.writerow(["dataset", "epochs", "seed", "fixed_labels", "kernel1d", "test_error", "paper_error", "seconds"])')

rep('w.writerow([a.dataset, a.epochs, a.seed, a.fix_labels, err, paper, secs])',
    'w.writerow([a.dataset, a.epochs, a.seed, a.fix_labels, a.kernel1d, err, paper, secs])')
open(p, "w").write(s)

# add the new column to the existing results file
df = pd.read_csv("results/resnet_runs.csv")
if "kernel1d" not in df.columns:
    df.insert(4, "kernel1d", False)
    df.to_csv("results/resnet_runs.csv", index=False)

# summary script: group by the new column too
q = "scripts/summarize_resnet.py"
t = open(q).read()
assert t.count('["dataset", "epochs", "fixed_labels"]') == 1
open(q, "w").write(t.replace('["dataset", "epochs", "fixed_labels"]',
                             '["dataset", "epochs", "fixed_labels", "kernel1d"]'))
print("patched")
