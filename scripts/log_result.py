"""Append one result row to results/reproduction_log.csv.
Usage: python scripts/log_result.py MODEL DATASET EPOCHS SEED PAPER_ERR MY_ERR [NOTE]
"""
import csv, os, sys
from datetime import datetime

if len(sys.argv) < 7:
    sys.exit(__doc__)

model, dataset, epochs, seed, paper, mine = sys.argv[1:7]
note = sys.argv[7] if len(sys.argv) > 7 else ""
path = "results/reproduction_log.csv"
new = not os.path.exists(path)

with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["time", "model", "dataset", "epochs", "seed",
                    "paper_error", "my_error", "diff", "note"])
    diff = round(float(mine) - float(paper), 4)
    w.writerow([datetime.now().isoformat(timespec="seconds"), model, dataset,
                epochs, seed, paper, mine, diff, note])
print("logged:", model, dataset, "paper", paper, "mine", mine, "diff", diff)
