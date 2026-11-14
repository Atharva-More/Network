#!/usr/bin/env bash
# One command per MLP result. Run from the repository root with the UCR 2015 archive in
# data/UCR_TS_Archive_2015 (folder per dataset, e.g. Adiac/Adiac_TRAIN).
set -e
python src/mlp_adiac.py --epochs 186  --tag A1   # A1: 186 epochs, last epoch      (~minutes)
python src/mlp_adiac.py --epochs 5000 --tag O5   # O5: 5000 epochs, min train loss (~hours on CPU)
python scripts/summarize_mlp.py                  # tables, statistics and Fig. (needs results/logs/*)
