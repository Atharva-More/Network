# Reproducing "Time Series Classification from Scratch with Deep Neural Networks" (Wang et al., 2016)

Reproduction and improvement study of the MLP / FCN / ResNet baselines on the UCR 2015 archive.
Original code: https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline (kept untouched in `original_code/`).

## Repository layout
- `original_code/` the authors' files, unchanged
- `patched_original/` minimal fixes so they run on modern Keras (diffs in `patches/`)
- `scripts/` runners, checks, summaries
- `src/` own implementations
- `results/` raw results (`resnet_runs.csv`), tables, figure, logs
- `notes.md` running log of every finding

## Setup (tested: macOS, CPU only, Python 3.12, TensorFlow 2.21)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Download the UCR 2015 archive from https://www.cs.ucr.edu/~eamonn/time_series_data/ (password on that page)
and unzip it so that `data/UCR_TS_Archive_2015/Adiac/Adiac_TRAIN` exists. Data is not committed.

## Reproducing the ResNet results
```bash
# one run: dataset, epochs, seed; add --kernel1d for k×1 kernels, --fix-labels for corrected labels
python scripts/run_resnet.py Adiac --epochs 300 --seed 0
python scripts/run_resnet.py Adiac --epochs 300 --seed 0 --kernel1d

# label-rescaling bug (affects 50words only)
python scripts/check_labels_all.py
python scripts/run_resnet.py 50words --epochs 300 --seed 0
python scripts/run_resnet.py 50words --epochs 300 --seed 0 --fix-labels

# parameter count / speed benchmark
python src/resnet.py

# tables and figure
python scripts/summarize_resnet.py
python scripts/make_tables.py
python scripts/make_figure.py
```
Every run appends a row to `results/resnet_runs.csv`; the log of each run is in `results/logs/`.

## Results
First reproduction run (authors' settings, Adiac, 1500 epochs, one unseeded run): test error 0.1765 vs 0.174 in the paper. Seeded follow-ups show the spread across seeds (Adiac 0.156-0.251), so this agreement is partly luck.

See `results/resnet_paper_vs_ours.md` (paper vs ours) and `results/resnet_figure.png`.

## Findings
See `notes.md`. Highlights: modern-Keras incompatibilities, undocumented hyperparameters,
a label-rescaling bug on 50words, large seed variance (Adiac: 0.156-0.251), and a k×1-kernel ResNet
with 5.4× fewer parameters and ~3.4× faster training at no detectable accuracy cost.

## Authors / contributions
(fill in per member)
