# Reproducing "Time Series Classification from Scratch with Deep Neural Networks" (Wang et al., 2016)

Reproduction and improvement study of the MLP / FCN / ResNet baselines on the UCR 2015 archive.
Original code: https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline (kept untouched in `original_code/`).

## Repository layout
- `original_code/` the authors' files, unchanged
- `patched_original/` minimal fixes so they run on modern Keras (diffs in `patches/`)
- `scripts/` runners, checks, summaries
- `src/` own implementations (`resnet.py`, `fcn_variants.py`, `mlp_adiac.py`, ...)
- `results/` raw results (`resnet_runs.csv`), tables, figure, logs
- `docs/README_MLP.md` MLP-specific instructions
- `report/` IEEE report source (`main.tex`) and figures
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

## Reproducing the FCN results
```bash
bash scripts/run_fcn_ablation.sh          # base / +SE / +LS / +SE+LS, 300 epochs
bash scripts/run_fcn_ablation_2000.sh     # 2000-epoch runs
python scripts/summarize_fcn.py           # tables from results/fcn_runs.csv
python scripts/run_fcn_original.py ...    # authors' FCN script, patched (see script header)
```
Rows are appended to `results/fcn_runs.csv` (dataset, seed, se, ls, epochs, errors, params, seconds).

## Reproducing the MLP results
```bash
bash scripts/run_mlp.sh                   # A1 (186 ep) and O5 (5000 ep), then the summary
python scripts/summarize_mlp.py           # table, statistics and figure from the saved logs
```
See `docs/README_MLP.md`.

## Results
First reproduction run (authors' settings, Adiac, 1500 epochs, one unseeded run): test error 0.1765 vs 0.174 in the paper. Seeded follow-ups show the spread across seeds (Adiac 0.156-0.251), so this agreement is partly luck.

See `results/resnet_paper_vs_ours.md` (paper vs ours) and `results/resnet_figure.png`.

### Paper vs ours on Adiac
| Model | Paper (error) | Published code in our hands | Ours / improved |
|---|---|---|---|
| MLP | 0.248 | 0.644 ± 0.013 (2017 Keras, 5 seeds); 0.775 (TF 2.x) | 0.265 ± 0.023 (128-unit MLP, 5 seeds) |
| FCN | 0.143 | 0.379 (300 ep) / 0.207 (2000 ep) | SE / label smoothing: no gain; 1-D kernels 4.2x fewer params |
| ResNet | 0.174 | 0.215 ± 0.028 (300 ep, 5 seeds) | 0.200 ± 0.036 (1-D kernels, 5.4x fewer params) |

### MLP (test accuracy %, Adiac)
| Setting | Runs | Epochs | Accuracy |
|---|---|---|---|
| Paper, 3x500 MLP, Adadelta | 1 | 5000 | 75.20 |
| Published code, TF 2.x (Adadelta default lr 0.001) | 1 | 5000 | 22.51 |
| Published code, 2017 Keras (min train loss) | 5 | 5000 | 35.65 ± 1.33 |
| Ours A1: 128 units, Adam, last epoch | 5 | 186 | 59.34 ± 2.93 |
| Ours O5: 128 units, Adam, min train loss | 5 | 5000 | 73.45 ± 2.29 |

### FCN (test error, mean ± std)
| Dataset | Paper | 300 epochs | 2000 epochs |
|---|---|---|---|
| Adiac | 0.143 | 0.379 ± 0.044 | 0.207 ± 0.049 |
| Beef | 0.250 | 0.444 ± 0.019 | 0.456 ± 0.051 |
| ECGFiveDays | 0.015 | 0.160 ± 0.044 | 0.214 ± 0.100 |
| FaceFour | 0.068 | 0.246 ± 0.093 | 0.295 ± 0.099 |
| GunPoint | 0.000 | 0.011 ± 0.019 | - |
| OliveOil | 0.167 | 0.767 ± 0.145 | 0.211 ± 0.051 |
| SwedishLeaf | 0.034 | 0.036 ± 0.004 | - |

## Findings
See `notes.md`. Highlights: modern-Keras incompatibilities, undocumented hyperparameters,
a label-rescaling bug on 50words, large seed variance (Adiac: 0.156-0.251), and a k×1-kernel ResNet
with 5.4× fewer parameters and ~3.4× faster training at no detectable accuracy cost.

## Authors / contributions
- Atharva More: environment and repo, FCN runs and SE/label-smoothing ablation, ResNet reproduction, label-bug scan, 1-D kernel analysis, tables and figures
- Payal KR, Vansh: MLP reproduction and compact-MLP experiments (TODO: confirm exact split)

Repository: https://github.com/Atharva-More/Network

In README:
### Squeeze-and-Excitation FCN (SE-FCN)
To run the 2,000-epoch SE-FCN training script and evaluate its performance against the 1D baseline on the Adiac dataset:
```bash
python src/se_fcn.py

Training logs are stored in logs/ and the comparative training dynamics figures (comparing SE-FCN and Baseline FCN over 2,000 epochs) are saved in results/se_fcn_vs_baseline_fcn_training_final.png
