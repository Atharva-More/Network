# Reproducibility report

Paper: Wang, Yan & Oates (2016), *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline*
(MLP / FCN / ResNet on the UCR 2015 archive). Code: https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
(kept untouched in `original_code/`). This file follows the four reproduction guidelines. Raw numbers are in `results/`,
the running log in `notes.md`, dataset details in `datasets/README.md`.

Conventions: errors are test errors at the epoch with the lowest training loss (the paper's rule). "Series" = test series
(one test series is 1/n of the error). Mean ± standard deviation over seeds; n = number of seeds.

## 1. Published code on the published datasets vs the paper

All three scripts needed patches to run on current Keras (`patched_original/`, diffs in `patches/`). We ran them on
Adiac (the main dataset) and on small datasets (CPU only).

| Model | Paper (Adiac error) | Published code in our hands | Run details |
|---|---|---|---|
| MLP | 0.248 | 0.775 (TF 2.x, Adadelta default lr 0.001); 0.644 ± 0.013 (2017 Keras/TF1, n=5) | 5000 epochs |
| FCN | 0.143 | 0.379 ± 0.044 (300 epochs, n=5); 0.207 ± 0.049 (2000 epochs, n=3) | FCN rebuilt with 1-D kernels |
| ResNet | 0.174 | 0.215 ± 0.028 (300 epochs, n=5); first unseeded 1500-epoch run 0.1765 | |

Other datasets: `results/resnet_paper_vs_ours.md` (ResNet, 11 datasets), the FCN table in `README.md` (7 datasets).
The MLP was also rebuilt as a compact 128-unit network (73.45 ± 2.29 % accuracy, paper 75.2 %); that is an improvement,
not a reproduction (`docs/README_MLP.md`).

## 2. Discrepancies: within expected variance, or systematic?

Rule we used: a gap is **within variance** if the paper's value lies inside the range of our seeds (or within 2 test series
for a single seed); it is **systematic** if every one of our seeds is on the same side of the paper's value by more than
that. With 3 to 5 seeds this is a screening rule, not a significance test.

| Case | Paper | Ours | Verdict |
|---|---|---|---|
| ResNet, Adiac (n=5, 300 ep) | 0.174 | 0.215 ± 0.028, range 0.18 to 0.25; one seed 0.156 with 1-D kernels | Within variance; mean is 16 series above the paper, a small systematic gap cannot be excluded |
| ResNet, first run (authors' settings) | 0.174 | 0.1765 | Same as the paper (1 series) |
| ResNet, Gun_Point, Trace, TwoLeadECG, SonyAIBO, FaceFour, ItalyPower, OliveOil | see tables | within 0 to 2 series (ItalyPower 4 series better) | Within variance |
| ResNet, Beef | 0.233 | 0.589 ± 0.051, all 3 seeds ≥ 0.53 | **Systematic**, unexplained |
| ResNet, ECGFiveDays | 0.045 | 0.173 ± 0.104 (110 series above) | **Systematic**, unexplained |
| ResNet, 50words | 0.273 | 0.323 (published labels), 0.314 (corrected) | Label bug changes 4 series: within noise; 4 to 5 points above the paper like Adiac |
| ResNet, 1500 vs 300 epochs (Adiac) | – | +0.008 for both seeds, 5× the time | No benefit; within variance |
| FCN, Adiac (2000 ep, n=3) | 0.143 | 0.207 ± 0.049 (25 series above) | Probably systematic, but only 3 seeds; one team 2000-epoch baseline run reached 0.159 |
| FCN, Beef, FaceFour, ECGFiveDays | 0.250 / 0.068 / 0.015 | 0.456 / 0.295 / 0.214 at 2000 epochs; no gain from 300 epochs | **Systematic** |
| FCN, OliveOil, Gun_Point, SwedishLeaf | 0.167 / 0.000 / 0.034 | 0.211 / 0.011 / 0.036 | Within variance |
| FCN at 300 epochs | – | far from the paper on Adiac and OliveOil, close at 2000 | Undertrained: a training-length effect, not a bug |
| MLP (2 environments) | 0.248 | 0.775 and 0.644 ± 0.013 | **Systematic**: no seed comes near; even the best test epoch averages 48 % accuracy |
| SE blocks / label smoothing on FCN (300 ep) | – | no measurable change; SE+LS worse on 7/7 datasets (p = 0.016) | Not an effect we can claim (baseline undertrained) |

**Size of expected variance.** Seed standard deviation is 0.028 on Adiac for ResNet (11 series), up to 0.10 to 0.15 on
30-series datasets (OliveOil, Beef) and 0.10 for FCN on ECGFiveDays. The paper reports one run per dataset without spread,
so differences of one to two standard deviations (e.g. ResNet on Adiac) cannot be called errors in the paper.

## 3. Missing details, unreported choices and preprocessing that matter

| Item | Paper | Code / our finding | Effect |
|---|---|---|---|
| Epochs | not stated | 1500 (ResNet), 2000 (FCN), 5000 (MLP) | ResNet: 300 vs 1500 no difference; FCN: large (undertrained at 300) |
| Batch size | not stated | `min(n/10, 16)` | Untested; batch 2 to 3 on tiny sets may matter for BatchNorm |
| Learning-rate schedule | Adam lr 0.001 | ReduceLROnPlateau (factor 0.5, patience 50, min 1e-4); ResNet log ends at 1e-4 | Untested |
| Number of runs / seeds | not stated | code seeds only NumPy; TF not seeded | Seed spread ±0.028 on Adiac (ResNet) |
| Model selection | "lowest training loss" | test set passed as `validation_data`; with long training the rule picks almost the last epoch | Test metrics logged every epoch invite selection on test |
| Shortcut | `y = h3 + x` (impossible when channels differ) | 1×1 conv + BN projection in all three blocks | Untested |
| Filter counts | text 64/128/128 | Figure 1 shows 64/64/64, 128×3, 128×3 | Code follows the text |
| Input BatchNorm | not mentioned | present in ResNet code | Untested |
| Adam epsilon | 1e-8 | current Keras default 1e-7 | Untested |
| Adadelta learning rate (MLP) | 0.1 | code calls `Adadelta()`: lr 1.0 in Keras 1/2, 0.001 in current Keras | **Large**: 22.5 % (0.001) vs 35.7 % (1.0) accuracy on Adiac |
| Kernel layout | k×1 kernels | `Conv2D(f, k, 1)` is k×k in current Keras | None on accuracy; FCN 4.2× and ResNet 5.4× more parameters, ResNet ~3.4× slower |
| Label rescaling | – | float truncation breaks 50words (50 → 44 classes) | 4 series on one run |
| API changes | – | `nb_epoch`, `val_acc`, `idxmin`, removed backend function in the CAM block | Code does not run unpatched |
| Dependencies | none given | no requirements file, hard-coded paths and dataset names | See setup below |
| MLP training accuracy | "almost all 100 %" | ~61 % at the selected epoch with the published settings | Published MLP underfits in our hands; cause not found |

## 4. Reproduction steps and how to verify them

**Environment (tested: macOS, CPU only, Python 3.12, TensorFlow 2.21).**
```bash
git clone https://github.com/Atharva-More/Network.git && cd Network
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The MLP baseline was also run in older environments (Python 3.7 / TF 1.15.5 and Python 2.7 / TF 1.0.1 / Keras 1.2.2,
Docker); the compact-MLP runs used Python 3.11, TF 2.16.2.

**Data (no download needed).**
```bash
python scripts/prepare_datasets.py --verify --root datasets/used    # shapes + SHA-256 checksums
mkdir -p data && ln -s ../datasets/used data/UCR_TS_Archive_2015      # where the scripts look
```
Details and the original source are in `datasets/README.md`.

**Run and compare (every command appends to `results/` or writes a table).**

| Result | Command | Compare with | Time (CPU) |
|---|---|---|---|
| ResNet, one run | `python scripts/run_resnet.py Adiac --epochs 300 --seed 0` | row in `results/resnet_runs.csv` (seed 0: 0.2455) | ~25 min |
| ResNet, 1-D kernels | `... --kernel1d` | 7.0 min, same accuracy range | ~7 min |
| Label bug | `python scripts/check_labels_all.py` (needs the full archive) | only 50words differs | seconds |
| Parameters and speed | `python src/resnet.py` | 2,866,149 vs 527,845 | seconds |
| ResNet tables, figure | `python scripts/summarize_resnet.py && python scripts/make_tables.py && python scripts/make_figure.py` | `results/resnet_paper_vs_ours.md` | seconds |
| FCN | `bash scripts/run_fcn_ablation.sh`, `bash scripts/run_fcn_ablation_2000.sh`, `python scripts/summarize_fcn.py` | FCN table in `README.md` | ~2.8 min per 300-epoch run, ~16 min per 2000-epoch run (Adiac) |
| MLP | `bash scripts/run_mlp.sh` (use `--data-root datasets/used` if the archive is not in `data/`) | `results/mlp_paper_vs_ours.md` | hours for the 5000-epoch runs |
| MLP from saved logs | `python scripts/summarize_mlp.py` | same tables, statistics and figure | seconds |

**How to judge whether you reproduced our numbers.** TensorFlow on CPU is not bit-for-bit repeatable even with seeds,
so compare against the spread, not a single value: ResNet Adiac at 300 epochs should fall between about 0.18 and 0.25
(mean 0.215), FCN Adiac at 2000 epochs near 0.21 ± 0.05, the compact MLP at 5000 epochs between about 70 % and 76 %
accuracy, and the published MLP (2017 environment) around 35 %. Five seeds should give a mean within roughly 0.03 of ours.

**Files that make the work checkable.** `original_code/` (untouched) vs `patched_original/` (minimal fixes) with diffs in
`patches/`; one raw row per run in `results/*_runs.csv`; per-run logs in `results/logs/`; checksums of the data in
`datasets/CHECKSUMS.sha256`; the seed lists (ResNet 0 to 4, FCN 0 to 4 / 0 to 2, MLP 813306 to 813310) in the scripts;
`notes.md` with every finding in the order we made it.

**Not covered.** Only 13 of the paper's 44 datasets were run; FCN used 3 seeds at 2000 epochs; explanations for the Beef,
ECGFiveDays and MLP gaps are untested (candidates: Adam epsilon, BatchNorm at tiny batch sizes, shortcut type, LR schedule,
unreported settings in the authors' MLP).
