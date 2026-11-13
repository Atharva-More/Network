# Reproduction notes

## Setup
- MacBook Air (CPU only), Python 3.12.10, TensorFlow 2.21.0, UCR 2015 archive
- original_code/ = untouched as published; patched_original/ = minimal fixes to run on modern Keras; src/ = my own clean implementation

## Problems running the published code (reproducibility findings)
1. Data paths are relative and hardcoded (e.g. Adiac/Adiac_TRAIN); scripts only run from inside their own folder.
2. Dataset name is hardcoded in each script (one dataset per edit).
3. Old Keras API: `nb_epoch` -> `epochs`, `val_acc` -> `val_accuracy`, `idxmin` needs `()`.
4. Adadelta default learning rate changed between Keras 1 (1.0) and modern Keras (0.001). With the modern default the MLP did not learn (loss ~2.7, val acc ~0.22 on Adiac).
5. No requirements file, no random seeds, no multi-dataset runner.
6. Convolution layers were written as Conv2D(n, k, 1, ...): in modern Keras the third argument is the stride, not a kernel width (to check).

## Details missing from the paper
- Number of epochs (code uses 1500 for ResNet), batch size
- Learning-rate schedule: ResNet log ends at lr = 1e-4 although the paper states Adam lr = 0.001
- Shortcut projection for mismatched channels (Eq. 3 says y = h3 + x, impossible when channels differ)
- Filter counts: text says {64,128,128}, Figure 1 shows 64/64/64, 128x3, 128x3
- Number of runs, seeds, and how "lowest training loss" is implemented
- "val" set in the code is the test set

## Results so far
| Model | Dataset | Epochs | Seed | Paper error | My error | Notes |
|---|---|---|---|---|---|---|
| MLP | Adiac | ____ | - | 0.248 | ____ | lr = ____ |
| ResNet | Adiac | 1500 | 1 run | 0.174 | 0.1765 | train acc 1.0, test loss ~0.89, ~5 s/epoch (~2 h) |
| FCN | Adiac | ____ | ____ | 0.143 | ____ | |

## Observations
- ResNet: 0.1765 vs 0.174 is one test series (1/391 = 0.0026), within noise.
- Last-epoch test accuracy fluctuates between 0.816 and 0.831, so epoch choice alone moves error ~0.7 pp.
- ResNet clearly overfits (train acc 1.0, train loss ~1e-4).

## Label rescaling check on all 85 UCR datasets (scripts/check_labels_all.py)
- Only 50words is affected: 124/450 train and 121/455 test labels differ, 50 classes -> 44 effective classes.
- No other dataset is affected, so the bug touches 1 of the 44 datasets in the paper's Table I.

## Exact per-seed Adiac results (ResNet, scripts/run_resnet.py)
| Seed | 300 epochs | 1500 epochs |
|---|---|---|
| 0 | 0.2455 | - |
| 1 | 0.2430 | 0.2506 |
| 2 | 0.1816 | 0.1893 |
- Same-seed 1500 vs 300 epochs: +0.0076 and +0.0077 (3 test series each): no benefit from 5x longer training (141/117 min vs 26/26 min).
- Results fall in two groups (~0.18-0.19 and ~0.24-0.25); the seed matters more than the epoch count. More seeds needed to tell whether this is a real two-mode pattern.
- Earlier unseeded 1500-epoch run: 0.1765 (better group), as is the paper's 0.174.

## ResNet Adiac, 5 seeds at 300 epochs
- Mean 0.2153, std 0.0282 (paper 0.174, one run): the paper's value is ~1.5 std below our mean. Consistent with a favourable single draw, but a small systematic gap cannot be excluded with 5 seeds.

## 50words label-bug comparison (300 epochs, seed 0)
- Published labels (44 effective classes): 0.3231 | corrected labels (50 classes): 0.3143 | paper: 0.273
- Difference 0.0088 = 4 test series; within seed noise, no measurable effect from one paired run.
- Both variants are ~4-5 points above the paper, like Adiac.

## ResNet on 10 small datasets, 300 epochs, seed 0
- Within 1-2 test series of the paper: Coffee, SonyAIBORobotSurface, TwoLeadECG, Trace, Gun_Point, FaceFour; ItalyPowerDemand 4 series better.
- Large gaps: Beef 0.533 vs 0.233 (9/30 series), OliveOil 0.333 vs 0.133 (6/30), ECGFiveDays 0.287 vs 0.045 (208/861).
- Not explained by training-set size alone: TwoLeadECG has the same 23 training series / 2 classes as ECGFiveDays and matches the paper.
- Hypothesis (untested): Conv2D(f, k, 1) is a k x k kernel in Keras 3 (k x 1 in Keras 1), so Glorot initialisation differs.

## k x 1 vs k x k kernels (ResNet, 300 epochs, 3 seeds, scripts/run_resnet.py --kernel1d)
| Dataset | baseline mean±std (min/run) | kernel1d mean±std (min/run) | paper |
|---|---|---|---|
| Beef | 0.589±0.051 (4.9) | 0.567±0.058 (1.9) | 0.233 |
| OliveOil | 0.200±0.116 (6.6) | 0.189±0.069 (2.2) | 0.133 |
| ECGFiveDays | 0.173±0.104 (19.9) | 0.114±0.066 (5.4) | 0.045 |
- kernel1d is 2.6-3.7x faster; no detectable accuracy difference with 3 seeds.
- Initialisation hypothesis (k x k vs k x 1) does not explain the Beef gap.
- OliveOil is consistent with the paper (seeds 1-2 average ~0.133); seed spread (std 0.07-0.12) shows single-run numbers are unreliable on 30-series test sets.
- Beef: all 3 seeds >= 0.53 vs 0.233 in the paper: unexplained. Untested candidates: Adam epsilon (paper 1e-8 vs Keras default), BatchNorm with batch size 3.

## Improvement 1: true 1-D kernels (k x 1 instead of k x k)
- Parameters: 2,866,149 -> 527,845 (5.4x fewer); benchmark 2.67 -> 0.68 s/epoch (3.9x); real Adiac runs 23.9 -> 7.0 min (3.4x).
- Adiac, 5 seeds, 300 epochs: baseline 0.2153 +- 0.0282, 1-D 0.1995 +- 0.0363 (Welch t ~ 0.8: no detectable difference). Paired by seed 0-2: -0.046, +0.008, -0.026.
- Beef, OliveOil, ECGFiveDays (3 seeds): no detectable difference; 2.6-3.7x faster.
- Paper's 0.174 lies inside the 1-D spread (2 of 5 seeds at or below 0.1765).
