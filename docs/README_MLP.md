# MLP part: reproduction and improvement (Adiac)

Paper: Wang, Yan & Oates (2016). Paper value for Adiac MLP: error 0.248 = **75.2 % accuracy**.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install tensorflow==2.16.2 numpy pandas scipy matplotlib
# UCR 2015 archive -> data/UCR_TS_Archive_2015/Adiac/Adiac_TRAIN, Adiac_TEST (...)
```

## Reproduce each result
| Result | Command | Output |
|---|---|---|
| Table/figure/statistics from saved logs | `python scripts/summarize_mlp.py` | `results/mlp_paper_vs_ours.{csv,md}`, `results/mlp_stats.txt`, `results/mlp_figure.png` |
| A1: compact MLP, 186 epochs, 5 seeds | `python src/mlp_adiac.py --epochs 186 --tag A1` | `results/A1_summary.csv` |
| O5: compact MLP, 5000 epochs, min-train-loss selection | `python src/mlp_adiac.py --epochs 5000 --tag O5` | `results/O5_summary.csv`, per-epoch histories |
| Everything | `bash scripts/run_mlp.sh` | |

Published-code runs (not re-runnable with the new TF): `results/logs/MLP_original_tf2_default.log`
(authors' script on TF 2.x, 22.51 %) and `results/logs/keras2017_seed_813306..813310.log`
(authors' script, 2017 Keras/TF1 environment, 5 seeds, 35.65 ± 1.33 %).

## Paper vs ours (Adiac test accuracy, %)
| Setting | Runs | Epochs | Accuracy |
|---|---|---|---|
| Paper, 3x500 MLP, Adadelta | 1 | 5000 | 75.20 |
| Published code, TF 2.x (Adadelta default lr 0.001) | 1 | 5000 | 22.51 |
| Published code, 2017 Keras (min train loss) | 5 | 5000 | 35.65 ± 1.33 |
| Ours A1: 128 units, Adam, last epoch | 5 | 186 | 59.34 ± 2.93 |
| Ours O5: 128 units, Adam, min train loss | 5 | 5000 | 73.45 ± 2.29 |

Seeds 813306-813310. Model: 176-128-37, ReLU, Adam 1e-3, batch 16, no dropout/L2, 27,429 parameters
(original: 608,037). The test set is never used for selection; best-test-epoch columns are labelled
`DIAGNOSTIC_ONLY`.
