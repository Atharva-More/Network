#!/bin/bash
for d in ECGFiveDays OliveOil Adiac FaceFour Beef; do
  for s in 0 1 2; do
    python src/fcn_variants.py $d --seed $s --epochs 2000 --se
    python src/fcn_variants.py $d --seed $s --epochs 2000 --ls 0.1
    python src/fcn_variants.py $d --seed $s --epochs 2000 --se --ls 0.1
  done
done
