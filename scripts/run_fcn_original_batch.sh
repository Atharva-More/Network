#!/bin/bash
for d in Beef OliveOil FaceFour ECGFiveDays; do
  for s in 0 1 2; do python scripts/run_fcn_original.py $d --seed $s; done
done
for s in 0 1 2; do python scripts/run_fcn_original.py Adiac --seed $s; done
for s in 0 1 2; do python scripts/run_fcn_original.py Adiac --seed $s --kernel1d; done
