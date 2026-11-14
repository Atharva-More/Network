#!/bin/bash
run_all() {   # dataset seed
  python src/fcn_variants.py $1 --seed $2
  python src/fcn_variants.py $1 --seed $2 --se
  python src/fcn_variants.py $1 --seed $2 --ls 0.1
  python src/fcn_variants.py $1 --seed $2 --se --ls 0.1
}
for s in 0 1 2 3 4; do run_all Adiac $s; done
for d in SwedishLeaf FaceFour ECGFiveDays Beef OliveOil Gun_Point; do
  for s in 0 1 2; do run_all $d $s; done
done
