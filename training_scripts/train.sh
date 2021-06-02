#!/bin/sh

env=1
[ $# -gt 0 ] && env="$1"
n_rocks=0
[ $# -gt 1 ] && n_rocks=$2
algo="a2c"
[ $# -gt 2 ] && algo="$3"

cd ..; python train.py --algo $algo --env ShipNav-v$env --env-kwargs n_rocks:$n_rocks n_lidars:10 n_rocks_obs:20 -n 100_000_000 -tb tb #--hyperparams frame_stack:10
