#!/bin/sh

env_id=1
[ $# -gt 0 ] && env_id="$1"
n_rocks=50
[ $# -gt 1 ] && n_rocks="$2"
algo="a2c"
[ $# -gt 2 ] && algo=$3

cd ..; python train.py --algo $algo --env ShipNav-v$env_id --env-kwargs n_rocks:$n_rocks n_lidars:10 n_rocks_obs:20 -optimize --n-trials 20 -n 3_000_000 # --hyperparams frame_stack:10
