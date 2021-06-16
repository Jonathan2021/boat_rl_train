#!/bin/sh

env_id=1
[ $# -gt 0 ] && env_id="$1"
n_rocks=50
[ $# -gt 1 ] && n_rocks="$2"
algo="a2c"
[ $# -gt 2 ] && algo=$3
n_jobs=3
[ $# -gt 3 ] && n_jobs=$4
n_ships=50
[ $# -gt 4 ] && n_rocks="$5"

cd ..; python train.py --algo $algo --env ShipNav-v$env_id --env-kwargs n_rocks:$n_rocks n_ships:$n_ships n_lidars:15 n_obstacles_obs:20 -optimize --n-trials 30 -n 1_000_000 --n-jobs $n_jobs --n-evaluation 40 # --hyperparams frame_stack:10
