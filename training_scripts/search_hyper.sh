#!/bin/sh

env_id=0
[ $# -gt 0 ] && env_id="$1"
n_rocks=50
[ $# -gt 1 ] && n_rocks="$2"
algo="a2c"
[ $# -gt 2 ] && algo="ppo"

cd ..; python train.py --algo $algo --env ShipNav-v$env_id --env-kwargs n_rocks:$n_rocks n_lidars:10 rock_scale:3 n_rocks_obs:20 --hyperparams frame_stack:10 -optimize --n-trials 100 -n 100000 # 10_000_000
