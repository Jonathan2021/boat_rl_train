#!/bin/sh

env_id=1
[ $# -gt 0 ] && env_id="$1"
n_rocks=30
[ $# -gt 1 ] && n_rocks="$2"
algo="a2c"
[ $# -gt 2 ] && algo=$3
n_jobs=3
[ $# -gt 3 ] && n_jobs=$4
n_ships=50
[ $# -gt 4 ] && n_ships="$5"
n_obstacles_obs=0
[ $# -gt 5 ] && n_obstacles_obs="$6"

echo "python train.py --algo $algo --env ShipNav-v$env_id --env-kwargs n_rocks:$n_rocks n_ships:$n_ships n_lidars:15 n_obstacles_obs:$n_obstacles_obs --waypoints:False -optimize --n-trials 40 -n 400_000 --n-jobs $n_jobs --n-evaluation 40 # --hyperparams frame_stack:10"

cd ..; python train.py --algo $algo --env ShipNav-v$env_id --env-kwargs n_rocks:$n_rocks n_ships:$n_ships n_lidars:15 n_obstacles_obs:$n_obstacles_obs waypoints:False -optimize --n-trials 20 -n 4_000_000 --n-jobs $n_jobs --n-evaluation 40 # --hyperparams frame_stack:10
