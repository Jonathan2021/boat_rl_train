#!/bin/sh

env=1
[ $# -gt 0 ] && env="$1"
n_rocks=0
[ $# -gt 1 ] && n_rocks=$2
algo="a2c"
[ $# -gt 2 ] && algo="$3"
n_ships=0
[ $# -gt 3 ] && n_ships=$4
n_obstacles_obs=0
[ $# -gt 4 ] && n_obstacles_obs=$5

cd ..; python train.py --algo $algo --env ShipNav-v$env --env-kwargs n_ships:$n_ships n_rocks:$n_rocks n_lidars:15 n_obstacles_obs:$n_obstacles_obs waypoints:False -n 10_000_000 -tb tb #--hyperparams frame_stack:10
