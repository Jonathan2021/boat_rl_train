#!/bin/sh

n_rocks=50
[ $# -eq 1 ] && n_rocks="$1"

cd ..; python train.py --algo a2c --env ShipNav-v1 --env-kwargs n_rocks:$n_rocks n_lidars:10 rock_scale:3 --hyperparams frame_stack:10 -n 10
