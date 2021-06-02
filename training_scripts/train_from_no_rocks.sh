#!/bin/sh

env_id=1
[ $# -gt 0 ] && env_id="$1"
PRE_TRAINED_ID=74
[ $# -gt 1 ] && PRE_TRAINED_ID="$2"

cd ..; python train.py --algo a2c --env ShipNav-v$env_id --env-kwargs n_rocks:150 n_lidars:10 n_rocks_obs:20 -n 100_000_000 -i logs/a2c/ShipNav-v${env_id}_$PRE_TRAINED_ID/ShipNav-v$env_id.zip -tb tb #--hyperparams frame_stack:10
