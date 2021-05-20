#!/bin/sh

env_id=0
[ $# -gt 0 ] && env_id="$1"
PRE_TRAINED_ID=11
[ $# -gt 1 ] && PRE_TRAINED_ID="$2"

cd ..; python train.py --algo a2c --env ShipNav-v$env_id --env-kwargs n_rocks:100 n_lidars:10 rock_scale:3 --hyperparams frame_stack:10 -n 10 -i logs/a2c/ShipNav-v${env_id}_$PRE_TRAINED_ID/ShipNav-v$env_id.zip
