#!/bin/sh

cd ..

base_id=11
version=0
[ $# -eq 1 ] && version=$1
first_id=16
increment=-1
cur_nb_rocks=1
nb_rocks_max=35
nb_obs=25
n_lidars=10
frame_stack=10
rock_scale=3

while [ $cur_nb_rocks -le $nb_rocks_max ]; do
    cur_id=$base_id
    if [ $increment -ne -1 ]; then
        cur_id=$((first_id+increment))
    fi
    python train.py --algo a2c --env ShipNav-v$version --env-kwargs n_rocks:$cur_nb_rocks n_lidars:$n_lidars n_rocks_obs:$nb_obs rock_scale:$rock_scale --hyperparams frame_stack:$frame_stack --num-threads 3 -i logs/a2c/ShipNav-v${version}_${cur_id}/ShipNav-v$version.zip
    cur_nb_rocks=$((cur_nb_rocks+1))
    increment=$((increment+1))
done
