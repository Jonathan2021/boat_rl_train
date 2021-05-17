#!/bin/sh

base_id=139
first_id=141
increment=-1
cur_nb_rocks=1
nb_rocks_max=35
nb_obs=25
frame_stack=20

while [ $cur_nb_rocks -le $nb_rocks_max ]; do
    cur_id=$base_id
    if [ $increment -ne -1 ]; then
        cur_id=$((first_id+increment))
    fi
    python train.py --algo a2c --env ShipNav-v0 --env-kwargs n_rocks:$cur_nb_rocks n_rocks_obs:$nb_obs --hyperparams frame_stack:$frame_stack --num-threads 3 -i logs/a2c/ShipNav-v0_$cur_id/ShipNav-v0.zip
    cur_nb_rocks=$((cur_nb_rocks+1))
    increment=$((increment+1))
done
