#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:10:17 2020

@author: sirehna
"""

# Train on GPU PPO2 on ShipNavigation.

import os
import gym_ShipNavigation

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

NUM_TIMESTEPS = int(1e9)
env_kwargs = {'n_rocks':30,'n_rocks_obs':1}
env = make_vec_env('ShipNav-v1', env_kwargs=env_kwargs, n_envs=2)

LOGDIR = "ppo2_ShipNav_retrain_multipleObstacles" # moved to zoo afterwards.
LOADDIR= "../zoo/ppo2_ShipNav_retrain" # moved to zoo afterwards.

# take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
model = PPO2.load(os.path.join(LOADDIR, "final_model"),env, verbose=2,tensorboard_log=LOGDIR)

try:
    model.learn(total_timesteps=NUM_TIMESTEPS)
except KeyboardInterrupt:
    model.save(os.path.join(LOGDIR, "partial_model"))

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode=0)
    
env.close()
