#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:09:22 2020

@author: sirehna
"""

# Train on GPU PPO2 on ShipNavigation.

import os
import gym_ShipNavigation

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

NUM_TIMESTEPS = 100000000

env = make_vec_env('ShipNav-v4', n_envs=1)

LOGDIR = "ppo2_ShipNav" # moved to zoo afterwards.

# take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
model = PPO2(MlpPolicy, env, verbose=2,tensorboard_log=LOGDIR)

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
