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
from stable_baselines.common.callbacks import CheckpointCallback



NUM_TIMESTEPS = 100000000

env = make_vec_env('ShipNav-v0', n_envs=1)

LOGDIR = "ppo2_ShipNav_" # moved to zoo afterwards.

# take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log=LOGDIR)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=LOGDIR,
                                         name_prefix='rl_model')

try:
    model.learn(total_timesteps=NUM_TIMESTEPS,callback=checkpoint_callback)
except KeyboardInterrupt:
    model.save(os.path.join(LOGDIR, "partial_model"))

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode=0)
    
env.close()
