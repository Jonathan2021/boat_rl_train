#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:23:21 2020

@author: gfo
"""

# Train single CPU PPO1 on ShipNavigation.

import os
import gym
import gym_ShipNavigation

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(1e7)
SEED = 4242
EVAL_FREQ = 250000
EVAL_EPISODES = 1000
LOGDIR = "ppo1_v4_4" # moved to zoo afterwards.

logger.configure(folder=LOGDIR)

env = gym.make("ShipNav-v4")
env.seed(SEED)

# take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=512, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
                 optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)
model.load_parameters("ppo1_v4_4/best_model.zip")
eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

try:
    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)
except KeyboardInterrupt:
    model.save(os.path.join(LOGDIR, "partial_model"))

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

env.close()
