#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:10:17 2020

@author: sirehna
"""

# Train on GPU PPO2 on ShipNavigation.

import os
import gym
import shipNavEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import numpy as np

NUM_TIMESTEPS = int(1e9)
env_kwargs = {'n_rocks':30,'n_rocks_obs':5}

env0 = make_vec_env(lambda **kwargs: gym.make('ShipNav-v0',**{'n_rocks':2,'n_rocks_obs':1}), env_kwargs=None, n_envs=1)
# env = make_vec_env(lambda **kwargs: gym.make('ShipNav-v0',**env_kwargs), env_kwargs=None, n_envs=4)
env = gym.make('ShipNav-v0', n_envs=1,**env_kwargs)

LOGDIR = "ppo2_ShipNav_retrain_multipleObstacles_2" # moved to zoo afterwards.
LOADDIR= "../zoo/ppo2_ShipNav_retrain" # moved to zoo afterwards.

# take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
model_legacy = PPO2.load(os.path.join(LOADDIR, "final_model"),env0, verbose=0,tensorboard_log=LOGDIR)
# model_legacy = PPO2.load(os.path.join(LOADDIR, "final_model"))
model = PPO2(MlpPolicy, env, gamma=0.99, n_steps=512, ent_coef=0.01, learning_rate=0.025, vf_coef=0.5,
             max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
             verbose=1, tensorboard_log=LOGDIR, _init_setup_model=True, policy_kwargs=None,
             full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)

# toto = np.clip(np.random.randn(64,6).astype('float32'),-1.0,1.0)
# pred0 = model.predict(toto)
# pred_leg = model_legacy.predict(toto)
model_params = model.get_parameters()
model_legacy_params = model_legacy.get_parameters()
    
for key in model_params.keys():
    s = model_legacy_params[key].shape
    # model_params[key][:] = 0.0
    if len(s)==1:
        model_params[key][0:s[0]] = model_legacy_params[key][0:s[0]]
    elif len(s) == 2:
        model_params[key][0:s[0],0:s[1]] = model_legacy_params[key][0:s[0],0:s[1]]

        
model.load_parameters(model_params)
# pred1 = model.predict(toto)
try:
    model.learn(total_timesteps=NUM_TIMESTEPS)
except KeyboardInterrupt:
    model.save(os.path.join(LOGDIR, "partial_model"))

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    
env.close()
