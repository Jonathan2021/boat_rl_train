#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:29:11 2020

@author: gfo
"""

import gym
import gym_ShipNavigation
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import csv
import numpy as np

env = gym.make("ShipNav-v1",n_rocks=30,n_rocks_obs = 1,obs_radius = 300)


# Load the trained agent
model = PPO2.load("../zoo/ppo2_ShipNav_retrain/final_model.zip")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent


    
with open('policy_logs.csv', mode='w') as policy_logs:
    policy_logs_writer = csv.writer(policy_logs, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < 1:
        env.render()
        a, _states = model.predict(o)
        policy_logs_writer.writerow(np.append(np.append(o,a),np.argmax(a)))
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
    
        if d or (ep_len == 10000):
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1