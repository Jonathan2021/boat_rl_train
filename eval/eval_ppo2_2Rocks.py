
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

env = gym.make("ShipNav-v1",n_rocks=30,n_rocks_obs = 1)

# Load the trained agent
model = PPO2.load("../zoo/ppo2_ShipNav_retrain/final_model.zip")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent

done = False
i = 0
reward =0
for i in range(10):
    obs = env.reset()
    j = 0
    done = False
    reward = 0
    while not (done or j >= 500):
        j+=1
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        reward+=rewards
        env.render()
    print("trial %d reward = %f"%(i,reward))