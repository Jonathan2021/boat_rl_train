
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:29:11 2020

@author: gfo
"""

import gym
import shipNavEnv
import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

env = gym.make("ShipNav-v0",n_rocks=3,n_rocks_obs = 1,obs_radius = 300)

# Load the trained agent

SAVED_MODEL_DIR = "../tools/export_model_tf"

model = tf.saved_model.load_v2(SAVED_MODEL_DIR)
infer = model.signatures["serving_default"]

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
        action1 = np.argmax(infer(tf.constant([obs[:6]]))['action_policy'],axis=1)
        action2 = np.argmax(infer(tf.constant([obs[:6]]))['action_policy_proba'],axis=1)
        action3 = infer(tf.constant([obs[:6]]))['action_deterministic'].numpy()[0]
        
        action = action3 #or action1, or action2, they should be the very same
        
        obs, rewards, done, info = env.step(action)
        reward+=rewards
        env.render()
    print("trial %d reward = %f"%(i,reward))