
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

env = gym.make("ShipNav-v0",n_rocks=30,n_rocks_obs = 1,obs_radius = 300)

# Load the trained agent

SAVED_MODEL_DIR = "../tools/export_model_tf"
model = tf.saved_model.load_v2(SAVED_MODEL_DIR,tags=['serve'])
x = tf.compat.v1.placeholder(dtype='float32',shape=[None, 6],name='obs')

infer = model.signatures["serving_default"](x)
toto = tf.constant(np.ones((1,6),dtype='float32'))
infer1 = infer['action_policy']
infer2 = infer['action_policy_proba']
infer3 = infer['action_deterministic']

init=tf.global_variables_initializer()
done = False
i = 0
reward =0
sess = tf.compat.v1.Session()
sess.run(init)
infer1.eval(session=sess,feed_dict={x:np.random.randn(1,6).astype('float32')})

# (out1,out2,out3) = sess.run((action1,action2,action3))

# for i in range(10):
#     obs = env.reset()
#     j = 0
#     done = False
#     reward = 0
#     while not (done or j >= 500):
#         j+=1
#         tf.global_variables_initializer().run(session=sess)
        
    
#         # action = np.argmax(infer(tf.reshape(tf.constant(obs[:6]),(1,6)))['action'],axis=1)
#         # obs, rewards, done, info = env.step(action)
#         # reward+=rewards
#         # env.render()
#         pass
#     print("trial %d reward = %f"%(i,reward))