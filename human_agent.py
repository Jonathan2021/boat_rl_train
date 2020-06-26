#%% imports
import gym.spaces
import time
import gym_ShipNavigation
import math

#%% reset env
try:
    myEnv.reset()
except NameError:
    # myEnv = gym.make('gym_ShipNavigation:ShipNavigation-v0')
   # myEnv = gym.make('gym_ShipNavigation:ShipNavigationLidar-v0')
    #myEnv = gym.make('gym_ShipNavigation:ShipNav-v0')
    myEnv = gym.make('gym_ShipNavigation:ShipNav-v1')
    #myEnv = gym.make('gym_ShipNavigation:ShipNav-v2')
    #myEnv = gym.make('gym_ShipNavigation:ShipNav-v4')
    
    myEnv.reset()

#%% other
PRINT_DEBUG_MSG = True

#%%
if not hasattr(myEnv.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = myEnv.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 2
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a < 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a < 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 2

myEnv.render()
myEnv.unwrapped.viewer.window.on_key_press = key_press
myEnv.unwrapped.viewer.window.on_key_release = key_release

def rollout(myEnv):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = myEnv.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = myEnv.step(a)
        #print("reward %0.3f distance %0.3f bearing %0.3f" % (r, obser[4]*1600, obser[5]*180))
        #print(obser)
        total_reward += r
        window_still_open = myEnv.render()
        if window_still_open==False: 
            myEnv.viewer.close()
            return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            myEnv.render()
            time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 4")

while 1:
    window_still_open = rollout(myEnv)
    if window_still_open==False: 
        print('break')
        break


