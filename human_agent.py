#%% imports
import gym.spaces
import time
import gym_ShipNavigation
import math

#%% reset env
try:
    myEnv.reset()
except NameError:
    # myEnv = gym.make('gym_ShipNavigation:ShipNav-v0',n_rocks=0)
    myEnv = gym.make('gym_ShipNavigation:ShipNav-v1',n_rocks=10,n_rocks_obs=1,control_throttle=True)
    
    myEnv.reset()

#%% other
PRINT_DEBUG_MSG = True

#%%

if type(myEnv.action_space) == gym.spaces.discrete.Discrete:
    ACTIONS = myEnv.action_space.n
    human_agent_action = 2
elif type(myEnv.action_space) == gym.spaces.multi_discrete.MultiDiscrete:
    ACTIONS = myEnv.action_space.nvec
    human_agent_action = [2,2]
else:
    raise Exception('Keyboard agent only supports discrete or multi-discrete action spaces')

SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.


human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    action0 = 2
    action1 = 2
    if key == 65361:
        action0 = 1
    elif key == 65363:
        action0 = 0
    elif key == 65362:
        action1 = 0
    elif key == 65364:
        action1 = 1
    if type(human_agent_action) == int:
        human_agent_action = action0
    else:
        human_agent_action[0] = action0
        human_agent_action[1] = action1

def key_release(key, mod):
    global human_agent_action
    if key in [65361, 65363]:
        if type(human_agent_action) == int:
            human_agent_action = 2
        else:
            human_agent_action[0] = 2
    elif (key in [65362, 65364]) and type(human_agent_action) is not int:
        human_agent_action[0] = 2
        
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
            time.sleep(0.01)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 4")

while 1:
    window_still_open = rollout(myEnv)
    if window_still_open==False: 
        print('break')
        break


