import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env4 import SliceCreationEnv4
from gym_examples.envs.slice_management_env1 import SliceManagementEnv1

from os import rename

env1 = SliceCreationEnv4()
env2 = SliceManagementEnv1()


model1 = DQN.load("dqn_slices4(Arch:16; learn:1e-3; starts:250k; fraction:0_5; train: 1.5M).zip", env1)
model2 = DQN.load("dqn_maintain1.zip", env2)

obs1, info1 = env1.reset()
obs2, info2 = env2.reset()

cont = 0

'''
for i in range(500):
    while cont<99:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print('Action: ', action,'Observation: ', obs, ' | Reward: ', reward, ' | Terminated: ', terminated)
        cont += 1
        if terminated or truncated:
            obs, info = env.reset()
    #cont = 0
    # Comment after training of Model 2
    #rename('Global_Parameters.db','Global_Parameters{}.db'.format(str(i+1)))
    #obs, info = env.reset()
    '''

while cont<99:
    action1, _states1 = model1.predict(obs1, deterministic=True)
    action2, _states2 = model2.predict(obs2, deterministic=True)

    obs1, reward1, terminated1, truncated1, info1 = env1.step(action1)
    obs2, reward2, terminated2, truncated2, info2 = env2.step(action2)

    print("Model 1: ",'Action: ', action1,'Observation: ', obs1, ' | Reward: ', reward1, ' | Terminated: ', terminated1)
    print("Model 2: ",'Action: ', action2,'Observation: ', obs2, ' | Reward: ', reward2, ' | Terminated: ', terminated2)

    cont += 1
    if terminated1 or truncated1:
        obs1, info1 = env1.reset()
    
    if terminated2 or truncated2:
        obs2, info2 = env2.reset()