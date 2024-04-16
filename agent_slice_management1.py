from stable_baselines3 import DQN

#import tensorflow as tf

from stable_baselines3.common.logger import configure

from gym_examples.envs.slice_management_env1 import SliceManagementEnv1

#from os.path import exists

from gymnasium.wrappers import TimeLimit

env = SliceManagementEnv1()
env = TimeLimit(env, max_episode_steps=20)

log_path = "/home/mario/Documents/DQN_Models/Model 2/gym-examples1/logs"
#log_path = "/data/scripts/DQN_models/Model 1/logs/"     #For pod
new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

policy_kwargs = dict(net_arch=[4])

model = DQN("MlpPolicy",env, 
        buffer_size=int(1e4),  # Replay buffer size
        learning_rate=1e-3,     # Learning rate
        learning_starts=10000,  # Number of steps before learning starts
        exploration_fraction=0.5,  # Fraction of total timesteps for exploration
        exploration_final_eps=0,  # Final exploration probability after exploration_fraction * total_timesteps
        train_freq=4,           # Update the model every `train_freq` steps
        gradient_steps=1,       # Number of gradient steps to take after each batch of data
        batch_size=32,          # Number of samples in each batch
        gamma=0.99,             # Discount factor
        tau=1.0,                # Target network update rate
        target_update_interval=1000,  # Interval (in timesteps) at which the target network is updated
        verbose=1,              # Verbosity level
        policy_kwargs=policy_kwargs)              

#model = DQN.load("dqn_slices1", env)
#model = DQN("MlpPolicy", env, verbose=1, exploration_final_eps=0, exploration_fraction=0.5)
model.set_logger(new_logger)
model.learn(total_timesteps=50000, log_interval=100)
model.save("/home/mario/Documents/DQN_Models/Model 2/gym-examples1/dqn_maintain1")