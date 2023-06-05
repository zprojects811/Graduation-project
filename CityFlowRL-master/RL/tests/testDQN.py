import os

# noinspection PyUnresolvedReferences
import CityFlowRL
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import replay


models_dir = "../models/"
env_kwargs = {'config': "hangzhou_1x1_bc-tyc_18041608_1h", 'steps_per_episode': 121, 'steps_per_action': 30}


def train():
    env = make_vec_env('CityFlowRL-v0', n_envs=1, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    # model = DQN('MlpPolicy', env, verbose=2, tensorboard_log="../tensorboard/")
    model = DQN.load(os.path.join(models_dir, "DQN_120000_steps.zip"), env=env)
    model.learn(total_timesteps=1000000, reset_num_timesteps=False,
                callback=CheckpointCallback(save_freq=10000, save_path="../models/", name_prefix="DQN", verbose=2))


def test():
    env = gym.make('CityFlowRL-v0', **env_kwargs)

    env = DummyVecEnv([lambda: env])

    model = DQN.load(os.path.join(models_dir, "DQN_120000_steps.zip"), env=env)

    episodes = 1
    for ep in range(episodes):
        obs = env.reset()
        rewards = []
        done = False
        while not done:
            action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
        print("Episode reward: ", sum(rewards))
        print(info)
    env.close()
    # replay.run(env_kwargs['config'])


if __name__ == "__main__":
    train()
