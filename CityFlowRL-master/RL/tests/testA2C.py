import os

# noinspection PyUnresolvedReferences
import CityFlowRL
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import replay

models_dir = "../models/"
env_kwargs = {'config': "hangzhou_1x1_bc-tyc_18041608_1h", 'steps_per_episode': 121, 'steps_per_action': 30}


def train():
    env = make_vec_env('CityFlowRL-v0', n_envs=12, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    # model = A2C('MlpPolicy', env, verbose=2, tensorboard_log="../tensorboard/")
    model = A2C.load(os.path.join(models_dir, "A2C_1320000_steps.zip"), env=env)
    model.learn(total_timesteps=120000 * 6, reset_num_timesteps=False,
                callback=CheckpointCallback(save_freq=10000, save_path="../models/", name_prefix="A2C", verbose=2))


def test(config=None):
    if config is not None:
        env_kwargs['config'] = config

    env = gym.make('CityFlowRL-v0', **env_kwargs)
    env.set_replay_path('a2cReplay.txt')


    env = DummyVecEnv([lambda: env])

    model = A2C.load(os.path.join(models_dir, "A2C_1320000_steps.zip"))

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
    return info[0]['avg_travel_time']


if __name__ == "__main__":
    test()
