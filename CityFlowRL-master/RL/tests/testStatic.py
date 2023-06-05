# noinspection PyUnresolvedReferences
import CityFlowRL
import random
import gym
from itertools import cycle
import replay

env_kwargs = {'config': "hangzhou_1x1_bc-tyc_18041608_1h", 'steps_per_episode': 121, 'steps_per_action': 30}
env = gym.make('CityFlowRL-v0', **env_kwargs)
env.set_replay_path('staticReplay.txt')


def test(config=None):
    if config is not None:
        env_kwargs['config'] = config

    # Check action space
    n = 0
    actions = []
    while env.action_space.contains(n):
        actions.append(n)
        n += 1
    env.reset()

    iter_cycle = cycle(actions)

    # iterate environment a little bit to test env

    rewards = []
    done = False
    while not done:
        action = next(iter_cycle)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
    print("Episode reward: ", sum(rewards))
    print(info)
    env.close()
    # replay.run(env_kwargs['config'])
    return info['avg_travel_time']


if __name__ == "__main__":
    test()
