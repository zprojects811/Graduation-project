# noinspection PyUnresolvedReferences
import CityFlowRL
import random
import gym

env_kwargs = {'config': "hangzhou_1x1_bc-tyc_18041608_1h", 'steps_per_episode': 121, 'steps_per_action': 30}
env = gym.make('CityFlowRL-v0', **env_kwargs)


def test():
    # Check action space
    actions = 0
    while env.action_space.contains(actions):
        actions += 1
    print(env.action_space)
    print(env.observation_space)
    env.reset()

    # iterate environment a little bit to test env

    # episodes = 1
    # for ep in range(episodes):
    #     obs = env.reset()
    #     rewards = []
    #     done = False
    #     while not done:
    #         action = random.randint(0, actions - 1)
    #         obs, reward, done, info = env.step(action)
    #         rewards.append(reward)
    #     print("Episode reward: ", sum(rewards))
    #     print(info)
    # env.close()

    # replay.run(env_kwargs['config'])


if __name__ == "__main__":
    test()
