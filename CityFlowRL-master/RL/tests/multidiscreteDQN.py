import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from collections import deque
import gym
from tqdm import tqdm
import CityFlowRL
import os
from gym.spaces import MultiDiscrete

models_dir = "../models/"
env_kwargs = {'config': "hangzhou_1x1_bc-tyc_18041608_1h", 'steps_per_episode': 121,
              'steps_per_action': 30}  # ep = 3630
env = gym.make('CityFlowRL-v0', **env_kwargs)


class MultiDiscreteDQN:
    def __init__(self, state_space, action_space, learning_rate=0.00001, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.2, buffer_size=100000, batch_size=64):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.policy_net = self.create_model().to(self.device)
        self.target_net = self.create_model().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.last_actions = deque(maxlen=3)

    def create_model(self):
        class MultiDiscreteDQNet(nn.Module):
            def __init__(self, input_shape, output_shape):
                super(MultiDiscreteDQNet, self).__init__()
                actions = 0
                while env.action_space.contains(actions):
                    actions += 1
                self.fc1 = nn.Linear(int(np.prod(input_shape)), 64)  ##64 ,6432 ,32
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, actions)
                # self.fc4 = nn.Linear(64, 32)
                # self.fc5 = nn.Linear(32, actions)
                self.dropout = nn.Dropout(p=0.2)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                # x = torch.softmax(self.fc3(x), dim=-1)
                if len(x) == 0:
                    return torch.empty((x.size(0),) + (5, 1), device=x.device)
                else:
                    return x

        return MultiDiscreteDQNet(self.state_space.shape, self.action_space.shape)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                action = int(q_values.argmax(dim=-1).cpu().numpy().reshape(-1))

        return action

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        states = states.view(self.batch_size, 1, -1)  # Reshape to (batch_size, 1, state_size)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        policy_output = self.policy_net(states)
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.steps % 20 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


def train():
    env.set_save_replay(False)
    obs = env.observation_space
    action = env.action_space
    model = MultiDiscreteDQN(obs, action)
    total_episodes = 300
    model.create_model()
    writer = SummaryWriter(log_dir="logs")
    # model.load(os.path.join(models_dir, "multi-hang-bc-608-3.3M-lr0.000000001-ed0.7-drop0.1"))
    for episode in range(1, total_episodes + 1):
        is_done = False
        score = 0
        state = env.reset()
        total_reward = 0
        while not is_done:
            action = model.act(state)
            next_state, reward, is_done, info = env.step(action)
            model.remember(state, action, reward, next_state, is_done)
            model.learn()
            state = next_state
            total_reward += reward
        writer.add_scalar('reward', total_reward, episode * 3600)
        model.epsilon = max(model.epsilon_min, model.epsilon * model.epsilon_decay)
        print(f"Episode {episode}: Total reward = {total_reward}: avg-travel-time = {info}")

    model.save(os.path.join(models_dir, "multi-hang-608-new32"))
    print("model saved")


def test():
    env.set_save_replay(True)
    env.set_replay_path('dqnReplay.txt')
    obs = env.observation_space
    action = env.action_space
    model = MultiDiscreteDQN(obs, action)
    model.load(os.path.join(models_dir, "multi-hang-bc-608-3.3M-lr0.00000001-ed0.1"))
    num_test_episodes = 1
    for episode in range(num_test_episodes):
        state = env.reset()
        is_done = False
        total_reward = 0
        while is_done == False:
            # Choose the action with the highest Q-value
            action = model.act(state)
            # Take the action and observe the next state and reward
            next_state, reward, is_done, info = env.step(action)

            # Update the current state
            state = next_state
            total_reward += reward

        # Print the total reward earned in the episode
        print(f"Episode {episode + 1}: Total reward = {total_reward}: avg-travel-time = {info}")


if __name__ == "__main__":
    test()
