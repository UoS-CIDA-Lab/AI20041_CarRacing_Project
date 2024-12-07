import math
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import gymnasium as gym
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
import random
import pickle
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_preprocessing(img):
    img = cv2.resize(img, dsize=(84, 84))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_features = 64 * 4 * 4  # 더 큰 피처맵 생성

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self._n_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(f"Shape: {x.shape}")
        x = x.view((-1, self._n_features))
        x = self.fc(x)
        return x

class CarEnvironment(gym.Wrapper):
    def __init__(self, env, skip_frames=2, stack_frames=4, no_operation=5, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames

    def reset(self):
        observation, info = self.env.reset()

        for i in range(self._no_operation):
            random_action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(random_action)
            print(info)
            
        observation = image_preprocessing(observation)
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info

    def step(self, action):
        total_reward = 0
        for i in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            if action == 3:
                reward = reward * 1.5
            total_reward += reward
            if terminated or truncated:
                break
        
        observation = image_preprocessing(observation)
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
        return self.stack_state, total_reward, terminated, truncated, info

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN:
    def __init__(self, action_space, batch_size, gamma, eps_start, eps_end, eps_decay, lr):
        self._n_observation = 4
        self._n_actions = 5
        self._action_space = action_space
        self._batch_size = batch_size
        self._gamma = gamma
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._lr = lr
        self._total_steps = 0
        self._evaluate_loss = []
        self.network = CNN(self._n_observation, self._n_actions).to(device)
        self.target_network = CNN(self._n_observation, self._n_actions).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self._lr, amsgrad=True)
        self._memory = ReplayMemory(50000)

    """
    This function is called during training & evaluation phase when the agent
    interact with the environment and needs to select an action.

    (1) Exploitation: This function feeds the neural network a state
    and then it selects the action with the highest Q-value.
    (2) Evaluation mode: This function feeds the neural network a state
    and then it selects the action with the highest Q'-value.
    (3) Exploration mode: It randomly selects an action through sampling

    Q -> network (policy)
    Q'-> target network (best policy)
    """
    def select_action(self, state, evaluation_phase=False):
        # Generating a random number for exploration vs exploitation
        sample = random.random()

        # Calculating the threshold - the more steps the less exploration we do
        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * \
                        math.exp(-1. * self._total_steps / self._eps_decay)
                        
        self._total_steps += 1

        if evaluation_phase:
            with torch.no_grad():
                return self.target_network(state).max(1).indices.view(1, 1)
        elif sample > eps_threshold:
            with torch.no_grad():
                return self.network(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self._action_space.sample()]], device=device, dtype=torch.long)

    def train(self):
        if len(self._memory) < self._batch_size:
            return

        # Initializing our memory
        transitions = self._memory.sample(self._batch_size)

        # Initializing our batch
        batch = Transition(*zip(*transitions))

        # Saving in a new tensor all the indices of the states that are non terminal
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=device, dtype=torch.bool)

        # Saving in a new tensor all the non final next states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Feeding our Q network the batch with states and then we gather the Q values of the selected actions
        state_action_values = self.network(state_batch).gather(1, action_batch)

        # We then, for every state in the batch that is NOT final, we pass it in the target network to get the Q'-values and choose the max one
        next_state_values = torch.zeros(self._batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values

        # Computing the expecting values with: reward + gamma * max(Q')
        expected_state_action_values = (next_state_values * self._gamma) + reward_batch

        # Defining our loss criterion
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Updating with back propagation
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()

        self._evaluate_loss.append(loss.item())

        return

    def copy_weights(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def get_loss(self):
        return self._evaluate_loss

    def save_model(self, i):
        torch.save(self.target_network.state_dict(), f'./reverse_catch_weights/model_weights_{i}.pth')

    def load_model(self, i):
        self.target_network.load_state_dict(torch.load(f'model_weights_{i}.pth', map_location=device))

if __name__ == "__main__":
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    rewards_per_episode = []
    episode_duration = []
    average_episode_loss = []

    episodes = 10000
    C = 5

    env = gym.make('CarRacing-v2', continuous=False)
    env = CarEnvironment(env)
    n_actions = env.action_space
    agent = DQN(
		action_space=n_actions,
        batch_size=64,        # 배치 크기를 32로 설정
		eps_start=1.0,        # 탐험 비율을 높게 시작
		eps_end=0.1,          # 탐험 비율의 하한값을 조금 높임
		eps_decay=1000,       # 탐험 비율 감소 속도를 느리게 설정
		gamma=0.98,            # 감쇠 계수 증가로 보상의 미래 중요성을 조금 더 강조
        lr=0.001
	)

    for episode in tqdm(range(1, episodes + 1)):
        print(f"Episode: {episode}")
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_total_reward = 0

        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            episode_total_reward += reward.item()
            done = terminated or truncated

            if terminated:
                next_state = None
                print("Finished the lap successfully!")
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent._memory.push(state, action, next_state, reward)
            state = next_state

            agent.train() 
            
            # print(f"action: {action}, reward: {reward.item()}")

            if done:
                episode_duration.append(t + 1)
                rewards_per_episode.append(episode_total_reward)
                ll = agent.get_loss()
                average_episode_loss.append(sum(ll) / len(ll))
                break

        if episode % C == 0:
            agent.copy_weights()
                    
        if episode % 100 == 0:
            agent.save_model(episode)
            print(f"Episode: {episode}, Total Reward: {episode_total_reward}, Average Loss: {average_episode_loss[-1]}")
            with open('statistics.pkl', 'wb') as f:
                pickle.dump((episode_duration, rewards_per_episode, average_episode_loss), f)

    agent.save_model(episodes)
    with open('statistics.pkl', 'wb') as f:
        pickle.dump((episode_duration, rewards_per_episode, average_episode_loss), f)