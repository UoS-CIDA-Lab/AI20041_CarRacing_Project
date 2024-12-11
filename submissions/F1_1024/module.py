import gymnasium as gym
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as T
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
import random
import math
import pickle
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._n_features = 32 * 9 * 9

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.ReLU(),
    )

    self.fc = nn.Sequential(
        nn.Linear(self._n_features, 256),
        nn.ReLU(),
        nn.Linear(256, out_channels),
    )


  def forward(self, x):
    x = self.conv(x)
    x = x.view((-1, self._n_features))
    x = self.fc(x)
    return x

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'info'))

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
  def __init__(self, action_space, batch_size=256, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, lr=0.001, use_vit=False):
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
      self.use_vit = use_vit  
      
      # 네트워크 초기화
      self.network = self.build_network()
      self.target_network = self.build_network()
      self.target_network.load_state_dict(self.network.state_dict())
      self.optimizer = optim.AdamW(self.network.parameters(), lr=self._lr, amsgrad=True)
      self.memory = ReplayMemory(10000)  
      
      # 초기 네트워크 파라미터 저장 (Shrink & Perturb용)
      self.initial_params = self.network.state_dict()  
      
  def build_network(self):
        if self.use_vit:
            # ViT 모델을 weights 매개변수를 사용하여 불러옴
            weights = ViT_B_16_Weights.DEFAULT
            model = vit_b_16(weights=weights)
            model.heads.head = nn.Linear(model.heads.head.in_features, self._n_actions)
            return model.to(device)
        else:
            return CNN(self._n_observation, self._n_actions).to(device)

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
    # Generating a random number for eploration vs exploitation
    sample = random.random()

    # Calculating the threshold - the more steps the less exploration we do
    eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * self._total_steps / self._eps_decay)
    self._total_steps += 1

    if evaluation_phase:
      with torch.no_grad():
        return self.target_network(state).max(1).indices.view(1, 1)
    elif sample > eps_threshold:
      with torch.no_grad():
        return self.network(state).max(1).indices.view(1, 1)
    else:
      return torch.tensor([[self._action_space.sample()]], device=device, dtype=torch.long)
        

  def train(self, replay_ratio=1):
    if len(self.memory) < self._batch_size:
        return
        
    for _ in range(replay_ratio):
        # Initializing our memory
        transitions = self.memory.sample(self._batch_size)
        # Initializing our batch
        batch = Transition(*zip(*transitions))
        
        # Saving in a new tensor all the indices of the states that are non terminal
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
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
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()
        self._evaluate_loss.append(loss.item())
            
  def shrink_and_perturb(self, alpha=0.9):
    current_params = self.network.state_dict()
    perturbed_params = {k: alpha * current_params[k] + (1 - alpha) * self.initial_params[k] for k in current_params}
    self.network.load_state_dict(perturbed_params)
      
  def copy_weights(self):
    self.target_network.load_state_dict(self.network.state_dict())

  def get_loss(self):
    return self._evaluate_loss

  def save_model(self, version, i):
    save_dir = f'../model/{version}'
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (이미 존재해도 무시)
    torch.save(self.target_network.state_dict(), f'model/{version}/model_weights_{i}.pth')

  def load_model(self, version, i):
    self.target_network.load_state_dict(torch.load(f'model/{version}/model_weights_{i}.pth', map_location=device))



# Constrained DQN 클래스 정의
class ConstrainedDQN:
    def __init__(self, action_space, use_constrained_rl=True, batch_size=256, gamma=0.99, lr=0.001,
                 lambda_lr=0.01, eps_start=0.9, eps_end=0.05, eps_decay=1000, replay_capacity=10000, alpha=0.9, max_speed=50):
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.max_speed = max_speed
        self.lr = lr
        self.use_constrained_rl = use_constrained_rl
        self.alpha = alpha  # Shrink & Perturb 비율
        self._total_steps = 0
        self._evaluate_loss = []

        # 라그랑지 승수 및 학습률
        self.lambda_speed = 0.0
        self.lambda_off_track = 0.0
        self.lambda_lr = lambda_lr

        # 네트워크 초기화
        self.network = CNN(4, action_space.n).to(device)
        self.target_network = CNN(4, action_space.n).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # 초기 네트워크 파라미터 저장 (Shrink & Perturb용)
        self.initial_params = self.network.state_dict()

        # 메모리 초기화
        self.memory = ReplayMemory(replay_capacity)

        # 학습 상태
        self.steps_done = 0


    def select_action(self, state, evaluation_phase=False):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if evaluation_phase:
            with torch.no_grad():
                return self.target_network(state).max(1).indices.view(1, 1)
        elif random.random() > eps_threshold:
            with torch.no_grad():
                return self.network(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.action_space.sample()]], device=device, dtype=torch.long)


    def compute_lagrangian_reward(self, reward, info):
        """
        제약 조건 반영 보상 계산 (Constrained RL 활성화 시).
        """
        if not self.use_constrained_rl:
            return reward

        speed = info.get("speed", 0)
        off_track = info.get("off_track", False)

        # 제약 조건 위반 계산
        speed_violation = max(0, speed - self.max_speed)  # 속도 초과량
        off_track_penalty = 1 if off_track else 0

        # 라그랑지 조정 보상
        return reward - self.lambda_speed * speed_violation - self.lambda_off_track * off_track_penalty


    def update_lagrange_multipliers(self, info):
        """
        라그랑지 승수 업데이트 (Constrained RL 활성화 시).
        """
        if not self.use_constrained_rl:
            return

        speed = info.get("speed", 0)
        off_track = info.get("off_track", False)

        # 라그랑지 승수 업데이트
        speed_violation = max(0, speed - 50)
        off_track_penalty = 1 if off_track else 0

        self.lambda_speed = max(0, self.lambda_speed + self.lambda_lr * speed_violation)
        self.lambda_off_track = max(0, self.lambda_off_track + self.lambda_lr * off_track_penalty)


    def train(self, replay_ratio=1):
        if len(self.memory) < self.batch_size:
            return

        for _ in range(replay_ratio):
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # 라그랑지 조정 보상 계산
            adjusted_rewards = torch.tensor(
                [self.compute_lagrangian_reward(r.item(), info) for r, info in zip(reward_batch, batch.info)],
                device=device,
                dtype=torch.float
            )

            # Q-value 업데이트
            state_action_values = self.network(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(self.batch_size, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values
            expected_state_action_values = (next_state_values * self.gamma) + adjusted_rewards
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # 네트워크 업데이트
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
            self.optimizer.step()
            self._evaluate_loss.append(loss.item())

            # 라그랑지 승수 업데이트
            for info in batch.info:
                self.update_lagrange_multipliers(info)


    def shrink_and_perturb(self, alpha=0.9):
        """
        Shrink & Perturb: 현재 네트워크 가중치와 초기 가중치를 결합하여 Plasticity를 조정합니다.
        """
        current_params = self.network.state_dict()
        perturbed_params = {k: self.alpha * current_params[k] + (1 - self.alpha) * self.initial_params[k]
                            for k in current_params}
        self.network.load_state_dict(perturbed_params)
    

    def copy_weights(self):
        """
        네트워크 가중치 복사 (Target 네트워크 업데이트).
        """
        self.target_network.load_state_dict(self.network.state_dict())


    def push_memory(self, *args):
        self.memory.push(*args)
    

    def get_loss(self):
        return self._evaluate_loss


    def save_model(self, version, i):
        save_dir = f'../model/{version}'
        os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (이미 존재해도 무시)
        torch.save(self.target_network.state_dict(), f'model/{version}/model_weights_{i}.pth')


    def load_model(self, version, i):
        self.target_network.load_state_dict(torch.load(f'model/{version}/model_weights_{i}.pth', map_location=device))
