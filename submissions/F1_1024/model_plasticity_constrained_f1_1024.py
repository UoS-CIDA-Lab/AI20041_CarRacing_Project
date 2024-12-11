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
from module import DQN, ConstrainedDQN
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_preprocessing(img):
  img = cv2.resize(img, dsize=(84, 84))
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
  return img

def plot_statistics(x, y, title, x_axis, y_axis):
    plt.plot(x, y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'{title.replace(" ", "_")}.png')  # 공백 대신 밑줄 사용
    plt.show()

class CarEnvironment(gym.Wrapper):
  def __init__(self, env, skip_frames=2, stack_frames=4, no_operation=5, **kwargs):
    super().__init__(env, **kwargs)
    self._no_operation = no_operation
    self._skip_frames = skip_frames
    self._stack_frames = stack_frames

  def reset(self):
    observation, info = self.env.reset()

    for i in range(self._no_operation):
      observation, reward, terminated, truncated, info = self.env.step(0)

    observation = image_preprocessing(observation)
    self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
    return self.stack_state, info


  def step(self, action):
    total_reward = 0
    for i in range(self._skip_frames):
      observation, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      if terminated or truncated:
        break

    observation = image_preprocessing(observation)
    self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
    return self.stack_state, total_reward, terminated, truncated, info

def reset_envs_with_random_seeds(envs):
    """
    각 환경에 랜덤 시드를 설정하고 초기화.
    """
    new_seeds = [np.random.randint(1000) for _ in range(len(envs))]
    states = []
    for env, seed in zip(envs, new_seeds):
        env.np_random = np.random.default_rng(seed)  # 랜덤 시드 재설정
        state, _ = env.reset()  # 환경 초기화
        states.append(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
    return states


# argparse를 이용해 파라미터를 받아오기
parser = argparse.ArgumentParser(description="Training Script")
parser.add_argument("--TU", type=int, default=3, help="Policy update cycle")
parser.add_argument("--TC", type=int, default=5, help="Target network update cycle")
parser.add_argument("--TR", type=int, default=50, help="Shrink & Perturb cycle")
parser.add_argument("--map_reset_interval", type=int, default=50, help="Map reset interval")
parser.add_argument("--replay_ratio", type=int, default=10, help="Replay Ratio")
parser.add_argument("--episodes", type=int, default=800, help="Number of episodes")
parser.add_argument("--alpha", type=float, default=0.9, help="Shrink & Perturb ratio")
parser.add_argument("--save_point", type=int, default=50, help="Model save interval")
parser.add_argument("--env_count", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--max_timestep", type=int, default=900, help="Range of Max Timestep")
parser.add_argument("--constrained_use", type=int, default=1, help="Constrained RL Usage")
parser.add_argument("--max_speed", type=int, default=50, help="Max speed of Constrained")

args = parser.parse_args()

# 설정값 가져오기
TU = args.TU
TC = args.TC
TR = args.TR
map_reset_interval = args.map_reset_interval
replay_ratio = args.replay_ratio
episodes = args.episodes
alpha = args.alpha
save_point = args.save_point
env_count = args.env_count
max_timestep = args.max_timestep
constrained_use = args.constrained_use
max_speed = args.max_speed

if constrained_use == 1:
    version=f"ConstrainedRL_mapinterval{map_reset_interval}_replayratio{replay_ratio}_alpha{alpha}_TU{TU}_TR{TR}_{map_reset_interval}_epi{episodes}_maxspeed{max_speed}"
else:
    version=f"basemodel_mapinterval{map_reset_interval}_replayratio{replay_ratio}_alpha{alpha}_TU{TU}_TR{TR}_{map_reset_interval}_epi{episodes}"
save_dir = f'model/{version}'
os.makedirs(save_dir, exist_ok=True)

# 초기 시드 설정
np.random.seed(42)  # NumPy 랜덤 시드 고정
torch.manual_seed(42)  # PyTorch 랜덤 시드 고정

# wandb로 트래킹
import wandb

# wandb 초기화
wandb.init(
    project="F1024 Project (Car-racing)",  # 프로젝트 이름
    name=version,          # 실험 이름
    config={               # 설정값 로깅
        "TU": TU,
        "TC": TC,
        "TR": TR,
        "map_reset_interval": map_reset_interval,
        "replay_ratio": replay_ratio,
        "episodes": episodes,
        "alpha": alpha,
        "env_count": env_count,
        "max_timestep": max_timestep,
        "constrained_use": constrained_use,
        "max_speed": max_speed
    }
)

# 최고 성능 따로 저장해두기
best_reward = float('-inf')  # 최고 보상 저장
best_episode = -1            # 최고 보상 에피소드
best_model_path = None       # 최고 보상 모델 경로

# 병렬 환경 생성
envs = [CarEnvironment(gym.make('CarRacing-v2', continuous=False, max_episode_steps=1000)) for _ in range(env_count)]  # 최초 환경 생성
if constrained_use == 1:
    use_constrained_rl = True
agent = ConstrainedDQN(action_space=envs[0].action_space, use_constrained_rl=use_constrained_rl, alpha=alpha, max_speed=max_speed)

# reward, duration, average loss 초기화
rewards_per_episode = []
episode_duration = []
average_episode_loss = []

for episode in tqdm(range(1, episodes + 1), desc="Training"):

    # 특정 주기마다 맵 변경
    if episode % map_reset_interval == 1:  # 맵 변경 주기 시작
        states = reset_envs_with_random_seeds(envs)
    else:
        states = [env.reset()[0] for env in envs]  # 기존 방식으로 환경 초기화
        states = [torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0) for s in states]

    episode_total_rewards = [0] * len(envs)  # 각 환경의 총 보상 저장
    episode_losses = []  # loss 저장
    episode_durations = [0] * len(envs)  # 각 환경의 지속 시간 저장
    done_mask = [False] * len(envs)  # 환경별 종료 상태 트래킹

    # 에피소드 시작 시 에이전트의 손실 초기화
    agent._evaluate_loss = []  # 손실 초기화

    for t in range(max_timestep):  # 무한 루프
        # CNN에 전달하기 위해 states를 텐서(batch)로 변환
        # 종료되지 않은 환경 상태를 결합
        valid_indices = [i for i in range(len(envs)) if not done_mask[i]]
        valid_states = [states[i] for i in valid_indices if states[i] is not None]
        if len(valid_states) == 0:  # 모든 환경이 종료된 경우
            break
        states_tensor = torch.cat(valid_states, dim=0)
        
        # 에이전트가 행동 선택
        actions = [
            agent.select_action(states_tensor[idx].unsqueeze(0))
            for idx in range(len(valid_indices))
        ]
        # 환경 상호작용 (종료되지 않은 환경만)
        for idx, env_idx in enumerate(valid_indices):
            next_state, reward, terminated, truncated, info = envs[env_idx].step(actions[idx].item())
            reward = torch.tensor([reward], device=device)
            done_flags = terminated or truncated
    
            if not done_flags:  # 환경이 종료되지 않은 경우 업데이트
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                agent.memory.push(states[env_idx], actions[idx], next_state, reward, info)
                states[env_idx] = next_state  # 상태 업데이트
                episode_total_rewards[env_idx] += reward
                episode_durations[env_idx] += 1
            else:
                states[env_idx] = None  # 종료된 환경은 상태를 None으로 유지
                done_mask[env_idx] = True  # 종료 상태 갱신

        # 학습 수행 후 손실을 기록
        if t % TU == 0:
            agent.train(replay_ratio)
            loss_values = agent.get_loss()
            if loss_values:  # 손실 리스트가 비어 있지 않은 경우
                episode_losses.append(loss_values)
                
        # 모든 환경이 종료되었는지 확인
        if all(done_mask):
            # 각 환경의 총 보상과 손실 평균을 계산하여 저장
            avg_reward = sum(episode_total_rewards) / len(envs)
            loss_record = sum([sum(losses) / len(losses) for losses in episode_losses if losses])
            avg_duration = sum(episode_durations) / len(envs)  # 평균 지속 시간 계산
            
            # 평균 보상 및 손실을 저장
            rewards_per_episode.append(avg_reward)
            average_episode_loss.append(loss_record)
            episode_duration.append(avg_duration)

            # 보상, 손실, 지속 시간을 각각 로깅
            wandb.log({
                "Episode": episode,           # x축 값
                "Average Reward": avg_reward, # 그래프 1: 평균 보상
                "Average Loss": loss_record,     # 그래프 2: 평균 손실
                "Average Duration": avg_duration  # 그래프 3: 평균 지속 시간
            })
            # 환경 변경 조건 확인
            if episode > map_reset_interval and avg_reward > best_reward:
                best_reward = avg_reward
                best_episode = episode
                best_model_path = f"{best_episode}_reward_{int(best_reward)}"
                agent.save_model(version, best_model_path)
                print(f"New best model saved at episode {best_episode} with reward: {best_reward}")
            break
            
    if episode % TC == 0:
        agent.copy_weights()

    if episode % TR == 0:
        agent.shrink_and_perturb(alpha)

    if episode % save_point == 0:
        print(f"{episode} episodes done")
        agent.save_model(version, episode)
        with open('statistics.pkl', 'wb') as f:
            pickle.dump((episode_duration, rewards_per_episode, average_episode_loss), f)

agent.save_model(version, episode)
with open('statistics.pkl', 'wb') as f:
    pickle.dump((episode_duration, rewards_per_episode, average_episode_loss), f)