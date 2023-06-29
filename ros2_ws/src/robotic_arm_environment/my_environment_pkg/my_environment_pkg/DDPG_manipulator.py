import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(
        self, 
        state_size, 
        hidden_size, 
        action_size,
        device
        ) -> None:
        
        super(Actor, self).__init__()
        
        self.device = device
        
        self.fc_1 = nn.Linear(in_features=state_size, out_features=hidden_size)
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=int(hidden_size / 2))
        self.fc_3 = nn.Linear(in_features=int(hidden_size / 2), out_features=action_size)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)
        
    def forward(self, state):
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        action = F.tanh(self.fc_3(x))
        return action

class Critic(nn.Module):
    def __init__(
        self, 
        state_size,
        hidden_size, 
        action_size,
        device
        ) -> None:
        
        super(Critic, self).__init__()
        
        self.device = device
        
        self.state_fc_1 = nn.Linear(state_size, hidden_size)
        self.state_fc_2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.action_fc_1 = nn.Linear(action_size, int(hidden_size / 2))
        self.add_fc_1 = nn.Linear(hidden_size, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)
        
    def forward(self, state, action):
        x_s = F.relu(self.state_fc_1(state))
        x_s = F.relu(self.state_fc_2(x_s))
        
        x_a = F.relu(self.action_fc_1(action))
        
        x = torch.cat((x_s, x_a), dim=1)
        x = F.tanh(self.add_fc_1(x))
        return x

class OU_Noise():
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.1, min_sigma=0.1, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noise(self, t=0):
        ou_state = self.evolve_state()
        decaying = float(float(t) / self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_state

class Replay_Buffer():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        experience = (state, action, np.array([reward]), next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)

        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)

class Agent():
    def __init__(
        self,
        hidden_size=256, 
        actor_learning_rate=1e-4, 
        critic_learning_rate=1e-4,
        gamma=0.99, 
        tau=0.9999,
        tau_actor=9e-4,
        tau_critic=1e-4,
        max_memory_size=50000,
        ou_noise=False,
        n_learn=16,
        device='cpu'
        ):
        
        self.state_size = 10 # goal(Success(1) or failure(0)) + Coordinate at the end of each link (18) + Euclidean distance from target point to end-effector (3)
        self.action_size = 6 # Changes in the angles of each joint and their current angles

        self.gamma = gamma
        self.tau = tau
        self.tau_actor = tau_actor
        self.tau_critic = tau_critic
        self.t_step = 0  # counter for activating learning every few steps
        self.ou_noise = ou_noise
        self.n_learn = n_learn
        self.device = device
        
        if self.device == 'cuda':
            self.gamma = torch.FloatTensor([gamma]).to(self.device)
            self.tau = torch.FloatTensor([tau]).to(self.device)
            self.tau_actor = torch.FloatTensor([tau_actor]).to(self.device)
            self.tau_critic = torch.FloatTensor([tau_critic]).to(self.device)
        
        print(f"[device: {self.device}]")
        
        # Initialization of the networks
        self.actor  = Actor(self.state_size, hidden_size, self.action_size, self.device)   # main network Actor
        self.critic = Critic(self.state_size, hidden_size, self.action_size, self.device)  # main network Critic

        self.actor_target = Actor(self.state_size, hidden_size, self.action_size, self.device)
        self.critic_target = Critic(self.state_size, hidden_size, self.action_size, self.device)

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.actor_target = self.actor_target.to(self.device)
        self.critic_target = self.critic_target.to(self.device)
        
        # print("actor:", self.actor.is_cuda())
        # print("critic:", self.critic.is_cuda())
        # print("actor_target:", self.actor_target.is_cuda())
        # print("critic_target:", self.critic_target.is_cuda())
        
        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Initialization memory
        self.memory = Replay_Buffer(max_memory_size)

        self.actor_optimizer  = optim.AdamW(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_learning_rate)
        
        if self.ou_noise is True:
            self.noise = OU_Noise(action_dim=self.action_size)

    def load_weights(self, actor_path=None, critic_path=None):
        if actor_path is None or critic_path is None: return
        
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
    
    def get_action(self, state):
        state = np.array(state)
        state_change = torch.from_numpy(state).float().to(self.device)
        
        action = self.actor.forward(state_change)
        
        if self.ou_noise is True:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(self.t_step))).to(self.device)
            action = torch.clamp(torch.add(action, noise), -3.14159, 3.14159).to(self.device)
            
        action = action.detach().cpu().numpy()
        
        float_action = []
        for float64_action in action:
            float_action.append(float64_action.item())
        
        return float_action
    
    def step_training(self, batch_size):
        self.t_step = self.t_step + 1
        
        if self.memory.__len__() > batch_size:
            for _ in range(self.n_learn):
                self.learn_step(batch_size)
            
            if self.device == 'cuda':
                return self.actor_loss.detach().cpu().numpy(), self.critic_loss.detach().cpu().numpy()
            else:
                return self.actor_loss.detach().numpy(), self.critic_loss.detach().numpy()
        
        return None, None
    
    def learn_step(self, batch_size):
        state, action, reward, next_state = self.memory.sample(batch_size)
        
        state  = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        next_state = np.array(next_state)
        
        state  = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        Q_value = self.critic.forward(state, action)
        next_action = self.actor_target.forward(next_state)
        next_Q_value = self.critic_target.forward(next_state, next_action)
        target_Q_value = reward + self.gamma * next_Q_value
        
        loss = nn.MSELoss()
        self.critic_loss = loss(Q_value, target_Q_value)
        self.actor_loss = - self.critic.forward(state, self.actor.forward(state)).mean()
        
        self.actor.zero_grad()
        self.actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau_actor + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau_critic + target_param.data * (1.0 - self.tau))

    def save_model(self, save_path):
        torch.save(self.actor.state_dict(), f"{save_path}/actor.pt")
        torch.save(self.critic.state_dict(), f"{save_path}/critic.pt")