import copy
from dataclasses import dataclass
from typing import Callable
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import buffer


@dataclass
class Hyperparameters:
	# Generic
	batch_size: int = 256
	buffer_size: int = 1e7
	discount: float = 0.99
	target_update_rate: int = 250
	exploration_noise: float = 0.1
	
	# TD3
	target_policy_noise: float = 0.2
	noise_clip: float = 0.5
	policy_freq: int = 2
	
	# LAP
	alpha: float = 0.4
	min_priority: float = 1
	
	# TD3+BC
	lmbda: float = 0.1
	
	# Checkpointing
	max_eps_when_checkpointing: int = 20
	steps_before_checkpointing: int = 75e4 
	reset_weight: float = 0.9
	
	# Encoder Model
	zs_dim: int = 256
	enc_hdim: int = 256
	enc_activ: Callable = F.elu
	encoder_lr: float = 3e-4
	
	# Critic Model
	critic_hdim: int = 256
	critic_activ: Callable = F.elu
	critic_lr: float = 3e-4
	
	# Actor Model
	actor_hdim: int = 256
	actor_activ: Callable = F.relu
	actor_lr: float = 3e-4
	
	# M2TD3 specific - Omega parameters
	omega_dim: int = 2
	omega_std_rate: float = 0.1
	min_omega_std_rate: float = 0.01
	omega_noise_rate: float = 0.2
	noise_clip_omega_rate: float = 0.5
	
	# HatOmega parameters
	hatomega_num: int = 5
	hatomega_hidden_num: int = 2
	hatomega_hidden_size: int = 64
	hatomega_lr: float = 3e-4
	
	# Restart mechanisms
	restart_distance: bool = True
	restart_probability: bool = True
	hatomega_parameter_distance: float = 0.1
	minimum_prob: float = 0.01
	
	# Environment omega bounds
	omega_min: list = None
	omega_max: list = None


def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
	return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class HatOmegaNetwork(nn.Module):
	"""HatOmega network for generating omega parameters"""
	def __init__(self, omega_dim, omega_min, omega_max, hidden_num=2, hidden_size=64, device=None):
		super(HatOmegaNetwork, self).__init__()

		self.device = device
		self.omega_dim = omega_dim
		self.omega_min = torch.tensor(omega_min, dtype=torch.float, device=device)
		self.omega_max = torch.tensor(omega_max, dtype=torch.float, device=device)
		
		# Simple network that takes a dummy input and outputs omega
		layers = []
		layers.append(nn.Linear(1, hidden_size))
		layers.append(nn.ReLU())
		
		for _ in range(hidden_num - 1):
			layers.append(nn.Linear(hidden_size, hidden_size))
			layers.append(nn.ReLU())
			
		layers.append(nn.Linear(hidden_size, omega_dim))
		layers.append(nn.Sigmoid())  # Output between 0 and 1
		
		self.network = nn.Sequential(*layers)
		
		# Initialize with random omega values
		self._initialize_random()
	
	def _initialize_random(self):
		"""Initialize network to output random omega values"""
		with torch.no_grad():
			# Set final layer bias to random values between omega_min and omega_max
			final_layer = self.network[-2]  # -1 is Sigmoid, -2 is Linear
			omega_range = self.omega_max - self.omega_min
			random_omega = torch.rand(self.omega_dim, device=self.device) * omega_range + self.omega_min
			# Convert to sigmoid space
			sigmoid_omega = (random_omega - self.omega_min) / omega_range
			# Convert to logit space for bias
			logit_omega = torch.log(sigmoid_omega / (1 - sigmoid_omega + 1e-8))
			final_layer.bias.data = logit_omega
	
	def forward(self, x):
		# x is dummy input, usually just [1]
		sigmoid_output = self.network(x)
		# Scale from [0,1] to [omega_min, omega_max]
		omega_range = self.omega_max - self.omega_min
		return sigmoid_output * omega_range + self.omega_min


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
		super(Actor, self).__init__()

		self.activ = activ

		self.l0 = nn.Linear(state_dim, hdim)
		self.l1 = nn.Linear(zs_dim + hdim, hdim)
		self.l2 = nn.Linear(hdim, hdim)
		self.l3 = nn.Linear(hdim, action_dim)
		

	def forward(self, state, zs):
		a = AvgL1Norm(self.l0(state))
		a = torch.cat([a, zs], 1)
		a = self.activ(self.l1(a))
		a = self.activ(self.l2(a))
		return torch.sigmoid(self.l3(a))


class Encoder(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Encoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, zs_dim)
		
		# state-action encoder
		self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
		self.zsa2 = nn.Linear(hdim, hdim)
		self.zsa3 = nn.Linear(hdim, zs_dim)
	

	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		return zs


	def zsa(self, zs, action):
		zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
		zsa = self.activ(self.zsa2(zsa))
		zsa = self.zsa3(zsa)
		return zsa


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, omega_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Critic, self).__init__()

		self.activ = activ
		
		# Modified to include omega in the input
		self.q01 = nn.Linear(state_dim + action_dim + omega_dim, hdim)
		self.q1 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q2 = nn.Linear(hdim, hdim)
		self.q3 = nn.Linear(hdim, 1)

		self.q02 = nn.Linear(state_dim + action_dim + omega_dim, hdim)
		self.q4 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q5 = nn.Linear(hdim, hdim)
		self.q6 = nn.Linear(hdim, 1)


	def forward(self, state, action, omega, zsa, zs):
		sao = torch.cat([state, action, omega], 1)  # Include omega
		embeddings = torch.cat([zsa, zs], 1)

		q1 = AvgL1Norm(self.q01(sao))
		q1 = torch.cat([q1, embeddings], 1)
		q1 = self.activ(self.q1(q1))
		q1 = self.activ(self.q2(q1))
		q1 = self.q3(q1)

		q2 = AvgL1Norm(self.q02(sao))
		q2 = torch.cat([q2, embeddings], 1)
		q2 = self.activ(self.q4(q2))
		q2 = self.activ(self.q5(q2))
		q2 = self.q6(q2)
		return torch.cat([q1, q2], 1)
	
	def Q1(self, state, action, omega, zsa, zs):
		sao = torch.cat([state, action, omega], 1)
		embeddings = torch.cat([zsa, zs], 1)
		
		q1 = AvgL1Norm(self.q01(sao))
		q1 = torch.cat([q1, embeddings], 1)
		q1 = self.activ(self.q1(q1))
		q1 = self.activ(self.q2(q1))
		return self.q3(q1)


class M2TD7Buffer(buffer.LAP):
	"""Extended LAP buffer to include omega"""
	def __init__(self, state_dim, action_dim, omega_dim, device, max_size=1e6, batch_size=256, 
				 max_action=1, normalize_actions=True, prioritized=True):
		super().__init__(state_dim, action_dim, device, max_size, batch_size, 
						max_action, normalize_actions, prioritized)
		
		# Add omega storage
		max_size = int(max_size)
		self.omega = np.zeros((max_size, omega_dim))
	
	def add(self, state, action, next_state, reward, done, omega):
		self.state[self.ptr] = state
		self.action[self.ptr] = action/self.normalize_actions
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.omega[self.ptr] = omega
		
		if self.prioritized:
			self.priority[self.ptr] = self.max_priority

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self):
		if self.prioritized:
			csum = torch.cumsum(self.priority[:self.size], 0)
			val = torch.rand(size=(self.batch_size,), device=self.device)*csum[-1]
			self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
		else:
			self.ind = np.random.randint(0, self.size, size=self.batch_size)

		return (
			torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.omega[self.ind], dtype=torch.float, device=self.device)
		)


class M2TD7Agent(object):
	def __init__(self, state_dim, action_dim, max_action, hp=Hyperparameters()): 
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.hp = hp
		self.max_action = max_action
		
		# Set default omega bounds if not provided
		if hp.omega_min is None:
			hp.omega_min = [-1.0] * hp.omega_dim
		if hp.omega_max is None:
			hp.omega_max = [1.0] * hp.omega_dim
			
		self.omega_min = np.array(hp.omega_min)
		self.omega_max = np.array(hp.omega_max)
		self.omega_min_tensor = torch.tensor(hp.omega_min, dtype=torch.float, device=self.device)
		self.omega_max_tensor = torch.tensor(hp.omega_max, dtype=torch.float, device=self.device)

		# Initialize networks
		self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.critic = Critic(state_dim, action_dim, hp.omega_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
		self.critic_target = copy.deepcopy(self.critic)

		self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
		self.fixed_encoder = copy.deepcopy(self.encoder)
		self.fixed_encoder_target = copy.deepcopy(self.encoder)

		# Initialize HatOmega networks
		self.hatomega_list = []
		self.hatomega_optimizers = []
		for i in range(hp.hatomega_num):
			hatomega = HatOmegaNetwork(
				hp.omega_dim, hp.omega_min, hp.omega_max, 
				hp.hatomega_hidden_num, hp.hatomega_hidden_size, self.device
			).to(self.device)
			optimizer = torch.optim.Adam(hatomega.parameters(), lr=hp.hatomega_lr)
			self.hatomega_list.append(hatomega)
			self.hatomega_optimizers.append(optimizer)

		# HatOmega selection probabilities
		self.hatomega_prob = [1.0 / hp.hatomega_num] * hp.hatomega_num
		
		# Omega noise parameters
		omega_range = self.omega_max - self.omega_min
		self.omega_std = hp.omega_std_rate * omega_range / 2
		self.min_omega_std = hp.min_omega_std_rate * omega_range / 2
		self.omega_noise = hp.omega_noise_rate * omega_range / 2
		self.noise_clip_omega = torch.tensor(
			hp.noise_clip_omega_rate * omega_range / 2, 
			device=self.device, dtype=torch.float
		)

		# Dummy inputs for HatOmega
		self.hatomega_input = torch.tensor([[1]], dtype=torch.float, device=self.device)
		self.hatomega_input_batch = torch.tensor(
			[[1]] * hp.batch_size, dtype=torch.float, device=self.device
		)

		self.checkpoint_actor = copy.deepcopy(self.actor)
		self.checkpoint_encoder = copy.deepcopy(self.encoder)

		self.replay_buffer = M2TD7Buffer(
			state_dim, action_dim, hp.omega_dim, self.device, hp.buffer_size, hp.batch_size, 
			max_action=1.0, normalize_actions=False, prioritized=True
		)

		self.training_steps = 0
		self.current_episode_len = 1

		# Checkpointing tracked values
		self.eps_since_update = 0
		self.timesteps_since_update = 0
		self.max_eps_before_update = 1
		self.min_return = 1e8
		self.best_min_return = -1e8

		# Value clipping tracked values
		self.max = -1e8
		self.min = 1e8
		self.max_target = 0
		self.min_target = 0

	def select_action(self, state, use_checkpoint=False, use_exploration=True):
		with torch.no_grad():
			state = torch.tensor(state.reshape(1,-1), dtype=torch.float, device=self.device)

			if use_checkpoint: 
				zs = self.checkpoint_encoder.zs(state)
				action = self.checkpoint_actor(state, zs) 
			else: 
				zs = self.fixed_encoder.zs(state)
				action = self.actor(state, zs) 
			
			if use_exploration: 
				action = action + torch.randn_like(action) * self.hp.exploration_noise
				action = torch.clamp(action, 0, 1)

			return action.cpu().data.numpy().flatten()

	def get_omega(self, greedy=False):
		"""Get omega parameter using HatOmega networks"""
		# Restart mechanisms
		dis_restart_flag = False
		prob_restart_flag = False
		
		if self.hp.restart_distance:
			change_indices = self._calc_diff()
			for idx in change_indices:
				self._init_hatomega(idx)
				self._init_hatomega_prob(idx)
				dis_restart_flag = True
				
		if self.hp.restart_probability:
			change_indices = self._minimum_prob()
			for idx in change_indices:
				self._init_hatomega(idx)
				self._init_hatomega_prob(idx)
				prob_restart_flag = True

		# Select HatOmega and generate omega
		hatomega_index = self._select_hatomega()
		omega = self.hatomega_list[hatomega_index](self.hatomega_input)

		if not greedy:
			noise = torch.tensor(
				np.random.normal(0, self.omega_std), 
				dtype=torch.float, device=self.device
			)
			omega += noise
			omega = torch.clamp(omega, self.omega_min_tensor, self.omega_max_tensor)

		return (
			omega.squeeze(0).detach().cpu().numpy(),
			dis_restart_flag,
			prob_restart_flag,
		)

	def train(self):
		self.training_steps += 1

		state, action, next_state, reward, not_done, omega = self.replay_buffer.sample()

		#########################
		# Update Encoder
		#########################
		with torch.no_grad():
			next_zs = self.encoder.zs(next_state)

		zs = self.encoder.zs(state)
		pred_zs = self.encoder.zsa(zs, action)
		encoder_loss = F.mse_loss(pred_zs, next_zs)

		self.encoder_optimizer.zero_grad()
		encoder_loss.backward()
		self.encoder_optimizer.step()

		#########################
		# Update Critic
		#########################
		with torch.no_grad():
			fixed_target_zs = self.fixed_encoder_target.zs(next_state)

			noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
			next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(0,1)
			
			# Add omega noise
			omega_noise = torch.max(
				torch.min(
					torch.randn_like(omega) * torch.tensor(self.omega_noise, device=self.device, dtype=torch.float),
					self.noise_clip_omega,
				),
				-self.noise_clip_omega,
			)
			next_omega = torch.clamp(omega + omega_noise, self.omega_min_tensor, self.omega_max_tensor)
			
			fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

			Q_target = self.critic_target(next_state, next_action, next_omega, fixed_target_zsa, fixed_target_zs).min(1,keepdim=True)[0]
			Q_target = reward + not_done * self.hp.discount * Q_target.clamp(self.min_target, self.max_target)

			self.max = max(self.max, float(Q_target.max()))
			self.min = min(self.min, float(Q_target.min()))

			fixed_zs = self.fixed_encoder.zs(state)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

		Q = self.critic(state, action, omega, fixed_zsa, fixed_zs)
		td_loss = (Q - Q_target).abs()
		critic_loss = LAP_huber(td_loss)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		
		#########################
		# Update LAP
		#########################
		priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
		self.replay_buffer.update_priority(priority)

		#########################
		# Update Actor and HatOmega
		#########################
		if self.training_steps % self.hp.policy_freq == 0:
			# Find worst-case omega for actor update (similar to M2TD3)
			worst_policy_loss_value = -np.inf
			worst_policy_loss = None
			worst_hatomega_index = 0
			
			with torch.no_grad():
				fixed_zs = self.fixed_encoder.zs(state)
			
			for hatomega_index in range(self.hp.hatomega_num):
				hatomega_batch = self.hatomega_list[hatomega_index](self.hatomega_input_batch)
				
				actor_output = self.actor(state, fixed_zs)
				fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_output)
				
				policy_loss = -self.critic(state, actor_output, hatomega_batch.detach(), fixed_zsa, fixed_zs).mean()
				
				if policy_loss.item() >= worst_policy_loss_value:
					worst_policy_loss_value = policy_loss.item()
					worst_policy_loss = policy_loss
					worst_hatomega_index = hatomega_index

			# Update HatOmega (adversarial to actor)
			hatomega_batch = self.hatomega_list[worst_hatomega_index](self.hatomega_input_batch)
			actor_output = self.actor(state, fixed_zs)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_output)
			hatomega_loss = self.critic(state, actor_output.detach(), hatomega_batch, fixed_zsa.detach(), fixed_zs).mean()
			
			self.hatomega_optimizers[worst_hatomega_index].zero_grad()
			hatomega_loss.backward()
			self.hatomega_optimizers[worst_hatomega_index].step()
			
			# Update actor with worst-case omega
			self.actor_optimizer.zero_grad()
			worst_policy_loss.backward()
			self.actor_optimizer.step()

			# Update HatOmega selection probability
			self._update_hatomega_prob(worst_hatomega_index)

		#########################
		# Update Target Networks
		#########################
		if self.training_steps % self.hp.target_update_rate == 0:
			self.actor_target.load_state_dict(self.actor.state_dict())
			self.critic_target.load_state_dict(self.critic.state_dict())
			self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
			self.fixed_encoder.load_state_dict(self.encoder.state_dict())
			
			self.replay_buffer.reset_max_priority()

			self.max_target = self.max
			self.min_target = self.min

	def _calc_diff(self):
		"""Calculate distance between HatOmega networks"""
		change_indices = []
		omega_outputs = []
		
		for i in range(self.hp.hatomega_num):
			omega = self.hatomega_list[i](self.hatomega_input).squeeze(0).detach().cpu().numpy()
			omega_outputs.append(omega)
		
		for i, j in itertools.combinations(range(len(omega_outputs)), 2):
			distance = np.linalg.norm(omega_outputs[i] - omega_outputs[j], ord=1)
			if distance <= self.hp.hatomega_parameter_distance:
				change_indices.append(i)
				
		return list(set(change_indices))

	def _minimum_prob(self):
		"""Find HatOmega networks with probability below threshold"""
		return [i for i, prob in enumerate(self.hatomega_prob) if prob < self.hp.minimum_prob]

	def _select_hatomega(self):
		"""Select HatOmega network based on probabilities"""
		probs = np.array(self.hatomega_prob)
		probs = probs / probs.sum()  # Normalize
		return np.random.choice(len(self.hatomega_list), p=probs)

	def _update_hatomega_prob(self, index):
		"""Update selection probability for HatOmega"""
		p = [0] * self.hp.hatomega_num
		p[index] = 1
		coeff = 1 / self.current_episode_len
		
		for i in range(self.hp.hatomega_num):
			self.hatomega_prob[i] = self.hatomega_prob[i] * (1 - coeff) + coeff * p[i]

	def _init_hatomega(self, index):
		"""Reinitialize a HatOmega network"""
		self.hatomega_list[index] = HatOmegaNetwork(
			self.hp.omega_dim, self.hp.omega_min, self.hp.omega_max,
			self.hp.hatomega_hidden_num, self.hp.hatomega_hidden_size, self.device
		).to(self.device)
		self.hatomega_optimizers[index] = torch.optim.Adam(
			self.hatomega_list[index].parameters(), lr=self.hp.hatomega_lr
		)

	def _init_hatomega_prob(self, index):
		"""Reinitialize selection probability for HatOmega"""
		self.hatomega_prob[index] = 0
		sum_prob = sum(self.hatomega_prob)
		if sum_prob > 0:
			p = sum_prob / (self.hp.hatomega_num - 1)
			self.hatomega_prob[index] = p
		else:
			self.hatomega_prob = [1.0 / self.hp.hatomega_num] * self.hp.hatomega_num

	def set_current_episode_len(self, episode_len):
		"""Set current episode length for probability updates"""
		self.current_episode_len = episode_len

	# Checkpointing methods (from TD7)
	def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
		self.eps_since_update += 1
		self.timesteps_since_update += ep_timesteps

		self.min_return = min(self.min_return, ep_return)

		# End evaluation of current policy early
		if self.min_return < self.best_min_return:
			self.train_and_reset()

		# Update checkpoint
		elif self.eps_since_update == self.max_eps_before_update:
			self.best_min_return = self.min_return
			self.checkpoint_actor.load_state_dict(self.actor.state_dict())
			self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())
			
			self.train_and_reset()


	# Batch training
	def train_and_reset(self):
		for _ in range(self.timesteps_since_update):
			if self.training_steps == self.hp.steps_before_checkpointing:
				self.best_min_return *= self.hp.reset_weight
				self.max_eps_before_update = self.hp.max_eps_when_checkpointing
			
			self.train()

		self.eps_since_update = 0
		self.timesteps_since_update = 0
		self.min_return = 1e8