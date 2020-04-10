from RLPython.statemodifier import ClassicModifier
from RLPython.replaybuffer import ReplayBuffer
import torch
import os
import numpy as np

class AgentInterface():
	def __init__(self, env, actor, critic, args, render):
		self.env = env
		self.actor = actor
		self.critic = critic
		self._episodes = 0
		self.render = render
		self.state_modifier = ClassicModifier()
		self.zero_state = torch.from_numpy(env[0].reset())
		self.maxscore = 0
		self.total_tuple = 0

		assert('steps' and 'batch_size' in args)
		self.steps = args['steps']
		self.batch_size = args['batch_size']

	def get_replay_buffer(self, gamma, env):
		total_score, steps, n = 0, 0, 0
		replay_buffer = ReplayBuffer()
		state = self.state_modifier.apply(torch.from_numpy(env.reset()))
		while steps < self.steps:
			self._episodes += 1
			n += 1
			if n == 1: print("0 state value {}".format(self.critic.get_values(state)[0]))
			score = 0

			while True: # timelimits
#				if self.render: env.render()
				action = self.actor.get_action(state)
				next_state, reward, done, _ = env.step(action.numpy())
				next_state = self.state_modifier.apply(torch.from_numpy(next_state))
				
				#if tl == 1: reward += self.critic.get_values(next_state)[0] * gamma
				
				score += reward
				replay_buffer.append(state, action, reward, done == 1)
					
				state = next_state
				total_score, steps = total_score + reward, steps + 1
				if done == 1: break
		
		print("episodes: {}, score: {}, avg steps: {}, avg reward {}".format(self._episodes, total_score / n, steps / n, total_score / steps))
		return replay_buffer, total_score / n

	def get_replay_buffer_multicpu(self, gamma, env):
		cpu = len(env)
		total_score, steps, n = 0, 0, 0
		replay_buffer = ReplayBuffer()
		replay_buffer_tmp = [ReplayBuffer() for i in range(cpu)]
		state = [self.state_modifier.apply(torch.from_numpy(env[i].reset())) for i in range(cpu)]
		env_list = [e for e in env]
		print("0 state value {}".format(self.critic.get_values(self.state_modifier.modify(self.zero_state))))
#		std = torch.zeros(env[0].action_space.shape[0])

		while len(env_list) != 0:
			if self.render: env[0].render()
			#for s in state: std += self.actor.get_std(s)
			action = self.actor.get_action(torch.stack(state))
			next_state, reward, done, tl, _ = env[0].multistep(env_list, action)
			steps = steps + len(env_list)
			for i in range(len(env_list)-1, -1, -1):
				next_state[i] = self.state_modifier.apply(torch.from_numpy(next_state[i]))
				if tl[i] == 1: reward[i] += self.critic.get_values(next_state[i])[0] * gamma
				replay_buffer_tmp[i].append(state[i], action[i], reward[i], done[i] == 1)
				if done[i] == 1 and steps >= self.steps:
					replay_buffer.merge(replay_buffer_tmp[i])
					replay_buffer_tmp.pop(i)
					env_list.pop(i)
					next_state.pop(i)
			
				total_score += reward[i]
				if done[i] == 1: n += 1
			
			state = next_state
		
#		std = std / steps
		self.total_tuple += steps
		self._episodes += n
		print("episodes: {}, tuples: {}, score: {}, avg steps: {}, avg reward {}".format(self._episodes, self.total_tuple, total_score / n, steps / n, total_score / steps), flush=True)
		return replay_buffer, total_score / n

	def train(self):
		assert(False)

	def next_action(self, state):
		state = self.state_modifier.modify(state)
		return self.actor.get_action(state.cuda()).cpu()

	def next_action_nodist(self, state):
		state = self.state_modifier.modify(state)
		return self.actor.get_action_nodist(state.cuda()).cpu()

	def get_ckpt(self):
		ckpt = {'episodes' : self._episodes, 'total_tuple' : self.total_tuple, 'actor' : self.actor.get_ckpt(), \
			'critic' : self.critic.get_ckpt(), 'state_modifier' : self.state_modifier.get_ckpt(), 'maxscore' : self.maxscore}
		return ckpt

	def set_ckpt(self, ckpt):
		self._episodes = ckpt['episodes']
		self.total_tuple = ckpt['total_tuple']
		#self.maxscore = ckpt['maxscore']
		self.actor.set_ckpt(ckpt['actor'])
		self.critic.set_ckpt(ckpt['critic'])
		self.state_modifier.set_ckpt(ckpt['state_modifier'])

class PPOAgent(AgentInterface):
	def __init__(self, env, actor, critic, args, render):
		super(PPOAgent, self).__init__(env, actor, critic, args, render)
		assert('gamma' and 'lamda' in args)
		self.gamma = args['gamma']
		self.lamda = args['lamda']

	def train(self, train_step, cpu, name, stable_test=False, value_only=False):
		print("value_only : {}".format(value_only))
		for _ in range(train_step): # train step
			self._episodes += 1

			self.actor.eval()
			self.critic.eval()
			with torch.no_grad():

				if cpu == 1: replay_buffer, score = self.get_replay_buffer(self.gamma, self.env[0])
				else: replay_buffer, score = self.get_replay_buffer_multicpu(self.gamma, self.env)

				if stable_test:
					state = self.state_modifier.modify(self.env[0].reset())
					steps, total_score, episodes = 0, 0, 0
					while steps < self.steps:
						episodes += 1
						while True: # timelimits
							action = self.actor.get_action_nodist(state)
							next_state, reward, done, tl, _ = self.env[0].step(action)
							state = self.state_modifier.modify(next_state)
							total_score, steps = total_score + reward, steps + 1
							if done: break
					print("[stable test] episodes: {}, score: {}, avg steps: {}, avg reward {}".format(episodes, total_score / episodes, steps / episodes, total_score / steps), flush=True)
		
				if self.maxscore < score:
					print("saved at {}".format(name))
					self.maxscore = score
					torch.save(self.get_ckpt(), name)

				states, actions = replay_buffer.get_tensor()

				old_policy, _ = self.actor.evaluate(states, actions)
				old_values = self.critic.get_values(states)

				returns, advants = replay_buffer.get_gae(old_values, self.gamma, self.lamda) # gamma
			
			criterion = torch.nn.MSELoss()
			n = len(states)
			batch_size = self.batch_size
			
			self.actor.train()
			self.critic.train()
			torch.enable_grad()
	
			# TODO : to GPU
			actor_loss_total, critic_loss_total, step = 0, 0, 0
			for epoch in range(1):
				arr = torch.randperm(n)
				for i in range(n // batch_size):
					batch_index = arr[batch_size * i : batch_size * (i+1)]
					states_samples = states[batch_index]
					returns_samples = returns[batch_index]
					advants_samples = advants[batch_index]
					actions_samples = actions[batch_index]
					oldvalues_samples = old_values[batch_index]
					oldpolicy_samples = old_policy[batch_index]

					# surrogate function
					new_policy, entropy = self.actor.evaluate(states_samples, actions_samples)
					ratio = torch.exp(new_policy - oldpolicy_samples)
					loss = ratio * advants_samples

					# clip
					values = self.critic.get_values(states_samples)
					#clipped_values = oldvalues_samples + torch.clamp(values - oldvalues_samples, -0.2, 0.2) # clip param
					#critic_loss1 = criterion(clipped_values, returns_samples)
					critic_loss2 = criterion(values, returns_samples)
					#critic_loss = torch.max(critic_loss1, critic_loss2).mean()

					clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) # clip param
					clipped_loss = clipped_ratio * advants_samples
					actor_loss = -torch.min(loss, clipped_loss).mean()
					
					# profile
					actor_loss_total += actor_loss.data
					critic_loss_total += critic_loss2.data
					step += 1

					# merge loss function
					if value_only:
						loss = 0.5 * critic_loss2
						self.critic.zero_grad()
						loss.backward()
						self.critic.step()

					else:
						loss = actor_loss + 0.5 * critic_loss2# - 0.01*entropy.mean()

						self.actor.zero_grad()
						self.critic.zero_grad()
						loss.backward()
						self.actor.step()
						self.critic.step()

#					self.actor.apply_loss(loss, retain_graph=True)
#					self.critic.apply_loss(loss, retain_graph=False)
			for e in self.env: e.post_action()
		self.actor.eval()
		self.critic.eval()
		print("actor loss avg: {}, critic loss avg: {}".format(actor_loss_total / step, critic_loss_total / step), flush=True)

class VanilaAgent(AgentInterface):
	def __init__(self, env, actor, critic, args, render):
		super(VanilaAgent, self).__init__(env, actor, critic, args, render)
		assert('gamma' in args)
		self.gamma = args['gamma']
		
	def train(self, train_step):
		for _ in range(train_step): # train step
			self.actor.mode_eval()
			self.critic.mode_eval()

			replay_buffer = self.get_replay_buffer()
			states, actions = replay_buffer.get_tensor()

			returns = replay_buffer.get_returns(self.gamma) # gamma
		
			self.actor.mode_train()
			self.critic.mode_train()
			
			log_policy = self.actor.log_policy(states, actions)
			returns = returns.unsqueeze(1)
			loss = -(returns * log_policy).mean()
			
			self.actor.apply_loss(loss)
