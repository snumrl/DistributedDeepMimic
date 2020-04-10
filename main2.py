import torch
import torch.optim as optim
import torch.nn.functional as F

from RLPython import *
from bipedenv.biped import *
import os, argparse
from time import sleep
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *

model_path = os.path.join(os.getcwd(),'save_model')

def argument_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_model', type=str, default=None)
	parser.add_argument('--load_model2', type=str, default=None)
	parser.add_argument('--supervised_model', type=str, default=None)
	parser.add_argument('--train', type=int, default=100)
	parser.add_argument('--period', type=int, default=1)
	parser.add_argument('--name', type=str, default='zigui')
	parser.add_argument('--env', type=str, default='DeepMimic')
	parser.add_argument('--render', default=False, action="store_true")
	parser.add_argument('--value_only', default=False, action="store_true")
#	parser.add_argument('--actor_only', default=False, action="store_true")
	parser.add_argument('--cpu', type=int, default=16)
	parser.add_argument('--std', type=float, default=1)
	parser.add_argument('--dist', type=str, default="Fixed")

	return parser.parse_args()

def PPO_critic_with_param(env, critic_network_size, critic_lr, critic_decay):
	o_size = env[0].observation_space.shape[0]
	
	critic_network = deepnetwork.CNN([o_size] + critic_network_size + [1], "critic").cuda()
	critic_opt = optim.Adam(critic_network.parameters(), lr=critic_lr, weight_decay=critic_decay)

	return model.Critic(critic_network, critic_opt)

def PPO_agent_with_param(env, actor_network_size, actor_lr, critic_network_size, critic_lr, critic_decay, rl_arg, args, zeroInit=False):
	o_size = env[0].observation_space.shape[0]
	a_size = env[0].action_space.shape[0]
	
	actor_network = deepnetwork.CNN([o_size] + actor_network_size + [a_size], "actor", zeroInit).cuda()
	actor_opt = optim.Adam(actor_network.parameters(), lr=actor_lr)
	
	if args.dist == "Fixed": 
		dist = distribution.FixedGaussianDistribution(torch.distributions.normal.Normal, torch.ones(a_size) * args.std)
	elif args.dist == "Net":
		dist_network = deepnetwork.CNN([o_size] + actor_network_size + [a_size], "actor", True).cuda()
		dist_opt = optim.Adam(dist_network.parameters(), lr=1e-5)
		dist = distribution.NetGaussianDistribution(torch.distributions.normal.Normal, dist_network, dist_opt, args.std)
	
	actor = model.Actor(actor_network, actor_opt, dist)

	critic = PPO_critic_with_param(env, critic_network_size, critic_lr, critic_decay)
	
	agent = policy.PPOAgent(env, actor, critic, rl_arg, args.render)
	return agent

def train(agent, train_step, cpu, env_name, args_name, value_only = False):
	for i in range(train_step):
#		print("dist : {}".format(agent.actor.dist.scale))
		agent.train(5, args.cpu, os.path.join(model_path, 'ckpt_{}_{}_max'.format(env_name, args_name)), value_only = value_only)
		torch.save(agent.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_{}'.format(env_name, args_name, i)))
		torch.save(agent.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_latest'.format(env_name, args_name)))
#		if agent.actor.dist.scale[0] >= 0.01: agent.actor.dist.scale *= 0.99

def demo(env, agent, render = 1):
	state = env.reset()
	t = 0
	while True:
		if t%render == 0: env.render()
		t += 1

		action = agent.next_action_nodist(state)
		state, reward, done, _, _ = env.step(action)
		if done:
			print("END")

def train_DReCon(args):
	env = [DReCon(args.render if i == 0 else False, False) for i in range(args.cpu)]
	agent = PPO_agent_with_param(env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':0.97, 'lamda':0.95, 'steps':20480, 'batch_size':1024}, args)
	if args.load_model is not None: agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	train(agent, args.train, args.cpu, env[0].name, args.name)
	if args.render: demo(env[0], agent)

def train_deepmimic(args):
	env = [DeepMimic({"render" : args.render} if i == 0 else {"render" : False}) for i in range(args.cpu)]
	agent = PPO_agent_with_param(env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':0.97, 'lamda':0.95, 'steps':20480, 'batch_size':1024}, args)
	if args.load_model is not None: agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	train(agent, args.train, args.cpu, env[0].name, args.name)
	if args.render: demo(env[0], agent)

def train_unstabledeepmimic(args):
	env = [UnstableDeepMimic(args.render if i == 0 else False) for i in range(args.cpu)]
	agent = PPO_agent_with_param(env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':0.97, 'lamda':0.95, 'steps':20480, 'batch_size':1024}, args)
	if args.load_model is not None: agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	train(agent, args.train, args.cpu, env[0].name, args.name)
	if args.render: demo(env[0], agent)

def train_deepmimictorque(args):
	torque_env = [DeepMimicTorque(args.render if i == 0 else False) for i in range(args.cpu)]
	torque_agent = PPO_agent_with_param(torque_env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':(1 - 0.03/20), 'lamda':(1 - 0.05/20), 'steps':20480 * 10, 'batch_size':1024 * 10}, args)

	#torque_agent.state_modifier = DefaultModifier()

	if args.supervised_model is not None:
		spd_env = [DeepMimic(False) for i in range(args.cpu)]
		for env in spd_env: env.setRecord(True)
		spd_agent = PPO_agent_with_param(spd_env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
			{'gamma':0.97, 'lamda':0.95, 'steps':20480, 'batch_size':1024}, args.render)
		spd_agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.supervised_model)))

		#torque_agent.critic = spd_agent.critic
		#torque_agent.state_modifier = spd_agent.state_modifier

		state = [spd_agent.state_modifier.modify(spd_env[i].reset()) for i in range(args.cpu)]
		idx = [spd_env[i].env for i in range(args.cpu)]
		
		criterion = torch.nn.MSELoss()
		
		for i in range(30):
			state_buffer = []
			action_buffer = []
			for tt in range(10240 // 20 // args.cpu * 100):
				action = spd_agent.actor.get_action_nodist(torch.stack(state))
				next_state, _, _, _, _ = spd_env[0].multistep(np.array(idx, dtype='i'), action)
				next_state = [spd_agent.state_modifier.modify(state) for state in next_state]
				for env in spd_env:
					s, a = env.getRecordedStateAction()
					state_buffer.append(s)
					action_buffer.append(a)
					state = next_state
				if tt%100 == 0: print(tt)
			
			loss = 0
			state_buffer = torque_agent.state_modifier.apply_list(torch.cat(state_buffer)).cuda()
			action_buffer = torch.cat(action_buffer).cuda()
			print(action_buffer)
			for _ in range(100):
				arr = torch.randperm(state_buffer.shape[0]).cuda()
				for i in range(100):
					batch_idx = arr[i*10240 : (i+1)*10240]
					action = torque_agent.actor.network(state_buffer[batch_idx]).cuda()
					loss = criterion(action, action_buffer[batch_idx])
			
					torque_agent.actor.zero_grad()
					loss.backward()
					torque_agent.actor.step()
				print(loss)
			torch.save(torque_agent.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_supervised'.format(torque_env[0].name, args.name)))
	
	if args.load_model is not None: torque_agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))

	for i in range(args.train):
		torque_agent.train(10, args.cpu, os.path.join(model_path, 'ckpt_{}_{}_max'.format(torque_env[0].name, args.name)))
		torch.save(torque_agent.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_{}'.format(torque_env[0].name, args.name, i)))
		torch.save(torque_agent.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_latest'.format(torque_env[0].name, args.name)))

	if args.render:
#		demo(torque_env[0], torque_agent)
		state = torque_env[0].reset()
		while True:
			torque_env[0].render()

			action = torque_agent.next_action_nodist(state)
			state, reward, done, _, _ = torque_env[0].step(action)
			if done:
				print("END")
				state = torque_env[0].reset()
		
def MotionMatching(args):
	env = Kinematics(args.render)
	while True:
		env.step()
		env.render()

def test_torque(args):
	spd_env = DeepMimic(False)
	torque_env = DeepMimicTorque(args.render)
	spd_agent = PPO_agent_with_param([spd_env], [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':0.97, 'lamda':0.95, 'steps':20480, 'batch_size':1024}, args)
	if args.load_model is not None: spd_agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	
	torque_env.reset()
	while True:
		torque_env.render()
		state = spd_env.sync_with_env(torque_env)
		spd_action = spd_agent.next_action_nodist(state)
		spd_env.step(spd_action)
		torque_action = spd_env.lastAvgTorque()
		state, reward, done, _, _ = torque_env.step(torque_action)
		if done:
			print("END")
			torque_env.reset()

def train_spd(args):
	spd_env = [DeepMimic(False)]
	spd_agent = PPO_agent_with_param(spd_env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':0.97, 'lamda':0.95, 'steps':20480, 'batch_size':1024}, False)
	spd_agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.supervised_model)))
	torque_env = [DeepMimicSPD(args.render if i == 0 else False, spd_agent) for i in range(args.cpu)]
	torque_agent = PPO_agent_with_param(torque_env, [512, 512, 512], 5e-4, [512, 512], 1e-3, 7e-4, \
		{'gamma':(1 - 0.03/20), 'lamda':(1 - 0.05/20), 'steps':4096, 'batch_size':128}, args, True)
	if args.load_model is not None: torque_agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	
	for i in range(args.train):
		torque_agent.train(10, args.cpu, os.path.join(model_path, 'ckpt_{}_{}_max'.format(torque_env[0].name, args.name)), stable_test=True)
		torch.save(torque_agent.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_{}'.format(torque_env[0].name, args.name, i)))
		torch.save(torque_agent.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_latest'.format(torque_env[0].name, args.name)))
	
	env = torque_env[0]
	agent = torque_agent
	if args.render:
		state = env.reset()
		while True:
			env.render()

			action = agent.next_action_nodist(state)
			state, reward, done, _, _ = env.step(action)
			print(reward)
			print(action)
			if done:
				print("END")

def train_slow(args):
	env = [SlowDeepMimic(args.render if i == 0 else False) for i in range(args.cpu)]
	agent = PPO_agent_with_param(env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':(1 - 0.03/20), 'lamda':(1 - 0.05/20), 'steps':20480 * 10, 'batch_size':1024 * 10}, args)
	if args.load_model is not None: agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	train(agent, args.train, args.cpu, env[0].name, args.name)
	if args.render: demo(env[0], agent)

def train_slowphase(args):
	env = [SlowPhaseDeepMimic(args.render if i == 0 else False) for i in range(args.cpu)]
	agent = PPO_agent_with_param(env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':(1 - 0.03/20), 'lamda':(1 - 0.05/20), 'steps':20480 * 10, 'batch_size':1024 * 10}, args)
	if args.load_model is not None: agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	train(agent, args.train, args.cpu, env[0].name, args.name)
	if args.render: demo(env[0], agent, 20)

def train_phaseshift(args):
	env = [PhaseShiftDeepMimic(args.render if i == 0 else False) for i in range(args.cpu)]
	agent = PPO_agent_with_param(env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':(1 - 0.03/20), 'lamda':(1 - 0.05/20), 'steps':20480 * 10, 'batch_size':1024 * 10}, args)
	if args.load_model is not None: agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	#if args.actor_only: agent.critic = PPO_critic_with_param(env, [512, 512], 1e-4, 7e-4)
	train(agent, args.train, args.cpu, env[0].name, args.name, args.value_only)
	if args.render: demo(env[0], agent, 20)

def my_demo(args):
	env = [PhaseShiftDeepMimic(True), DeepMimic(True)]
	repeat = [20, 1]

	agent = [ \
		PPO_agent_with_param([env[0]], [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
			{'gamma':(1 - 0.03/20), 'lamda':(1 - 0.05/20), 'steps':20480 * 10, 'batch_size':1024 * 10}, args.render), \
		PPO_agent_with_param([env[1]], [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
			{'gamma':0.97, 'lamda':0.95, 'steps':20480, 'batch_size':1024}, args.render)]
	
	if args.load_model is not None: agent[0].set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	if args.load_model2 is not None: agent[1].set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model2)))

	state = [e.reset() for e in env]
	while True:
		env[0].render()
		for i in range(len(env)):
			for _ in range(repeat[i]):
				action = agent[i].next_action_nodist(state[i])
				state[i], reward, done, _, _ = env[i].step(action)
				if done: state[i] = env[i].reset()

def train_dropoutdeepmimic(args):
	env = [DropoutDeepMimic(args.render if i == 0 else False) for i in range(args.cpu)]
	agent = PPO_agent_with_param(env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':0.97, 'lamda':0.95, 'steps':20480, 'batch_size':1024}, args)
	if args.load_model is not None: agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	train(agent, args.train, args.cpu, env[0].name, args.name)
	if args.render: demo(env[0], agent)

'''
def train_hiro(args):
	env = [HIROSlowPhaseDeepMimic(args.render if i == 0 else False) for i in range(args.cpu)]
	for e in env: e.lower()
	agent_lo = PPO_agent_with_param(env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':1, 'lamda':1, 'steps':20480 * 10, 'batch_size':1024 * 10}, args.render)
	if args.load_model is not None: agent_lo.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	for e in env: e.model_lower = agent_lo
	
	agent_hi = PPO_agent_with_param(env, [512, 512, 512], 5e-5, [512, 512], 1e-4, 7e-4, \
		{'gamma':1 - 0.03, 'lamda':1 - 0.05, 'steps':20480, 'batch_size':1024}, args.render)
	if args.load_model2 is not None: agent_hi.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model2)))
	for e in env: e.model_upper = agent_hi

	for i in range(args.train_step):
		for e in env: e.lower()
		agent_lo.train(args.period, args.cpu, os.path.join(model_path, 'ckpt_{}_{}_max'.format(env[0].name, args.name)))
		torch.save(agent_lo.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_{}'.format(env[0].name, args.name, i)))
		torch.save(agent_lo.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_latest'.format(env[0].name, args.name)))
		
		for e in env: e.upper()
		agent_hi.train(args.peroid, args.cpu, os.path.join(model_path, 'ckpt_{}_{}_max'.format(env[0].name, args.name)))
		torch.save(agent_hi.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_{}'.format(env[0].name, args.name, i)))
		torch.save(agent_hi.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_latest'.format(env[0].name, args.name)))
'''

if __name__ == "__main__":
	args = argument_parse()
	if args.env == "MM": MotionMatching(args)
	elif args.env == "DeepMimic": train_deepmimic(args)
	elif args.env == "UnstableDeepMimic": train_unstabledeepmimic(args)
	elif args.env == "Torque": train_deepmimictorque(args)
	elif args.env == "DReCon": train_DReCon(args)
	elif args.env == "TestTorque": test_torque(args)
	elif args.env == "Micro" : train_micro(args)
	elif args.env == "SPD" : train_spd(args)
	elif args.env == "Slow" : train_slow(args)
	elif args.env == "SlowPhase" : train_slowphase(args)
	elif args.env == "PhaseShift" : train_phaseshift(args)
	elif args.env == "MyDemo" : my_demo(args)
	elif args.env == "DropoutDeepMimic": train_dropoutdeepmimic(args)
#	elif args.env == "HIRO" : train_hiro(args) # https://arxiv.org/abs/1805.08296 but not exactly same
