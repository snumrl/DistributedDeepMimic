import bipedenv.libbiped as lib
import numpy as np
import gym

def globalRender():
	lib.globalRender()

class DefaultEnv(gym.Env):
	def __init__(self, env_config):
		self.env = None
		self.render = env_config["render"]

	def reset(self):
		if self.env == None: self.env = lib.DeepMimicInit(self.render)
		return lib.reset(self.env)

	# double[:], double, bool, bool
	def step(self, action):
		state, reward, done, info = lib.step(self.env, action)
		if self.render: globalRender()
		return state, reward, done, info

	def post_action(self):
		pass

class DeepMimic(DefaultEnv):
	def __init__(self, env_config):
		super().__init__(env_config)
		self.name = "DeepMimic"
		self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(lib.DeepMimicActionSize(),), dtype=np.float32)
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(lib.DeepMimicObservationSize(),), dtype=np.float32)
		print(self.action_space)
		print(self.observation_space)
'''
	# double[cpu][:], double[cpu], int[cpu], int[cpu]
	def multistep(self, env, action):
		state, reward, done, tl, info = lib.multistep(np.array([e.env for e in env], dtype='i'), action.numpy())
		return [torch.from_numpy(s) for s in state], reward, done, tl, info
'''
