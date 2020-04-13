import tensorflow as tf
import numpy as np
import argparse
import gym, ray, os
from ray import tune
from ray.rllib.agents import ppo
from bipedenv.biped import DeepMimic

def argument_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_model', type=str, default=None)
	parser.add_argument('--train', type=int, default=1000)
	parser.add_argument('--render', default=False, action="store_true")
	parser.add_argument('--cpu', type=int, default=16)

	return parser.parse_args()

def set_config(args):
	global ppo_config, sac_config
	ppo_config = ppo.DEFAULT_CONFIG.copy()
	
	ppo_additional = {
		# === Settings for Rollout Worker processes ===
		"num_workers": args.cpu,
		"num_envs_per_worker": 1,
		"num_gpus": 1. / 16,

		# === Environment Settings ===
		"gamma": 0.97,
		"horizon": 1000, 
		"no_done_at_end": True,
		"env_config": {'render': args.render},  # config to pass to env class
		"lr": 5e-5,

		# === Settings for the Trainer process ===
		"model": {
			"fcnet_activation": "relu",
			"conv_activation": "relu",
			"fcnet_hiddens": [512, 512, 512],
		},

		"use_pytorch": True,
#		"eager": False,
#		"eager_tracing": False,
		
		# === PPO Settings ===
		"lambda": 0.95,
		"kl_coeff": 0.2,
    	"kl_target": 0.1,
#		"rollout_fragment_length": 20480 // args.cpu,
		"train_batch_size": 20480,
		"sgd_minibatch_size": 1024,
		"num_sgd_iter": 20,
		"clip_param": 0.3,
		"vf_clip_param": 100.0,
#		"observation_filter": "MeanStdFilter",
    	"entropy_coeff": 0.00,
	}

	for name, value in ppo_additional.items():
		ppo_config[name] = value

if __name__ == "__main__":
	args = argument_parse()
	set_config(args)
	ray.init()
	trainer = ppo.PPOTrainer(env=DeepMimic, config=ppo_config)
	
	if args.load_model: trainer.restore(args.load_model)

	for i in range(args.train):
		trainer.train()
		checkpoint = trainer.save("save_model/{}".format(i))
		print("checkpoint saved at {}".format(checkpoint))
