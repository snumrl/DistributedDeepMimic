import argparse
import gym, ray, os
from ray import tune
from ray.rllib.agents import ppo, sac
from bipedenv.biped import DeepMimic
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchMultiCategorical, TorchDiagGaussian

def argument_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_model', type=str, default=None)
	parser.add_argument('--train', type=int, default=0)
	parser.add_argument('--render', default=False, action="store_true")
	parser.add_argument('--cpu', type=int, default=1)

	return parser.parse_args()

def set_config(args):
	global ppo_config, sac_config
	ppo_config = {
		# === Settings for Rollout Worker processes ===
		"num_workers": args.cpu,
		"num_envs_per_worker": 1,
		"num_gpus": 1. / 16,

		# === Environment Settings ===
		"gamma": 0.97,
		"horizon": 200, 
		"no_done_at_end": True,
		"env_config": {'render': args.render},  # config to pass to env class
		"lr": 5e-5,

		# === Settings for the Trainer process ===
		"model": {
			"fcnet_activation": "relu",
			"conv_activation": "relu",
			"fcnet_hiddens": [512, 512, 512],
		},

		# === PPO Settings ===
		"lambda": 0.95,
		"train_batch_size": 20480,
		"sgd_minibatch_size": 1024,
		"num_sgd_iter": 1,
		"clip_param": 0.2,
		"observation_filter": "MeanStdFilter",
	}

	sac_config = {
		# === Settings for Rollout Worker processes ===
		"num_workers": args.cpu,
		"num_envs_per_worker": 1,

		# === Environment Settings ===
		"gamma": 0.97,
		"horizon": 200,
		"no_done_at_end": True,
		"env_config": {'render': args.render},  # config to pass to env class
		"lr": 5e-5,

		# === SAC Model ===
		"Q_model": {
			"hidden_activation": "relu",
			"hidden_layer_sizes": (512, 512),
		},
		"policy_model": {
			"hidden_activation": "relu",
			"hidden_layer_sizes": (512, 512, 512),
		},
		"normalize_actions": False,

		# === SAC Learning ===
		"timesteps_per_iteration": 8192,

		# === SAC Optimization ===
		"optimization": {
			"actor_learning_rate": 1e-4,
			"critic_learning_rate": 1e-4,
			"entropy_learning_rate": 1e-4,
		},
		"learning_starts": 8192,
		"train_batch_size": 8192,
	}

if __name__ == "__main__":
	args = argument_parse()
	set_config(args)
	ray.init()
#	trainer = sac.SACTrainer(env=DeepMimic, config=sac_config)
	trainer = ppo.PPOTrainer(env=DeepMimic, config=ppo_config)
#	tune.run(ppo.PPOTrainer, config=ppo_config) 
	
	if args.load_model: trainer.restore(args.load_model)

	for i in range(args.train):
		trainer.train()
		checkpoint = trainer.save("save_model/{}".format(i))
		print("checkpoint saved at {}".format(checkpoint))
