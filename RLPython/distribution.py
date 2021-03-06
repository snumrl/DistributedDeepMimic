import torch
import torch.distributions as tdist
import math

class DistributionInterface():
    def get_scale(self, state):
        assert(False)

    def sample(self, mu, state):
        assert(False)
    
    def log_density(self, act, mu, state):
        assert(False)

    def train(self):
        pass

    def eval(self):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

    def set_ckpt(self, ckpt):
        pass

    def get_ckpt(self):
        return {}

class FixedGaussianDistribution(DistributionInterface):
    def __init__(self, dist, scale):
        self.dist = dist
        self.scale = scale

    def get_scale(self, state):
        return self.scale
    
    def sample(self, mu, state):
        return self.dist(mu, self.scale).sample()

    def log_prob(self, act, mu, state):
        return self.dist(mu, self.scale).log_prob(act)
    
    def entropy(self, mu, state):
        return self.dist(mu, self.scale).entropy()

    def set_ckpt(self, ckpt):
        assert('scale' in ckpt)
        self.scale = ckpt['scale']
        #if 'scale' in ckpt: self.scale = ckpt['scale']

    def get_ckpt(self):
        return {'scale' : self.scale}

class NetGaussianDistribution(DistributionInterface):
    def __init__(self, dist, network, opt, scale):
        self.dist = dist
        self.network = network
        self.opt = opt
        self.scale = scale

    def get_scale(self, state):
        res = torch.exp(self.network(state.cuda()).cpu()) * self.scale
        return res

    def sample(self, mu, state):
        return self.dist(mu, self.get_scale(state)).sample()

    def log_prob(self, act, mu, state):
        return self.dist(mu, self.get_scale(state)).log_prob(act)
    
    def entropy(self, mu, state):
        return self.dist(mu, self.get_scale(state)).entropy()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()

    def set_ckpt(self, ckpt):
        assert('netdist' and 'scale' in ckpt)
        self.network.load_state_dict(ckpt['netdist'])
        self.scale = ckpt['scale']

    def get_ckpt(self):
        return {'netdist' : self.network.state_dict(), \
            'scale' : self.scale}
