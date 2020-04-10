import torch
import numpy as np
import RLPython.RLHelper.libRLHelper as lib

# Replay Buffer do:
# store (state, action, reward, done) information (append)
# cap path (path_finish)
# get state, action list (get_tensor)
# calculate returns (get_returns)
# calculate gae function (get_gae)

class ReplayBuffer():
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []

    # store (state, action, reward, done) info
    # input: state(state)
    # action(action)
    # reward(float)
    # done(boolean)

    def append(self, state, action, reward, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(1 if done or done == 1 else 0)

    def merge(self, replay_buffer):
        self.state += replay_buffer.state
        self.action += replay_buffer.action
        self.reward += replay_buffer.reward
        self.done += replay_buffer.done
    
    # cap path
    def path_finish(self):
        self.done[-1] = True
   
    # return state, action list
    # return: states, actions
    # states(torch.Tensor(state array: size = [:][state])
    # actions(torch.Tensor(action array: size = [:][action])
    def get_tensor(self):
        return torch.stack(self.state), torch.stack(self.action)

    # return calculated returns
    # return: returns
    # returns(torch.Tensor(float array: size = [:][1])
    def get_returns(self, gamma):
        current_value = 0.
        prev_value = 0.
        returns = torch.zeros(len(self.reward))

        for t in reversed(range(len(self.reward))):
            prev_value = current_value if not self.done[t] else 0.
            
            current_value = self.reward[t] + gamma * prev_value
            returns[t] = current_value
        returns = returns.unsqueeze(1)
        return returns

    # return calculated gae function
    # return: returns, advants
    # returns(torch.Tensor(float array: size = [:][1])
    # advants(torch.Tensor(float array: size = [:][1])
    def get_gae(self, values, gamma, lamda):
        returns_np, advants_np = lib.getGAE(values.numpy(), np.array(self.reward), np.array(self.done, dtype='i'), gamma, lamda)
        return torch.from_numpy(returns_np).unsqueeze(1), torch.from_numpy(advants_np).unsqueeze(1)
