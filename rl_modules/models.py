import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP, FourierMLP

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.
"""


# define the actor network
class actor(nn.Module):
    def __init__(self, args, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        if args.network_class == 'MLP':
            self.net = MLP(env_params['obs'] + env_params['goal'],
                           env_params['action'],
                           n_hidden=args.n_hidden,
                           hidden_dim=args.hidden_dim,
                           first_dim=args.first_dim)
        elif args.network_class == 'FourierMLP':
            self.net = FourierMLP(env_params['obs'] + env_params['goal'],
                                  env_params['action'],
                                  n_hidden=args.n_hidden,
                                  hidden_dim=args.hidden_dim,
                                  sigma=args.sigma,
                                  fourier_dim=args.fourier_dim,
                                  train_B=args.train_B,
                                  concatenate_fourier=args.concatenate_fourier)
        else:
            raise NotImplementedError

    def forward(self, x):
        actions = self.max_action * torch.tanh(self.net(x))
        return actions


class critic(nn.Module):
    def __init__(self, args, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        if args.network_class == 'MLP':
            self.net = MLP(env_params['obs'] + env_params['goal'] + env_params['action'],
                           1,
                           n_hidden=args.n_hidden,
                           hidden_dim=args.hidden_dim,
                           first_dim=args.first_dim)
        elif args.network_class == 'FourierMLP':
            self.net = FourierMLP(env_params['obs'] + env_params['goal'] + env_params['action'],
                                  1,
                                  n_hidden=args.n_hidden,
                                  hidden_dim=args.hidden_dim,
                                  sigma=args.sigma,
                                  fourier_dim=args.fourier_dim,
                                  train_B=args.train_B,
                                  concatenate_fourier=args.concatenate_fourier)
        else:
            raise NotImplementedError

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        q_value = self.net(x)
        return q_value
