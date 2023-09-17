
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    """ weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

class DQNPolicy(nn.Module):
    def __init__(self, obs_shape, action_shape, args):
        super(DQNPolicy, self).__init__()
        self.net_args = {'observation_shape': obs_shape,
                            'action_space': action_shape,
                            'activation_fn': nn.ReLU(), #nn.LeakyReLU() if args.dqn.activation == 'leaky_relu' else nn.ReLU(),
                            #'orthogonal_init': True if args.dqn.orthogonal_init else False,
                            'activation_fn_out': nn.Identity(), #nn.Sigmoid() if (args.dqn.out_activation=='sigmoid') else nn.Identity(),
                            'hidden_dim': args.dqn.hidden_dim, #args.dqn.units,
                            'n_layers': args.dqn.n_layers,
                            }
        self.device = args.device
        self.action_space = action_shape
        self.n_actions = action_shape if isinstance(action_shape, int) else action_shape.max_dim
            
        #torch.manual_seed(args.seed)
        """
        if args.dueling_dqn:
            self.q_net = self.make_dueling_q_net()
            self.q_net_target = self.make_dueling_q_net()
        else:
            self.q_net = self.make_q_net()
            self.q_net_target = self.make_q_net()
        """
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict()) # init target net
        #print(self.q_net)

    def make_q_net(self):
        return QNetwork(self.net_args).to(self.device)

    def make_dueling_q_net(self):
        return DuelingQNetwork(self.net_args).to(self.device)

    def forward(self, obs, tasks):
        return self._predict(obs, tasks)
    
    def predict(self, obs, tasks):
        return self.q_net._predict(obs, tasks)


class QNetwork(nn.Module):
    def __init__(self, net_args):
        super(QNetwork, self).__init__()
        lower_modules = []
        in_dim = net_args['observation_shape']
        hidden_dim = net_args['hidden_dim']
        for i in range(net_args['n_layers']): 
            lower_modules.append(nn.Linear(in_dim, hidden_dim))
            lower_modules.append(net_args['activation_fn'])
            in_dim = hidden_dim
        self.net = nn.Sequential(*lower_modules)

        self.action_space = net_args['action_space']
        out_dim = self.action_space if isinstance(self.action_space, int) else self.action_space.max_dim
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.activation_fn_out = net_args['activation_fn_out']

        #if net_args['orthogonal_init']:
        #    self.apply(weight_init)

    def features(self, x):
        x = self.net(x)
        return x

    def q_values(self, x):
        x = self.linear(x)
        x = self.activation_fn_out(x)
        return x

    def forward(self, x, t=None):
        x = self.features(x)
        x = self.q_values(x)
        if t is not None:
            x = self.select_valid_outputs(x, t)
        return x

    def select_valid_outputs(self, output, t):
        #https://github.com/imirzadeh/CL-Gym/blob/main/cl_gym/backbones/base.py
        for i, tt in enumerate(t):
            offset = self.action_space.get_dim_at_task(tt)
            output[i, offset:].data.fill_(-10e10)
        return output

    def _predict(self, obs, t):
        q_values = self.forward(obs, t)
        # Greedy action
        actions = torch.argmax(q_values, dim=1).view(-1)
        return actions

"""
class DuelingQNetwork(nn.Module):
    def __init__(self, net_args):
        super(DuelingQNetwork, self).__init__()
        lower_modules = []
        in_dim = net_args['observation_shape']
        hidden_dim = net_args['hidden_dim']
        self.action_space = net_args['action_space']
        out_dim = self.action_space.max_dim
        self.activation_fn_out = net_args['activation_fn_out']

        # Build network
        for i in range(net_args['n_layers']-1): 
            lower_modules.append(nn.Linear(in_dim, hidden_dim))
            lower_modules.append(net_args['activation_fn'])
            in_dim = hidden_dim
        self.net = nn.Sequential(*lower_modules)

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            net_args['activation_fn'],
            nn.Linear(hidden_dim, out_dim),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            net_args['activation_fn'],
            nn.Linear(hidden_dim, 1),
        )        
        
    def features(self, x):
        x = self.net(x)
        return x

    def q_values(self, x, t):
        v = self.value(x)
        #v = self.activation_fn_out(v)
        adv = self.advantage(x)
        adv, adv_mean = self.select_valid_advantages(adv, t)
        q = v + adv - adv_mean
        q = self.activation_fn_out(q)
        return q

    def forward(self, x, t=None):
        x = self.features(x)
        q_values = self.q_values(x, t)
        if t is not None:
            q_values = self.select_valid_outputs(q_values, t)
        return q_values

    def select_valid_outputs(self, output, t):
        #https://github.com/imirzadeh/CL-Gym/blob/main/cl_gym/backbones/base.py
        for i, tt in enumerate(t):
            offset = self.action_space.get_dim_at_task(tt)
            output[i, offset:].data.fill_(-10e10)
        return output

    def select_valid_advantages(self, adv, t):
        adv_mean = torch.zeros(adv.size(0), dtype=torch.float32, device=adv.get_device())
        for i, tt in enumerate(t):
            offset = self.action_space.get_dim_at_task(tt)
            adv[i, offset:].data.fill_(-10e10)
            adv_mean[i] = torch.sum(adv[i, :offset]) / float(offset)
        return adv, adv_mean.unsqueeze(-1)

    def _predict(self, obs, t):
        q_values = self.forward(obs, t)
        # Greedy action
        actions = torch.argmax(q_values, dim=1).view(-1)
        return actions
"""