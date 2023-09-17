
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.a2c.utils import init
from rl.a2c.distributions import Categorical, CategoricalMasked 

from envs.action_spaces import DiscreteActionSpace 

class ActorCriticPolicy(nn.Module):

    def __init__(self, obs_shape, action_shape, args=None):
        super(ActorCriticPolicy, self).__init__()
        self.net_args = {'observation_shape': obs_shape,
                            'action_space': action_shape,
                            'activation_fn': args.actor_critic.activation, 
                            #'activation_fn_out': nn.Identity(), 
                            'hidden_dim': args.actor_critic.hidden_dim, 
                            'n_layers': args.actor_critic.n_layers,
                            }
        self.device = args.device
        self.obs_shape = obs_shape
        self.action_space = action_shape
        self.n_actions = action_shape if isinstance(action_shape, int) else action_shape.max_dim
        self.base = self.make_nets()
        #self.base.to(self.device)
        
        self.use_task_ids = False
        if isinstance(action_shape, int):
            self.dist = Categorical(self.base.output_size, self.n_actions)
        elif isinstance(action_shape, DiscreteActionSpace):
            self.dist = CategoricalMasked(self.base.output_size, self.n_actions, self.action_space)
            self.use_task_ids = True
        else:
            self.dist = DiagGaussian(self.base.output_size, self.num_outputs)
            self.use_task_ids = True
        self.dist.to(self.device)
            #self.dist = SquashedDiagGaussian(self.base.output_size, num_outputs)

    def make_nets(self):
        return MLPBase(self.net_args).to(self.device)

    def act(self, inputs, task_ids=None, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features) if task_ids is None else self.dist(actor_features, task_ids)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        #print(action)
        action_log_probs = dist.log_probs(action) if task_ids is None else dist.log_prob(action.view(-1))
        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action, task_ids=None):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features, task_ids) 

        action_log_probs = dist.log_probs(action) if task_ids is None else dist.log_prob(action.view(-1))
        dist_entropy = dist.entropy() #.mean()

        return value, action_log_probs, dist_entropy

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

"""
class GaussianPolicy(Policy):

    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, args=None):
        super(GaussianPolicy, self).__init__(obs_shape, action_space, base, base_kwargs, args)

    def act(self, inputs, task_ids, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features, task_ids) 
        mask = self.get_action_mask(task_ids)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        # mask action and action_log_probs
        #print('action: ', action)
        #print('torch.softmax(action): ', F.softmax(action, dim=-1))
        action = action * mask 
        #print('masked action: ', action)
        action_log_probs = dist.log_probs(action)
        #print('action_log_probs: ', action_log_probs)
        #print('action_log_probs * mask: ', action_log_probs * mask)
        #print()
        action_log_probs = torch.sum(action_log_probs * mask, dim=-1, keepdim=True)
        return value, action, action_log_probs    

    def evaluate_actions(self, inputs, task_ids, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features, task_ids) 
        mask = self.get_action_mask(task_ids)
        #print('in evaluate_actions, mask: ', mask)

        action_log_probs = dist.log_probs(action)
        #print('in eval, masked action: ', action)
        #print('in eval, action_log_probs: ', action_log_probs)
        #print('in eval, action_log_probs * mask: ', action_log_probs * mask)
        
        action_log_probs = torch.sum(action_log_probs * mask, dim=-1, keepdim=True)
        dist_entropy = dist.entropy() #.mean()
        #print('dist_entropy: ', dist_entropy)
        #print()
        dist_entropy = torch.sum(dist_entropy * mask, dim=-1)

        return value, action_log_probs, dist_entropy

    def get_action_mask(self, task_ids):
        #print(task_ids)
        mask = torch.arange(self.num_outputs, device=self.args.device)
        #print(mask)
        mask = mask.expand(len(task_ids), self.num_outputs) 
        #print(mask)
        mask = (mask <= task_ids).float() #.unsqueeze(1)
        #print(mask)
        return mask 
"""


class MLPBase(nn.Module):

    #def __init__(self, num_inputs, hidden_size=64, n_layers=2):
    def __init__(self, net_args):
        super(MLPBase, self).__init__()

        actor_modules, critic_modules = [], []
        self.in_dim = net_args['observation_shape']
        self.hidden_dim = net_args['hidden_dim']
        n_layers = net_args['n_layers']
        
        if net_args['activation_fn'] == 'relu':
            actitvation = nn.ReLU(inplace=True)
        elif net_args['activation_fn'] == 'leaky_relu':
            actitvation = nn.LeakyReLU(inplace=True)
        else:    
            actitvation = nn.Tanh()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        # Build actor
        in_dim_ = self.in_dim
        hidden_dim_ = self.hidden_dim
        for i in range(n_layers): 
            actor_modules.append(init_(nn.Linear(in_dim_, hidden_dim_)))
            actor_modules.append(actitvation)
            in_dim_ = hidden_dim_
        self.actor = nn.Sequential(*actor_modules)

        # Build critic
        in_dim_ = self.in_dim
        hidden_dim_ = self.hidden_dim
        for i in range(n_layers): 
            critic_modules.append(init_(nn.Linear(in_dim_, hidden_dim_)))
            critic_modules.append(actitvation)
            in_dim_ = hidden_dim_
        critic_modules.append(init_(nn.Linear(hidden_dim_, 1)))
        self.critic = nn.Sequential(*critic_modules)
        
        self.train()

    @property
    def output_size(self):
        return self.hidden_dim

    def forward(self, x):
        value = self.critic(x)
        hidden_actor = self.actor(x)
        return value, hidden_actor