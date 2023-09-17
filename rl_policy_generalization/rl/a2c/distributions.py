
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import einsum

from rl.a2c.utils import init

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, action_space=None):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.logits = init_(nn.Linear(num_inputs, num_outputs))
        
        if action_space is not None:
            self.action_space = action_space # used for selecting valid outputs

    def forward(self, x, t=None):
        x = self.logits(x)
        if t is not None:
            x = self.select_valid_outputs(x, t)
        return FixedCategorical(logits=x)

    def select_valid_outputs(self, output, t):
        #https://github.com/imirzadeh/CL-Gym/blob/main/cl_gym/backbones/base.py
        for i, tt in enumerate(t):
            offset = self.action_space.get_dim_at_task(tt)
            output[i, offset:].data.fill_(-10e10)
        return output

##
class FixedCategoricalMasked(torch.distributions.Categorical):

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(FixedCategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(FixedCategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        #print(p_log_p)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -torch.sum(p_log_p, dim=-1) #-reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class CategoricalMasked(nn.Module):
    def __init__(self, num_inputs, num_outputs, action_space=None):
        super(CategoricalMasked, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.logits = init_(nn.Linear(num_inputs, num_outputs))
        self.num_outputs = num_outputs
        
        if action_space is not None:
            self.action_space = action_space # used for selecting valid outputs

    def forward(self, x, t=None):
        x = self.logits(x)
        #print(t)
        mask = self.get_action_mask(t) if t is not None else None
        #if t is not None:
        #    x = self.select_valid_outputs(x, t)
        return FixedCategoricalMasked(logits=x, mask=mask)

    def get_action_mask(self, task_ids):
        nb_actions = [self.action_space.get_dim_at_task(tt) for tt in task_ids]
        nb_actions = torch.tensor(nb_actions, device=task_ids.device)
        mask = torch.arange(self.num_outputs, device=task_ids.device)
        mask = mask.expand(len(nb_actions), self.num_outputs) < nb_actions.unsqueeze(1)
        return mask

    #def select_valid_outputs(self, output, t):
    #    #https://github.com/imirzadeh/CL-Gym/blob/main/cl_gym/backbones/base.py
    #    for i, tt in enumerate(t):
    #        offset = self.action_space.get_dim_at_task(tt)
    #        output[i, offset:].data.fill_(-10e10)
    #    return output

# Normal
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        #return super().log_prob(actions).sum(-1, keepdim=True)
        return super().log_prob(actions)

    def entropy(self):
        #return super().entropy().sum(-1)
        return super().entropy()

    def mode(self):
        return self.mean

    def sample(self):
        return super().rsample()

class DiagGaussian(nn.Module):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.
    :param action_dim:  Dimension of the action space.
    """
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        #self.mean_actions = nn.Linear(num_inputs, num_outputs)
        self.mean_actions = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.zeros(num_outputs), requires_grad=True)

    def forward(self, x, t):
        action_mean = self.mean_actions(x)
        action_std = torch.ones_like(action_mean) * self.log_std.exp()
        #action_mean, action_std = self.select_valid_outputs(action_mean, action_std, t)
        return FixedNormal(action_mean, action_std)
    
    def select_valid_outputs(self, means, stds, t):
        #https://github.com/imirzadeh/CL-Gym/blob/main/cl_gym/backbones/base.py
        for i, tt in enumerate(t):
            offset = tt+1
            means[i, offset:].data.fill_(-1e10)
            #stds[i, offset:].data.fill_(1.0) # set to 1 which is suitable when computing the entropy because of the log det(sigmas)
            stds[i, offset:].data.fill_(1e-8)
        return means, stds
    

class FixedSquashedNormal(torch.distributions.Normal):
    def __init__(self, means, stds, epsilon=1e-6):
        super(FixedSquashedNormal, self).__init__(means, stds)
        self.epsilon = epsilon 

    def log_probs(self, actions, task_ids):
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        # It will be clipped to avoid NaN when inversing tanh
        gaussian_actions = TanhBijector.inverse(actions)

        # use mask to only compute log_prob with legal action outputs 
        masks = torch.ones_like(actions)
        for i, tt in enumerate(task_ids):
            masks[i, tt:].data.fill_(0.0)

        # Log likelihood for a Gaussian distribution
        log_prob = (super().log_prob(gaussian_actions) * masks).sum(-1)
        #print('log_prob: ', log_prob)
        
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - actions**2 + self.epsilon)*masks, dim=1)
        return log_prob

    def entropy(self):
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def mode(self):
        gaussian_actions = self.mean
        return torch.tanh(gaussian_actions)

    def sample(self):
        gaussian_actions = super().rsample()
        return torch.tanh(gaussian_actions)

class SquashedDiagGaussian(DiagGaussian):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.
    :param action_dim:  Dimension of the action space.
    """
    def __init__(self, num_inputs, num_outputs, log_std_init=0.0, epsilon=1e-6):
        super(SquashedDiagGaussian, self).__init__(num_inputs, num_outputs)
        self.epsilon = epsilon

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        #self.mean_actions = nn.Linear(num_inputs, num_outputs)
        self.mean_actions = init_(nn.Linear(num_inputs, num_outputs))
        #self.log_std = nn.Parameter(torch.ones(num_outputs)*log_std_init, requires_grad=True)
        self.log_std = nn.Parameter(-0.5*torch.ones(num_outputs))

    def forward(self, x, t):
        action_mean = self.mean_actions(x)
        action_std = torch.ones_like(action_mean) * self.log_std.exp()
        action_mean, action_std = self.select_valid_outputs(action_mean, action_std, t)
        #print('in forward: ', action_mean, action_std)
        return FixedSquashedNormal(action_mean, action_std)

    def select_valid_outputs(self, means, stds, t):
        #https://github.com/imirzadeh/CL-Gym/blob/main/cl_gym/backbones/base.py
        #print(t)
        for i, tt in enumerate(t):
            offset = tt+1 # increment by one since the input task_id starts from 0 in this function
            #print('offset: ', offset)
            #print('mean: ', means[i])
            means[i, offset:].data.fill_(-10) # enough to set illegal means to -10 for Tanh activation
            stds[i, offset:].data.fill_(1e-8)
        return means, stds

# https://github.com/DLR-RM/stable-baselines3/blob/e24147390d2ce3b39cafc954e079d693a1971330/stable_baselines3/common/distributions.py
class TanhBijector(object):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super(TanhBijector, self).__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of Tanh
        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        """
        Inverse tanh.
        :param y:
        :return:
        """
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        # Squash correction (from original SAC implementation)
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)