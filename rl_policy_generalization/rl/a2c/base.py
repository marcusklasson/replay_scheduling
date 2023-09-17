
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py

# https://github.com/maximilianigl/rl-iter/blob/master/torch_rl/torch_rl/algos/base.py
import os
import shutil
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from rl.a2c.storage import RolloutStorage
from rl.a2c.utils import explained_variance
from training.logger import Logger

def from_numpy_to_torch(x, device):
    if not torch.is_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        else:
            x = torch.tensor([x]) 
    return x.to(device)

class BaseAlgo(object):

    def __init__(self, actor_critic, n_processes, n_steps, value_loss_coef, entropy_coef, discount, lr, max_grad_norm, 
                    use_gae=True, gae_lambda=0.95, use_proper_time_limits=False, device='cpu', log_dir='./'):                

        self.actor_critic = actor_critic
        self.n_processes = n_processes
        self.n_steps = n_steps
        self.obs_shape = actor_critic.obs_shape 
        self.action_shape = actor_critic.n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build rollout buffer
        self.rollouts = RolloutStorage(n_steps, 
                                num_processes=n_processes, 
                                obs_shape=self.obs_shape, 
                                action_shape=1) # action shape=1 because discrete
        #self.rollouts.to(self.device)

        # hyperparams
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.discount = discount
        #print(self.discount)
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae 
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits
        # other 
        self.logger = Logger(log_dir=log_dir)
        self.n_updates = 0

    def predict(self, obs, task_ids=None, deterministic=False):
        value, action, action_log_probs = self.actor_critic.act(obs, task_ids, deterministic)
        return value, action, action_log_probs

    def insert_rollout(self, obs, action, action_log_prob, value, reward, masks, task_ids=-1, bad_masks=-1):
        # 
        obs = from_numpy_to_torch(obs, device=self.device)
        reward = from_numpy_to_torch(reward, device=self.device)
        task_ids = from_numpy_to_torch(task_ids, device=self.device)
        bad_masks = from_numpy_to_torch(bad_masks, device=self.device)

        if self.n_processes > 1:
            action = action.unsqueeze(1)
            action_log_prob = action_log_prob.unsqueeze(1)

        # insert rollout to storage
        self.rollouts.insert(obs, action, action_log_prob, value, reward, task_ids, masks, bad_masks)

    def get_value(self, obs):
        value = self.actor_critic.get_value(obs)
        return value 

    def collect_experiences(self):

        #with torch.no_grad():
        next_value = self.actor_critic.get_value(self.rollouts.obs[-1]).detach()
        self.rollouts.compute_returns(next_value, gamma=self.discount, use_gae=self.use_gae,
                                 gae_lambda=self.gae_lambda, use_proper_time_limits=self.use_proper_time_limits)
        exps = self.rollouts 
        return exps

    def save_checkpoint(self, state, is_best, checkpoint_dir='./', filename='checkpoint.pth.tar'):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, filename))
        if is_best:
            shutil.copyfile(os.path.join(checkpoint_dir, filename),
                            os.path.join(checkpoint_dir, 'policy_best.pth.tar'))

    def load_checkpoint_simple(self, checkpoint_dir, filename):
        """ Load checkpoint for model and optimizer to trainer from file.
        """
        # Load q_nets
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        self.actor_critic.load_state_dict(checkpoint['ac_model_state_dict']) 

    ### Plotting utils
    def log_scalar(self, category, key, value, it):
        self.logger.add(category, key, value, it)

    def plot_scalar(self, category, key):
        # Get stats from category
        assert category in self.logger.stats.keys(), "Key {:s} doesn't exist in logger stats.".format(category)
        assert key in self.logger.stats[category].keys(), "Key {:s} doesn't exist in logger stats.".format(key)
        values = np.array(self.logger.stats[category][key])

        save_dir = self.logger.log_dir
        fix, ax = plt.subplots()
        x = np.arange(len(values))
        ax.plot(x, values, 'b-')
        ax.set_xlabel('iterations')
        ax.set_ylabel('{:s}/{:s}'.format(category, key))
        ax.set_title('Scalar value of {:s}/{:s}'.format(category, key))
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/{:s}-{:s}.png'.format(category, key))
        plt.close()

    def plot_returns(self, key, value):
        assert key in self.logger.stats.keys(), "Key {:s} doesn't exist in logger stats.".format(key) 
        values = np.array(self.logger.stats[key][value])
        if 'running_mean/%s' %value in self.logger.stats[key].values():
            run_means = np.array(self.logger.stats[key]['running_mean/%s' %value])

        save_dir = self.logger.log_dir
        fig, ax = plt.subplots()
        x = np.arange(1, len(values)+1)
        ax.plot(x, values, label='{:s}/{:s}'.format(key, value))
        if 'running_mean/%s' %value in self.logger.stats[key].values():
            ax.plot(x, run_means, label='Running mean')
        ax.legend()
        ax.set_xlabel('Time step')
        ax.set_ylabel(key)
        ax.set_title('Val. Env. {:s} {:s}'.format(key, value))        
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/val_{:s}_{:s}.png'.format(key, value))
        plt.close()

    def log_statistics(self, category, data, it, histogram=False):
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        # Add aggregated statistics
        self.logger.add(category, 'mean', np.mean(data), it)
        self.logger.add(category, 'std', np.std(data), it)
        # Add max/min
        self.logger.add(category, 'max', np.max(data), it)
        self.logger.add(category, 'min', np.min(data), it)
        # Add histogram option
        if histogram:
            self.logger.add_hist(data=data, class_name=category, it=it)

    def plot_aggregated_statistics(self, category):
        # Get stats from category
        assert category in self.logger.stats.keys(), "Key {:s} doesn't exist in logger stats.".format(category)
        means = np.array(self.logger.stats[category]['mean'])
        stds = np.array(self.logger.stats[category]['std'])
        maxs = np.array(self.logger.stats[category]['max'])
        mins = np.array(self.logger.stats[category]['min'])

        save_dir = self.logger.log_dir
        fix, ax = plt.subplots()
        x = np.arange(len(means))
        ax.plot(x, means, 'b-', label='mean')
        ax.fill_between(x, means+stds, means-stds, alpha=0.2)
        ax.plot(x, maxs, 'g-', label='max')
        ax.plot(x, mins, 'r-', label='min')
        ax.legend()
        ax.set_xlabel('batch updates')
        ax.set_ylabel(category)
        ax.set_title('Aggregated stats of {:s}'.format(category))
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/{:s}.png'.format(category))
        plt.close()

    def plot_gradient_statistics(self):
        # plot abs-max and mean square value of gradients
        assert 'grad_abs_max' in self.logger.stats.keys(), "Key {:s} doesn't exist in logger stats.".format('grad_abs_max')
        grad_abs_max = self.logger.stats['grad_abs_max']
        
        save_dir = self.logger.log_dir
        fix, ax = plt.subplots()
        for name, abs_max in grad_abs_max.items():
            x = np.arange(len(abs_max))
            ax.plot(x, abs_max, label=name)
        ax.legend()
        ax.set_xlabel('batch updates')
        ax.set_ylabel('max(|grad|)')
        ax.set_title('Absolute maximum of gradients')
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/grad_abs_max.png')
        plt.close() 

        assert 'grad_mean_sq' in self.logger.stats.keys(), "Key {:s} doesn't exist in logger stats.".format('grad_mean_sq')
        grad_mean_sq = self.logger.stats['grad_mean_sq']
        fix, ax = plt.subplots()
        for name, mean_sq_value in grad_mean_sq.items():
            x = np.arange(len(mean_sq_value))
            ax.plot(x, mean_sq_value, label=name)
        ax.legend()
        ax.set_xlabel('batch updates')
        ax.set_ylabel('mean(grad**2)')
        ax.set_title('Mean square value of gradients')
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/grad_mean_sqmean_sq_grad.png')
        plt.close()  