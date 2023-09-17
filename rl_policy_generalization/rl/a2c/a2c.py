# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py
#import os
import numpy as np 
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from rl.a2c.base import BaseAlgo

class A2C(BaseAlgo):

    def __init__(self, actor_critic, n_processes, n_steps, value_loss_coef=0.5, entropy_coef=0.01, 
                    discount=0.99, lr=7e-4, max_grad_norm=0.5, use_gae=True, gae_lambda=0.95, use_proper_time_limits=False, 
                    rmsprop_alpha=0.99, rmsprop_eps=1e-5, device='cpu', log_dir='./'):
        
        super().__init__(actor_critic, n_processes, n_steps, value_loss_coef, entropy_coef, discount, lr, 
                            max_grad_norm, use_gae, gae_lambda, use_proper_time_limits, device, log_dir)

        self.optimizer = torch.optim.RMSprop(self.actor_critic.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self):
        
        # prepare rollouts for training
        rollouts = self.collect_experiences()
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        logs = {}

        # Initialize update values
        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        #print('task_ids: ', rollouts.task_ids[:-1].view(-1))
        # compute loss
        if self.actor_critic.use_task_ids:
            values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.actions.view(-1, action_shape),
                rollouts.task_ids[:-1].view(-1, 1)-1)
        else:
            values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.actions.view(-1, action_shape))
        # reshape and get advantages
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        advantages = rollouts.returns[:-1] - values # discard last return value
        # sum losses 
        entropy_loss = dist_entropy.mean()

        policy_loss = -(advantages.detach() * action_log_probs).mean()

        value_loss = advantages.pow(2).mean()

        loss = policy_loss + value_loss * self.value_loss_coef - entropy_loss * self.entropy_coef

        # Update batch values
        update_entropy += entropy_loss.item()
        update_value += values.mean().item()
        update_policy_loss += policy_loss.item()
        update_value_loss += value_loss.item()
        update_loss += loss

        # Update actor-critic
        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.actor_critic.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # set latest obs, action as first in rollout storage 
        self.rollouts.after_update()
        self.n_updates += 1

        # Log some values
        logs["entropy"] = update_entropy
        logs["value"] = update_value
        logs["policy_loss"] = update_policy_loss
        logs["value_loss"] = update_value_loss
        logs["grad_norm"] = update_grad_norm

        return logs

"""
class A2C():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 args=None):

        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(actor_critic.parameters(), lr)
        self._n_updates = 0
        self.logger = Logger(log_dir=args.log_dir,
                        monitoring=args.monitoring,
                        monitoring_dir=os.path.join(args.log_dir, 'monitoring', 'a2c'))
        self.args = args

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        #print('task_ids: ', rollouts.task_ids[:-1].view(-1))

        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.task_ids[:-1].view(-1, 1)-1,
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values # discard last return value
        if self.args.verbose > 1:
            print('action_log_probs: ', action_log_probs.view(-1))
            print('returns: ', rollouts.returns[:-1].view(-1))
            print('values: ', values.view(-1))
            print('advantages: ', advantages.view(-1))
            print()
        value_loss = advantages.pow(2).mean()

        # policy gradient loss
        pg_loss = -(advantages.detach() * action_log_probs).mean()
        # entropy bonus
        if dist_entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = torch.mean(-action_log_probs)
            if entropy_loss < 0:
                print('action_log_probs: ', action_log_probs)
            # log probabilities for continuous values can be positive
            #assert entropy_loss >= 0.0, "entropy {} must be positive!".format(entropy_loss)
        else:
            entropy_loss = torch.mean(dist_entropy)

        self.optimizer.zero_grad()
        loss = pg_loss + value_loss * self.value_loss_coef - entropy_loss * self.entropy_coef
        loss.backward()

        
        print('Trying to print gradients: ')
        print(self.actor_critic.dist)
        print('means: ', self.actor_critic.dist.mean_actions)
        #print('mean weights: ', self.actor_critic.dist.mean_actions.weight)
        print('mean gradients: ', self.actor_critic.dist.mean_actions.weight.grad)
        print('log_std: ', self.actor_critic.dist.log_std)
        print('log_std gradients: ', self.actor_critic.dist.log_std.grad)
        print()
        

        if self.max_grad_norm:
            #print('clip gradients!!')
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Log results
        self._n_updates += 1

        if (self._n_updates % self.args.log_freq == 0):
            explained_var = explained_variance(y_pred=values.detach().flatten().cpu().numpy(), 
                                                y_true=rollouts.returns[:-1].flatten().cpu().numpy())
            # Log results
            self.logger.log_scalar('train', 'value_loss', value_loss, self._n_updates)
            self.logger.log_scalar('train', 'policy_loss', pg_loss, self._n_updates)
            self.logger.log_scalar('train', 'entropy_loss', entropy_loss, self._n_updates)
            self.logger.log_scalar('train', 'loss', loss.item(), self._n_updates)
            self.logger.log_scalar('train', 'explained_variance', explained_var, self._n_updates)

            self.log_statistics(category='advantages', data=advantages.detach().flatten().cpu().numpy(), it=self._n_updates)
            self.log_statistics(category='value_preds', data=rollouts.value_preds.detach().flatten().cpu().numpy(), it=self._n_updates)
            self.log_statistics(category='returns', data=rollouts.returns.detach().flatten().cpu().numpy(), it=self._n_updates)
            if self.args.action_type == 'discrete':
                self.log_statistics(category='entropy', data=dist_entropy.detach().cpu().numpy(), it=self._n_updates)
            else:
                self.log_statistics(category='entropy', data=(-action_log_probs).detach().cpu().numpy(), it=self._n_updates)
                self.log_statistics(category='policy_gauss_std', data=self.actor_critic.dist.log_std.data.exp().cpu().numpy(), it=self._n_updates)

        return value_loss.item(), pg_loss.item(), entropy_loss.item()

    def predict(self, obs, task_ids, deterministic=False):
        value, action, action_log_probs = self.actor_critic.act(obs, task_ids, deterministic)
        return value, action, action_log_probs

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
"""