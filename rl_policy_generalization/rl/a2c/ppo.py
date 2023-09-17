# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/1120cdfe94e79294a52486590d9c2bcc5c01730d/a2c_ppo_acktr/algo/ppo.py#L7

import os 
import numpy as np 
import torch
import torch.nn as nn

from rl.a2c_ppo_new.base import BaseAlgo
from rl.a2c_ppo.utils import explained_variance

class PPO(BaseAlgo):

    def __init__(self, actor_critic, n_processes, n_steps, value_loss_coef=0.5, entropy_coef=0.01, 
                    discount=0.99, lr=7e-4, max_grad_norm=0.5, use_gae=True, gae_lambda=0.95, use_proper_time_limits=False, 
                    adam_eps=1e-5, clip_eps=0.2, epochs=10, batch_size=64, use_clipped_value_loss=True, device='cpu', log_dir='./'):

        super().__init__(actor_critic, n_processes, n_steps, value_loss_coef, entropy_coef, discount, lr, 
                            max_grad_norm, use_gae, gae_lambda, use_proper_time_limits, device, log_dir)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.adam_eps = adam_eps
        self.lr = lr
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), self.lr, eps=self.adam_eps)

    def policy_loss(self, action_log_probs, old_action_log_probs, advantages):
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        return -torch.min(surr1, surr2).mean()

    def value_loss(self, value_preds, returns, values):
        if self.use_clipped_value_loss:
            value_clipped = value_preds + torch.clamp(values - value_preds, -self.clip_eps, self.clip_eps)
            surr1 = (values - returns).pow(2)
            surr2 = (value_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(surr1, surr2).mean()
        else:
            value_loss = 0.5 * (returns - values).pow(2).mean()
        return value_loss

    def update_parameters(self):

        # prepare rollouts for training
        _ = self.collect_experiences()
        value_loss_epoch = 0
        policy_loss_epoch = 0
        entropy_epoch = 0
        approx_kl_div_epoch = 0
        logs = {}

        # normalize advantages
        advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.epochs):
            data_generator = self.rollouts.feed_forward_generator(advantages, mini_batch_size=self.batch_size)

            log_entropies = []
            log_values = []
            log_policy_loss = []
            log_value_loss = []
            log_grad_norms = []
            log_approx_kl_divs = []
            n_mini_batches = 0

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, task_id_batch, old_action_log_probs_batch, adv_targ = sample
                """
                print('obs_batch: ', obs_batch.view(-1))
                print('actions_batch: ', actions_batch.view(-1))
                print('value_preds_batch: ', value_preds_batch.view(-1))
                print('return_batch: ', return_batch.view(-1))
                print('old_action_log_probs_batch: ', old_action_log_probs_batch.view(-1))
                print('adv_targ: ', adv_targ.view(-1))
                """
                n_mini_batches += 1

                # compute loss
                if self.actor_critic.use_task_ids:
                    values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch, task_id_batch)
                else:
                    values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch)
                
                # sum losses 
                entropy_loss = dist_entropy.mean()
                policy_loss = self.policy_loss(action_log_probs, 
                                            old_action_log_probs_batch, 
                                            adv_targ) 
                value_loss = self.value_loss(value_preds_batch, 
                                            return_batch, 
                                            values) 
                #value_loss = (return_batch - values).pow(2).mean()
                loss = policy_loss + value_loss * self.value_loss_coef - entropy_loss * self.entropy_coef

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                # See https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
                # and https://spinningup.openai.com/en/latest/algorithms/ppo.html
                with torch.no_grad():
                    log_ratio = action_log_probs - old_action_log_probs_batch
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio) #cpu().numpy()
                    log_approx_kl_divs.append(approx_kl_div.item())

                # Update actor-critic with batch
                self.optimizer.zero_grad()
                loss.backward()
                update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.actor_critic.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values
                log_entropies.append(entropy_loss.item())
                log_values.append(values.mean().item())
                log_policy_loss.append(policy_loss.item())
                log_value_loss.append(value_loss.item())
                log_grad_norms.append(update_grad_norm.item())

                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy_loss.item()
                approx_kl_div_epoch += approx_kl_div.item()

        # set latest obs, action as first in rollout storage 
        self.rollouts.after_update() 
        n_updates_ = self.epochs * n_mini_batches
        self.n_updates += n_updates_

        # Log some values
        logs["entropy"] = np.mean(log_entropies)
        logs["value"] = np.mean(log_values)
        logs["policy_loss"] = np.abs(np.mean(log_policy_loss))
        logs["value_loss"] = np.mean(log_value_loss)
        logs["grad_norm"] = np.mean(log_grad_norms)
        logs["approx_kl"] = np.mean(log_approx_kl_divs)

        logs["value_loss_epoch"] = value_loss_epoch / n_updates_
        logs["policy_loss_epoch"] = policy_loss_epoch / n_updates_
        logs["entropy_epoch"] = entropy_epoch / n_updates_
        logs["approx_kl_epoch"] = approx_kl_div_epoch / n_updates_

        return logs           


                
"""
import os 
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from training.logger import Logger
from rl.a2c_ppo.utils import explained_variance

class PPO():
    def __init__(self,
                 args,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 adam_eps=1e-5,
                 max_grad_norm=None,
                 target_kl=None,
                 use_clipped_value_loss=True):
        self.args = args
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.use_clipped_value_loss = use_clipped_value_loss
        self.adam_eps = adam_eps
        #print('use_clipped_value_loss: ', self.use_clipped_value_loss)

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=self.adam_eps)

        self._n_updates = 0

        self.logger = Logger(log_dir=args.log_dir,
                        monitoring=args.monitoring,
                        monitoring_dir=os.path.join(args.log_dir, 'monitoring', 'ppo'))

    def update(self, rollouts):
        # Normalize advantage (here, or normalize for mini-batch advantages?)
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5) #standardized advantages

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        for e in range(self.ppo_epoch):
            approx_kl_divs = []
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, \
                   value_preds_batch, return_batch, task_id_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, task_id_batch-1, actions_batch)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                # clipped surrogate loss
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param) * adv_targ
                pg_loss = -torch.min(surr1, surr2).mean()
                # logging
                pg_losses.append(pg_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_param).float()).item()
                clip_fractions.append(clip_fraction)

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_loss = (values - return_batch).pow(2)
                    value_loss_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_loss,
                                                 value_loss_clipped).mean()
                else:
                    # no clipping
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                value_losses.append(value_loss.item())

                if dist_entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = torch.mean(-action_log_probs)
                    if entropy_loss < 0:
                        print('action_log_probs: ', action_log_probs)
                else:
                    entropy_loss = torch.mean(dist_entropy)
                entropy_losses.append(entropy_loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                # See https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
                # and https://spinningup.openai.com/en/latest/algorithms/ppo.html
                with torch.no_grad():
                    log_ratio = action_log_probs - old_action_log_probs_batch
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.args.verbose >= 1:
                        print(f"Early stopping at step {e} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + pg_loss - entropy_loss * self.entropy_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if not continue_training:
                break

        self._n_updates += 1

        explained_var = explained_variance(y_pred=rollouts.value_preds[:-1].flatten().cpu().numpy(), 
                                            y_true=rollouts.returns[:-1].flatten().cpu().numpy())
        #print('explained_var: ', explained_var)
        # Log results
        self.logger.log_scalar('train', 'value_loss', np.mean(value_losses), self._n_updates)
        self.logger.log_scalar('train', 'policy_loss', np.mean(pg_losses), self._n_updates)
        self.logger.log_scalar('train', 'entropy_loss', np.mean(entropy_losses), self._n_updates)
        self.logger.log_scalar('train', 'clip_fraction', np.mean(clip_fractions), self._n_updates)
        self.logger.log_scalar('train', 'approx_kl', np.mean(approx_kl_divs), self._n_updates)
        self.logger.log_scalar('train', 'loss', loss.item(), self._n_updates)
        self.logger.log_scalar('train', 'explained_variance', explained_var, self._n_updates)

        self.log_statistics(category='advantages', data=advantages.flatten().cpu().numpy(), it=self._n_updates)
        self.log_statistics(category='value_preds', data=rollouts.value_preds.flatten().cpu().numpy(), it=self._n_updates)
        self.log_statistics(category='returns', data=rollouts.returns.flatten().cpu().numpy(), it=self._n_updates)
        if self.args.action_type == 'discrete':
            self.log_statistics(category='entropy', data=dist_entropy.detach().cpu().numpy(), it=self._n_updates)
        else:
            self.log_statistics(category='entropy', data=(-action_log_probs).detach().cpu().numpy(), it=self._n_updates)
            self.log_statistics(category='policy_gauss_std', data=self.actor_critic.dist.log_std.data.exp().cpu().numpy(), it=self._n_updates)

        return np.mean(value_losses), np.mean(pg_losses), np.mean(entropy_losses)

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