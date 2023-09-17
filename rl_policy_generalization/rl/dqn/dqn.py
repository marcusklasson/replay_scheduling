
import os, shutil
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from training.logger import Logger
from rl.common.schedules import LinearSchedule
from rl.dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQN():
    def __init__(self,
                 dqn_policy,
                 num_episodes=100, 
                 gamma=0.99,
                 lr=1e-3,
                 opt='adam',
                 loss='huber', 
                 buffer_size=10000, 
                 target_update_freq=100, 
                 exploration_fraction=0.1,
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.02,
                 max_grad_norm=1.0,
                 double_dqn=False,
                 prioritized_er=False,
                 device='cpu',
                 log_dir='./',
                 lr_scheduler='none'
                 ):

        self.policy = dqn_policy
        #self.args = args
        #self.device = args.device
        self.device = dqn_policy.device #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_episodes = num_episodes
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq 
        self.exploration_fraction = exploration_fraction        
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.prioritized_er = prioritized_er
        self.double_dqn = double_dqn

        self._setup_model()
        
        self.replay_buffer = self._make_replay_buffer(buffer_size)

        self.gamma = gamma
        #self.optimizer = optim.Adam(self.q_net.parameters(), lr)
        #opt_eps = 1e-5 if args.dqn.opt_eps is None else args.dqn.opt_eps
        if opt == 'rmsprop':
            self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=lr, alpha=0.99)#, eps=opt_eps) 
        elif opt == 'adam':
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, betas=(0.9, 0.999))#, eps=opt_eps) 
        elif opt == 'sgd':
            self.optimizer = optim.SGD(self.q_net.parameters(), lr=lr) 

        if lr_scheduler == 'cyclic':
            self.lr_scheduler = optim.lr_scheduler.CyclicLR(
                                    self.optimizer, 
                                    base_lr=args['dqn']['base_lr'], 
                                    max_lr=args['dqn']['max_lr'],
                                    step_size_up=args['dqn']['step_size_up'],
                                    mode=args['dqn']['cyclic_lr_mode'],
                                    gamma=0.99, # is only used when mode='exp_range'
                                    cycle_momentum=False,
                                    verbose=False)
        else:
            self.lr_scheduler = None
        

        if loss == 'mse':
            self.loss = nn.MSELoss(reduction='none').to(self.device)
        else:# args.dqn.loss == 'huber':
            self.loss = nn.SmoothL1Loss(reduction='none').to(self.device) 
        #print(self.loss)
        self.exploration_rate = self.exploration_initial_eps #1.0
        self.exploration_steps = 0
        self.beta = 0.4
        self.beta_steps = 0
        self.n_param_updates = 0
        self.n_timesteps = 0
        self.learning_started = False

        self.logger = Logger(log_dir=log_dir,
                        #monitoring=args.monitoring,
                        #monitoring_dir=os.path.join(args.log_dir, 'monitoring', 'dqn')
                        )
        
    def _setup_model(self):
        self.exploration_schedule = LinearSchedule(int(self.exploration_fraction * self.num_episodes), #* (self.args.n_tasks-1)), 
                                                    self.exploration_final_eps,
                                                    self.exploration_initial_eps)
        self._create_aliases()
        if self.prioritized_er:
            self.beta_schedule = LinearSchedule(int(self.num_episodes), #* (self.args.n_tasks-1) - 10),#- 1 - self.args.dqn.batch_size),
                                                    initial_p=0.4, #self.args.prioritized_replay_beta0,
                                                    final_p=1.0)

    def _create_aliases(self):
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target
        self.action_space = self.policy.action_space
        self.n_actions = self.policy.n_actions

    def hard_target_update(self):
        #print(' Updating Target Network with Policy Params...')
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def after_episode(self,):
        self.exploration_steps += 1
        self.exploration_rate = self.exploration_schedule.value(self.exploration_steps)
        if self.prioritized_er:
            self.beta_steps += 1
            self.beta = self.beta_schedule.value(self.beta_steps)
        #print('eps: %.3f' %(self.exploration_rate))

    def _on_step(self):
        
        # update exploaration rate and target network
        """
        if ((self.n_timesteps+1) % self.target_update_freq == 0):
            print('Updating Target Network with Policy Params...')
            self.q_net_target.load_state_dict(self.q_net.state_dict()) #self.update_target_net()
        """
        self.exploration_rate = self.exploration_schedule.value(self.n_param_updates)
        #self.exploration_steps += 1
        self.log_scalar('hyper_params', 'exploration', self.exploration_rate, self.n_timesteps)
        if self.prioritized_er:
            self.beta = self.beta_schedule.value(self.beta_steps)
            self.beta_steps += 1
            self.log_scalar('hyper_params', 'beta', self.beta, self.beta_steps)
        # LR scheduler
        if self.lr_scheduler:
            last_lr = self.lr_scheduler.get_last_lr()
            self.log_scalar('hyper_params', 'lr', last_lr, self.n_timesteps)
            self.lr_scheduler.step()

    def predict(self, obs, task=None, deterministic=False):
        # selecting actions
        #if not deterministic and self.rs.uniform() < self.exploration_rate: #np.random.rand() < self.exploration_rate:
        if not deterministic and np.random.rand() < self.exploration_rate: #np.random.rand() < self.exploration_rate:
            # take random action
            if task is None:
                act = np.random.randint(self.n_actions) 
            else:
                n_actions = self.action_space.get_dim_at_task(task.item())
                act = np.random.randint(n_actions) #self.rs.randint(self.n_actions)
            action = torch.as_tensor([act], dtype=torch.long, device=self.device)
            #print('output action1: ', action)
        else:
            action = self.policy.predict(obs, task)
            #print('output action2: ', action)
        #print('output action: ', action)
        return action 


    def update(self, batch_size):
        ### Update policy network with mini-batch of experiences.

        # Shorthands
        optimizer = self.optimizer 
        gamma = self.gamma 
        device = self.device

        obs_t, act_t, rew_t, obs_tp1, done_mask, task_ids, importance_weights, batch_idxes = self.get_experiences_from_buffer(batch_size)
        
        # Compute targets
        with torch.no_grad():
            next_q_values = self.q_net_target(obs_tp1, task_ids+1) if (task_ids is not None) else self.q_net_target(obs_tp1)
            
            if self.double_dqn:
                q_tp1_values = self.q_net(obs_tp1, task_ids+1) if (task_ids is not None) else self.q_net(obs_tp1)
                _, a_prime = q_tp1_values.max(1) # get actions with highest q-value, 1d tensor
                next_q_values = next_q_values.gather(1, a_prime.unsqueeze(1)) # get target q-values wit hactions selected by online network
                next_q_values = next_q_values.squeeze(1) 
                #print('a_prime: ', a_prime.view(-1))
                #print('next_q_values: ', next_q_values.view(-1))
                target_q_values = rew_t + (1.0 - done_mask) * gamma * next_q_values
            else:
                next_q_values, a_prime = next_q_values.max(1)
                #print('a_prime: ', a_prime.view(-1))
                #print('next_q_values: ', next_q_values.view(-1))
                target_q_values = rew_t + (1.0 - done_mask) * gamma * next_q_values

        # Get current Q values
        current_q_values = self.q_net(obs_t, task_ids) if (task_ids is not None) else self.q_net(obs_t)
        current_q_values = current_q_values.gather(dim=1, index=act_t.view(-1, 1)).squeeze(1)

        # compute the error (potentially clipped)
        td_error = (current_q_values - target_q_values) #.squeeze(1)
        loss = self.loss(current_q_values, target_q_values) # .squeeze(1) # turn to 1d tensor
        # weighted loss for prioritized replay
        weighted_loss = torch.mean(importance_weights * loss)

        # Optimize the model 
        optimizer.zero_grad()
        weighted_loss.backward()
        grad_norm_train = sum(p.grad.data.norm(2).item() ** 2 for p in self.q_net.parameters()) ** 0.5
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        optimizer.step()

        # update parameters
        self.n_param_updates += 1

        # Get gradient statistics
        grad_stats = {}
        for name, param in self.q_net.named_parameters():
            grads = param.grad.data.cpu().numpy()
            #if name == 'linear.bias':
                #print(grads)
                #print(torch.unique(act_t.view(-1)))
            abs_max = np.max(np.abs(grads))
            mean_sq_value = np.mean(grads**2)
            grad_stats[name + '.grad'] = {'abs_max': abs_max, 'mean_sq_value': mean_sq_value}

        # update priorities in Prioritized ER
        if self.prioritized_er:
            new_priorities = np.abs(td_error.detach().cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

        # Setup summary of results from batch
        batch_summary = {'q_selected': current_q_values.view(-1).detach().cpu().numpy(),
                        'q_selected_target': target_q_values.cpu().numpy(),
                        'loss': loss.detach().cpu().numpy(),
                        'td_error': td_error.detach().cpu().numpy(),
                        'grad_stats': grad_stats,
                        'grad_norm_train': grad_norm_train,
                        }
        return batch_summary

    def get_experiences_from_buffer(self, batch_size):
        if self.prioritized_er:
            experience = self.replay_buffer.sample(batch_size, beta=self.beta)
            (obses_t, actions, rewards, obses_tp1, dones, task_ids, weights, batch_idxes) = experience
        else:
            obses_t, actions, rewards, obses_tp1, dones, task_ids = self.replay_buffer.sample(batch_size)
            weights, batch_idxes = np.ones_like(rewards), None

        # Get batch from replay buffer
        obs_t = torch.as_tensor(obses_t, dtype=torch.float32, device=self.device)
        act_t =  torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rew_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        obs_tp1 = torch.as_tensor(obses_tp1, dtype=torch.float32, device=self.device)
        done_mask = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        task_ids = torch.as_tensor(task_ids, dtype=torch.long, device=self.device)
        if torch.any(task_ids == -1):
            # Set task_ids to None if task_ids are not used for environment, a bit "hacky"
            task_ids = None 
        importance_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        return obs_t, act_t, rew_t, obs_tp1, done_mask, task_ids, importance_weights, batch_idxes

    def _make_replay_buffer(self, buffer_size, prioritized_replay_alpha=0.6):
        # Create the replay buffer
        if self.prioritized_er:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        else:
            replay_buffer = ReplayBuffer(buffer_size)
        return replay_buffer

    def save_checkpoint(self, state, is_best, checkpoint_dir='./', filename='checkpoint.pth.tar'):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, filename))
        if is_best:
            shutil.copyfile(os.path.join(checkpoint_dir, filename),
                            os.path.join(checkpoint_dir, 'policy_best.pth.tar'))

    def load_checkpoint(self, checkpoint_dir, filename):
        """ Load checkpoint for model and optimizer to trainer from file.
        """
        # Load q_nets
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        self.q_net.load_state_dict(checkpoint['q_net_state_dict']) 
        self.q_net_target.load_state_dict(checkpoint['q_net_target_state_dict']) 
        # Load optimizer and replay buffer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.replay_buffer = checkpoint['replay_buffer']

    def load_checkpoint_simple(self, checkpoint_dir, filename):
        """ Load checkpoint for model and optimizer to trainer from file.
        """
        # Load q_nets
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        self.q_net.load_state_dict(checkpoint['q_net_state_dict']) 
        self.q_net_target.load_state_dict(self.q_net.state_dict()) 

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

    def log_scalar(self, category, key, value, it):
        self.logger.add(category, key, value, it)

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

    def plot_action_trajectories(self, action_trajectories):
        trajectories = np.array(action_trajectories)
        n_actions = np.max(trajectories)
        n_lines = len(trajectories)
        c = np.arange(1, n_lines+1)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        save_dir = self.logger.log_dir
        fig, ax = plt.subplots()
        x = np.arange(1, len(trajectories[0])+1)
        
        for i, traj in enumerate(trajectories, start=1):
            ax.plot(x, traj, 'o-', c=cmap.to_rgba(i + 1))
        ax.set_xlabel('Time step')
        ax.set_ylabel('Action Index')
        ax.set_title('Action trajectories')
        ax.set_xticks(x)
        ax.set_yticks(np.arange(n_actions+1))
        fig.colorbar(cmap)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/action_trajectories.png')
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