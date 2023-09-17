
import os 
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from training.logger import LoggerDQN
from rl.dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQN(object):

    def __init__(self,
                obs_shape, 
                action_space,
                device,
                hidden_dim=256,
                lr=1e-4,
                optimizer='adam',
                opt_eps=1e-5, 
                loss='huber',
                batch_size=32, 
                discount=0.99,
                buffer_size=1000,
                target_update_freq=10, 
                exploration_steps=1000,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                max_grad_norm=1.0,
                double_dqn=False,
                prioritized_er=False,
                prioritized_er_alpha=0.6,
                prioritized_er_beta0=0.4,
                prioritized_er_steps=1000,
                seed=0, 
                log_dir='./logs',
                monitoring=None):
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.action_shape = action_space.max_dim
        self.device = device 

        # make dqn 
        self.make_q_nets(obs_shape, action_space.max_dim, hidden_dim)

        if optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=lr, alpha=0.99, eps=opt_eps)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=opt_eps)

        if loss == 'mse':
            self.loss = nn.MSELoss(reduction='none').to(self.device)
        elif loss == 'huber':
            self.loss = nn.SmoothL1Loss(reduction='none').to(self.device) 

        # dqn parameters
        self.discount = discount 
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.exploration_rate = exploration_initial_eps
        self.eps_start = exploration_initial_eps
        self.eps_end = exploration_final_eps
        self.eps_decay = exploration_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.double_dqn = double_dqn
        self.prioritized_er = prioritized_er
        self.alpha = prioritized_er_alpha
        self.beta = prioritized_er_beta0
        self.beta_start = prioritized_er_beta0
        self.beta_steps = prioritized_er_steps
        # make replay buffer
        self.make_replay_buffer(buffer_size, prioritized_er)

        # rng for selecting random actions
        self.rs = np.random.RandomState(seed)
        self.n_updates = 0
        self.learning_started = False
        self.logger = LoggerDQN(log_dir=log_dir,
                        monitoring=monitoring,
                        monitoring_dir=os.path.join(log_dir, 'monitoring', 'dqn'))

    def make_q_nets(self, obs_dim, action_dim, hidden_dim):
        self.q_net = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q_net_target = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.hard_target_update() # initialize target as q_net

    def make_replay_buffer(self, buffer_size, prioritized_er=False):
        # Create the replay buffer
        if prioritized_er:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=self.alpha)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, obs, task, deterministic=False):
        # NOTE: this should only take in one obs I think
        if not deterministic and self.rs.uniform() < self.exploration_rate:
            if isinstance(task, torch.Tensor):
                task = task.item()
            n_actions_at_task = self.action_space.get_dim_at_task(task)
            action = self.rs.randint(n_actions_at_task)
            action = torch.tensor([action], dtype=torch.long).to(self.device)
        else:
            with torch.no_grad():
                q_values = self.q_net(obs)
                q_values = self.mask_invalid_actions(q_values, task)
                action = torch.argmax(q_values, dim=1).view(-1)
        return action 

    def predict(self, obs, task, deterministic=False):
        # used in test_in_env function
        return self.select_action(obs, task, deterministic)

    def select_actions_from_batch(self, obs_batch, task_batch):
        q_values = self.q_net(obs_batch)
        q_values = self.mask_invalid_actions(q_values, task_batch)
        actions = torch.argmax(q_values, dim=1).view(-1)
        return actions

    def mask_invalid_actions(self, q_values, tasks):
        for i, tt in enumerate(tasks):
            offset = self.action_space.get_dim_at_task(tt)
            q_values[i, offset:].data.fill_(-1e8)
        return q_values

    def get_q_values_given_net(self, q_net, obs, tasks):
        q_values = q_net(obs)
        q_values = self.mask_invalid_actions(q_values, tasks)
        return q_values

    def get_q_values_given_net_and_actions(self, q_net, obs, act, tasks=None):
        q_values = q_net(obs)
        selected_q_values = q_values.gather(dim=1, index=act.view(-1, 1)).squeeze(1)
        return selected_q_values

    def update(self):
        ### Update q-net with mini-batch of experiences.
        obs_t, act_t, rew_t, obs_tp1, done_mask, task_ids, importance_weights = self.get_experiences_from_buffer(self.batch_size)

        # Compute targets
        with torch.no_grad():
            next_q_values = self.get_q_values_given_net(self.q_net_target, obs_tp1, task_ids+1)
            
            if self.double_dqn:
                q_tp1_values = self.get_q_values_given_net(self.q_net, obs_tp1, task_ids+1)
                _, a_prime = q_tp1_values.max(1) # get actions with highest q-value, 1d tensor
                next_q_values = next_q_values.gather(1, a_prime.unsqueeze(1)) # get target q-values wit hactions selected by online network
                next_q_values = next_q_values.squeeze(1) 
                target_q_values = rew_t + (1.0 - done_mask) * self.discount * next_q_values
            else:
                next_q_values, _ = next_q_values.max(1)
                target_q_values = rew_t + (1.0 - done_mask) * self.discount * next_q_values

        # Get current Q values
        current_q_values = self.get_q_values_given_net_and_actions(self.q_net, obs_t, act_t, task_ids)
        #current_q_values = current_q_values.gather(dim=1, index=act_t.view(-1, 1)).squeeze(1)

        # compute the error (potentially clipped)
        #print('current_q_values: ', current_q_values)
        td_error = (current_q_values - target_q_values) #.squeeze(1)
        loss = self.loss(current_q_values, target_q_values) # .squeeze(1) # turn to 1d tensor
        weighted_loss = torch.mean(importance_weights * loss)

        # Optimize the model 
        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # update parameters - target update and exploration rate
        self.n_updates += 1

        # time to do target update?
        if self.n_updates % self.target_update_freq == 0:
            self.hard_target_update()

        # decay exploration rate (epsilon)
        self.exploration_rate = max(self.eps_end, self.exploration_rate - ((self.eps_start-self.eps_end) / self.eps_decay))

        # update priorities in Prioritized ER
        if self.prioritized_er:
            new_priorities = np.abs(td_error.detach().cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)
            self.beta = min(1.0, self.beta + ((1.0-self.beta_start) / self.beta_steps))

        # Get gradient statistics
        grad_stats = {}
        for name, param in self.q_net.named_parameters():
            grads = param.grad.data.cpu().numpy()
            #print(name, np.mean(grads))
            abs_max = np.max(np.abs(grads))
            mean_sq_value = np.mean(grads**2)
            grad_stats[name + '.grad'] = {'abs_max': abs_max, 'mean_sq_value': mean_sq_value}
        # Setup summary of results from batch
        batch_summary = {'q_selected': current_q_values.view(-1).detach().cpu().numpy(),
                        'q_selected_target': target_q_values.cpu().numpy(),
                        'loss': loss.detach().cpu().numpy(),
                        'td_error': td_error.detach().cpu().numpy(),
                        'grad_stats': grad_stats,
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
        importance_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        return obs_t, act_t, rew_t, obs_tp1, done_mask, task_ids, importance_weights

    def hard_target_update(self): 
        # copy q_net weights to target
        self.q_net_target.load_state_dict(self.q_net.state_dict())

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
        

def weight_init(m):
    """ weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        q_values = self.net(obs)
        self.outputs['q'] = q_values
        return q_values



class PerActionDQN(DQN):

    def __init__(self,
                obs_shape, 
                action_space,
                device,
                hidden_dim=256,
                lr=1e-4,
                optimizer='adam',
                opt_eps=1e-5, 
                loss='huber',
                batch_size=32, 
                discount=0.99,
                buffer_size=1000,
                target_update_freq=10, 
                exploration_steps=1000,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                max_grad_norm=1.0,
                double_dqn=False,
                prioritized_er=False,
                prioritized_er_alpha=0.6,
                prioritized_er_beta0=0.4,
                prioritized_er_steps=1000,
                seed=0, 
                log_dir='./logs',
                monitoring=None):
        super().__init__(obs_shape, action_space, device, hidden_dim, lr, optimizer, opt_eps, loss, batch_size, discount, 
                buffer_size, target_update_freq, exploration_steps, exploration_initial_eps, exploration_final_eps, max_grad_norm, 
                double_dqn, prioritized_er, prioritized_er_alpha, prioritized_er_beta0, prioritized_er_steps,
                seed, log_dir, monitoring)
        # make per-action q_net
        action_dim = self.action_space.n_tasks-1
        self.make_q_nets(obs_shape, action_dim, hidden_dim)

    def make_q_nets(self, obs_dim, action_dim, hidden_dim):
        self.q_net = PerActionQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q_net_target = PerActionQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.hard_target_update() # initialize target as q_net

    def select_action(self, obs, task, deterministic=False):
        # NOTE: this should only take in one obs I think
        if not deterministic and self.rs.uniform() < self.exploration_rate:
            if isinstance(task, torch.Tensor):
                task = task.item()
            n_actions_at_task = self.action_space.get_dim_at_task(task)
            action = self.rs.randint(n_actions_at_task)
            action = torch.tensor([action], dtype=torch.long).to(self.device)
        else:
            with torch.no_grad():
                actions = self.action_space.get_actions_at_task(task.item())
                actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
                obs = obs.repeat(len(actions), 1)
                q_values = self.q_net(obs, actions)
                action = torch.argmax(q_values, dim=0).view(-1)
        return action 

    def select_actions_from_batch(self, obs_batch, task_batch):
        actions = []
        for obs, task in zip(obs_batch, task_batch):
            act = self.select_action(obs, task, deterministic=True)
            actions.append(act)
        actions = torch.stack(actions, dim=0)
        return actions

    def get_q_values(self, obs, tasks):
        q_values = []
        for obs, task in zip(obs, tasks):
            actions = self.action_space.get_actions_at_task(task.item())
            obs.repeat(len(actions), 1)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            q_val, _ = self.q_net(obs, actions).max(dim=0)
            q_values.append(q_val)
        q_values = torch.stack(q_values, dim=0)
        return q_values

    def get_q_values_given_net(self, q_net, obs, tasks):
        q_values = []
        for x, task in zip(obs, tasks):
            #print(task)
            actions = self.action_space.get_actions_at_task(task.item())
            #print(actions)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            #print(actions.requires_grad)
            x = x.repeat(len(actions), 1)
            #print(x.requires_grad)
            #print(x.size())
            
            #print(actions.size())
            q_val, _ = q_net(x, actions).max(dim=0)
            q_values.append(q_val)
        q_values = torch.stack(q_values, dim=0)
        return q_values

    def get_q_values_given_net_and_actions(self, q_net, obs, acts, tasks):
        q_values = []
        # collect actions from index
        actions = []
        for i in range(len(obs)):
            act_idx, t = acts[i].item(), tasks[i].item()
            action = self.action_space.get_action_by_index(t, act_idx)
            actions.append(action)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        q_values = q_net(obs, actions)
        q_values = q_values.view(-1)
        return q_values


class PerActionQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, act):
        obs = torch.cat([obs, act], dim=-1)
        #print('obs: ', obs)
        q_values = self.net(obs)
        self.outputs['q'] = q_values
        return q_values

