
import os
import time
import random
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from envs_new.env_fixed_seed import EnvFixedSeed
from training.utils import load_pickle, save_pickle, set_random_seed

class EnvBFS(EnvFixedSeed):
    """ Training environment for CL classifier which RL algorithm can make calls to
        for recieving observations, rewards, etc. Similar to OpenAI gym environment.
    """
    def __init__(self, observation_dim, action_space, dataset, seed, args):
        super().__init__(observation_dim, action_space, dataset, seed, args)

    def reset(self):
        # initialize CL network and its optimizer
        self.trainer.reset()
        self.res_all = []

        # train on first task
        task_dataset = self.dataset.get_dataset_for_task(task_id=0)
        self.trainer.train_single_task_in_bfs(task_id=0, train_dataset=task_dataset['train'])
        #self.trainer.train_model_at_task(t=0, train_dataset=task_dataset['train'], actions=[])
        res = self.evaluate_model_at_task(current_task=0)
        self.res_all.append(res) # used for getting state

        # Get initial observation from starting distribution
        state = self._get_state()
        # Update history with initial info
        info = {'state': state, 
                'acc': res['val']['acc'], 
                'loss': res['val']['loss'], 
                'val': res['val'],
                'test': res['test'],
                'task': 0, 
                }
        #self.ep_info.append(info)
        self.save_transition(cmd='reset', info=info)
        return state, info

    def step(self, t, actions):
        """ return observation (state), reward, done (boolean), info (dict with training info)
        """
        action = actions[-1]
        #task_proportion = self.action_space.get_action_by_index(t-1, action)
        current_task_proportion, task_proportions, actions = self.get_task_proportions(t, actions)
        #print(task_proportions)
        task_dataset = self.dataset.get_dataset_for_task(task_id=t)
        self.trainer.train_single_task_in_bfs(task_id=t, train_dataset=task_dataset['train'], task_proportions=current_task_proportion)
        #self.train_model_at_task(t, task_dataset['train'], action, current_task_proportion)
        #self.trainer.train_model_at_task_in_bfs(t, task_dataset['train'], actions, current_task_proportion)
        res = self.evaluate_model_at_task(current_task=t)
        self.res_all.append(res) # used for getting state

        # Observe next state and reward
        state = self._get_state()
        done = (t+1 >= self.n_tasks)
        reward = self._get_reward(terminal_state=done)
        # Prepare output information
        info = {'state': state, 
                'done': done,
                'reward': reward,
                'action': action,
                'acc': res['val']['acc'], 
                'loss': res['val']['loss'], 
                'task_proportion': current_task_proportion,
                'rs': task_proportions,
                'actions': actions,
                'episode': {'r': reward},
                'val': res['val'],
                'test': res['test'],
                #'accs': res['accs'],
                #'losses': res['losses'],
                'task': t, 
                } 
        #self.ep_info.append(info)
        #if self.verbose > 0:
        #    if t+1 == self.n_tasks:
        #        print(np.stack(res['accs']['test'], axis=0))
        self.save_transition(cmd='step', info=info)
        return state, reward, done, info

    def evaluate_model_at_task(self, current_task):
        # Evaluate model
        res = {}
        acc_test, acc_val = np.zeros(self.n_tasks), np.zeros(self.n_tasks)
        loss_test, loss_val = np.zeros(self.n_tasks), np.zeros(self.n_tasks) 
        for u in range(current_task+1):
            task_dataset = self.dataset.get_dataset_for_task(u)
            val_res = self.trainer.eval_task(current_task, task_dataset['valid']) # get task accuracy
            test_res = self.trainer.eval_task(current_task, task_dataset['test']) # get task accuracy
            if self.verbose > 0:
                print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, task_dataset['name'],
                                                                                            test_res['loss_t'],
                                                                                            test_res['acc_t']*100))
            acc_test[u] = test_res['acc_t']
            loss_test[u] = test_res['loss_t']
            acc_val[u] = val_res['acc_t']
            loss_val[u] = val_res['loss_t']
        res['test'] = {'acc': acc_test, 'loss': loss_test}
        res['val'] = {'acc': acc_val, 'loss': loss_val}
        #print('acc_val: ', acc_val)

        return res

    def _get_state(self):
        # state based on task performance for simplicity 
        res_curr = self.res_all[-1]['val']
        state = res_curr['acc'][:-1]
        return state

    def _get_reward(self, terminal_state=False):
        """ Compute reward
        """
        # Shorthands
        res = self.res_all[-1]['val']
        reward_type = self.reward_type
        if reward_type == 'sparse':
            reward = 0.0
            if terminal_state:
                accs = res['acc'] #history['states'][-1] # get final state
                reward = np.mean(accs)
        elif reward_type == 'dense': # NOTE: need to think about this one!
            reward = np.sum(res['acc']) / np.sum(res['acc'] > 0).astype(np.float32) # get mean acc to scale reward in between range [0, 1]
        else:
            raise ValueError('Reward type %s is invalid.' %(reward_type))
        return reward

    def get_task_proportions(self, t, actions):
        """
        # get actions
        if t > 1:
            actions = [info['action'] for info in self.ep_info[1:]]
            actions.append(action)
        elif t == 1:
            actions = [0]
        else:
            raise ValueError('Can only get task proportions for task {}>=1'.format(t))
        """
        task_proportions = []
        for t, a in enumerate(actions):
            task_prop = self.action_space.get_action_by_index(t, a)
            task_proportions.append(task_prop)
        return task_proportions[-1], task_proportions, actions

