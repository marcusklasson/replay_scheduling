import os
import time
import random
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from envs.base import EnvBase
from envs.base_simple import EnvBaseSimple
from envs.utils import get_filename_for_table
from training.utils import load_pickle, save_pickle, set_random_seed

class EnvTable(EnvBase):
    """ Training environment for CL classifier which RL algorithm can make calls to
        for recieving observations, rewards, etc. Similar to OpenAI gym environment.
    """

    def __init__(self, observation_dim, action_space, dataset, seed, args, shuffle_labels=False):
        super().__init__(observation_dim, action_space, dataset, seed, args, shuffle_labels)
        self.load_checkpoints = args.load_checkpoints
        self.filename_transition_table = get_filename_for_table(args, dataset_name=self.dataset_name, task_ids=self.task_ids, seed=self.seed)
        self.transition_table_dir = args.table_dir
        self.transitions_info = self.get_transition_table(fname=os.path.join(self.transition_table_dir, self.filename_transition_table))
        self.transitions = self.transitions_info['data']

    def get_transition_table(self, fname):
        """ Get dictionary that stores all state transitions. 
        """
        if os.path.exists(fname):
            print('Loading table with transitions: %s' %(fname))
            return load_pickle(fname)
        else:
            raise ValueError('Could not load table {}.\nEnvironment class only works with universal tables... '.format(fname))

    def get_transition(self, t, index):
        if index in self.transitions[t].keys(): 
            transition = self.transitions[t][index]
        else:
            raise ValueError('Transition key {} is missing in table from {}'.format(index, self.filename_transition_table))
        return transition

    def reset(self):
        # reset environment 
        self.ep_res, self.ep_info = [], []
        transition = self.get_transition(t=0, index='0')
        self.ep_info.append(transition)
        self.ep_res.append(transition)
        state = self._get_state()
        return state, transition # return transition as info

    def step(self, t, action):
        # Get transition index
        if t > 1:
            actions = [info['action'] for info in self.ep_info[1:]] #[info['action'] for info in self.ep_info[1:]]
            actions.append(action)
        elif t == 1:
            actions = [0]
        else:
            raise ValueError('Can only get task proportions for task {}>=1'.format(t))
        
        index = '0-' + '-'.join([str(a) for a in actions])
        #print('index: ', index)
        transition = self.get_transition(t=t, index=index)
        info = transition.copy()
        info['task_proportion'] = self.action_space.get_action_by_index(t-1, action)
        self.ep_res.append(info)
        self.ep_info.append(info)

        # Observe next state and reward
        state = self._get_state()
        done = (t+1 >= self.n_tasks)
        reward = self._get_reward(terminal_state=done)
        return state, reward, done, info #transition

    def _get_state(self):
        res_curr = self.ep_res[-1]['val']
        res_prev = self.ep_res[-2]['val'] if (len(self.ep_res) > 1) else None
        T = self.n_tasks
        t = len(self.ep_res)

        state = res_curr['acc'][:T-1]
        if self.state_add_delta:
            if res_prev is not None:
                """
                accs_prev = res_prev['acc'][:T-1]
                delta = state - accs_prev
                """
                delta = np.zeros(T-1)
                accs_prev = res_prev['acc'][:t-1]
                delta[:t-1] += state[:t-1] - accs_prev
                
            else:
                delta = np.zeros(T-1) #state.copy() #np.zeros(len(state))
            state = np.concatenate([state, delta], axis=-1)

        if self.state_add_delta_max:
            accs = np.array([res['val']['acc'] for res in self.ep_res])
            if res_prev is not None:
                delta_max = np.zeros(T-1)
                accs_max = np.max(accs, axis=0)[:t-1] 
                delta_max[:t-1] += state[:t-1] - accs_max
            else:
                delta_max = np.zeros(T-1)
            state = np.concatenate([state, delta_max], axis=-1)

        if self.state_add_forgetting:
            accs = np.array([res['val']['acc'] for res in self.ep_res])
            accs = accs[:, :T-2] # remove final task
            if accs.shape[0] > 1:
                bwt = np.zeros(T-2)
                bwt[:t] = accs[-1, :t]-np.diag(accs)
                state = np.concatenate([state, bwt], axis=-1)
            else:
                state = np.concatenate([state, np.zeros(accs.shape[1])], axis=-1)

        if self.state_add_time:
            t = len(self.ep_res)-1 # get current time step
            state = np.append(state, [t/T], axis=-1)

        if self.state_add_bwt:
            accs = np.array([res['val']['acc'] for res in self.ep_res])
            accs = accs[:, :t] # remove final task
            if accs.shape[0] > 1:
                bwt = self.compute_bwt(accs)
                state = np.append(state, [bwt], axis=-1)
            else:
                state = np.append(state, [0.0], axis=-1)
        #print('state: ', state)
        #print('state.shape: ', state.shape)
        #print()
        return state

    def save_transition_table(self):
        save_pickle(data=self.transitions, 
                    path=os.path.join(self.transition_table_dir, self.filename_transition_table))

    def compute_bwt(self, acc):
        gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
        return gem_bwt

class EnvTableFromFile(EnvBaseSimple):#EnvTableFromFile(EnvBase):
    """ Training environment for CL classifier which RL algorithm can make calls to
        for recieving observations, rewards, etc. Similar to OpenAI gym environment.
    """

    def __init__(self, table_filename, observation_dim, action_space, dataset, seed, args, shuffle_labels=False):
        super().__init__(observation_dim, action_space, dataset, seed, args, shuffle_labels)
        self.obs_dim = observation_dim
        self.load_checkpoints = args.load_checkpoints
        self.table_filename = table_filename
        #self.seed, self.cl_seed, self.dataset_name = self.get_seed_from_table_filename(filepath=table_filename)
        print('in envtable from file: ', self.table_filename)
        #print(self.seed, self.cl_seed, self.dataset_name)
        self.table = self.get_transition_table(fname=table_filename)
        self.transitions = self.table['data']
        self.dataset_name = self.table['dataset_name']
        self.task_ids = self.table['task_ids']
        ## delete dataset to reduce requirement on memory space
        #del self.dataset

    def get_seed_from_table_filename(self, filepath):
        filename = filepath.split('/')[-1]
        filename = filename.split('.pkl')[0]
        dataset_name = filename.split('-')[0]
        dataset_seed = int(filename.split('-')[-1])
        cl_seed = int(filename.split('-')[-3])
        return dataset_seed, cl_seed, dataset_name

    def get_transition_table(self, fname):
        """ Get dictionary that stores all state transitions. 
        """
        if os.path.exists(fname):
            print('Loading table with transitions: %s' %(fname))
            return load_pickle(fname)
        else:
            raise ValueError('Could not load table {}.\nEnvironment class only works with universal tables... '.format(fname))

    def get_transition(self, t, index):
        #print('in get_transition, index: ', index)
        if index in self.transitions[t].keys(): 
            transition = self.transitions[t][index]
        else:
            raise ValueError('Transition key {} is missing in table from {}'.format(index, self.table_filename))
        return transition

    def reset(self):
        # reset environment 
        self.ep_res, self.ep_info = [], []
        transition = self.get_transition(t=0, index='0')
        self.ep_info.append(transition)
        self.ep_res.append(transition)
        state = self._get_state()
        return state, transition # return transition as info

    def step(self, t, action):
        # Get transition index
        #print('step, action: ', action)
        if t > 1:
            actions = [info['action'] for info in self.ep_info[1:]] #[info['action'] for info in self.ep_info[1:]]
            actions.append(action)
        elif t == 1:
            actions = [0]
        else:
            raise ValueError('Can only get task proportions for task {}>=1'.format(t))
        
        index = '0-' + '-'.join([str(a) for a in actions])
        #print('index: ', index)
        transition = self.get_transition(t=t, index=index)
        info = transition.copy()
        #print('step, action: ', action)
        #print('step, task: ', t)
        info['task_proportion'] = self.action_space.get_action_by_index(t-1, action)
        self.ep_res.append(info)
        self.ep_info.append(info)

        # Observe next state and reward
        state = self._get_state()
        done = (t+1 >= self.n_tasks)
        reward = self._get_reward(terminal_state=done)
        return state, reward, done, info #transition

    def _get_state(self):
        res_curr = self.ep_res[-1]['val']
        res_prev = self.ep_res[-2]['val'] if (len(self.ep_res) > 1) else None
        T = self.n_tasks
        t = len(self.ep_res)

        accs = np.array([res['val']['acc'] for res in self.ep_res])

        state = res_curr['acc'][:T-1]
        if self.state_add_delta:
            if res_prev is not None:
                delta = np.zeros(T-1)
                accs_prev = res_prev['acc'][:t-1]
                delta[:t-1] += state[:t-1] - accs_prev
            else:
                delta = np.zeros(T-1) #state.copy() #np.zeros(len(state))
            state = np.concatenate([state, delta], axis=-1)

        if self.state_add_delta_max:
            if res_prev is not None:
                delta_max = np.zeros(T-1)
                accs_max = np.max(accs, axis=0)[:t-1] 
                delta_max[:t-1] += state[:t-1] - accs_max
            else:
                delta_max = np.zeros(T-1)
            state = np.concatenate([state, delta_max], axis=-1)

        if self.state_add_forgetting:
            accs = np.array([res['val']['acc'] for res in self.ep_res])
            accs = accs[:, :T-2] # remove final task
            if accs.shape[0] > 1:
                bwt = np.zeros(T-2)
                bwt[:t] = accs[-1, :t]-np.diag(accs)
                state = np.concatenate([state, bwt], axis=-1)
            else:
                state = np.concatenate([state, np.zeros(accs.shape[1])], axis=-1)

        if self.state_add_time:
            t = len(self.ep_res)-1 # get current time step
            state = np.append(state, [t/T], axis=-1)

        if self.state_add_bwt:
            accs = np.array([res['val']['acc'] for res in self.ep_res])
            accs = accs[:, :t] # remove final task
            if accs.shape[0] > 1:
                bwt = self.compute_bwt(accs)
                state = np.append(state, [bwt], axis=-1)
            else:
                state = np.append(state, [0.0], axis=-1)
        #print('state: ', state)
        #print('state.shape: ', state.shape)
        #print()
        return state

