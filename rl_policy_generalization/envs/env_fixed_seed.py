
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from envs.base import EnvBase
from envs.utils import get_filename_for_table
from training.utils import load_pickle, save_pickle, set_random_seed
from envs.action_spaces import DiscreteActionSpace, ContinuousActionSpace

class EnvFixedSeed(EnvBase):
    """ Training environment for CL classifier which RL algorithm can make calls to
        for recieving observations, rewards, etc. Similar to OpenAI gym environment.
    """

    def __init__(self, observation_dim, action_space, dataset, seed, args, shuffle_labels=False):
        super().__init__(observation_dim, action_space, dataset, seed, args, shuffle_labels)
        self.load_checkpoints = args.load_checkpoints
        self.save_transitions = args.save_transitions
        self.filename_transition_table = get_filename_for_table(args, dataset_name=self.dataset_name, task_ids=self.task_ids, seed=self.seed)
        self.transition_table_dir = args.table_dir
        #create_dir(self.transition_table_dir)
        self.transitions = {t:dict() for t in range(self.n_tasks)}
        #print(self.transitions)

    def reset(self):
        # initialize CL network and its optimizer
        self.trainer.reset()
        self.ep_res, self.ep_info = [], []

        # train on first task
        task_dataset = self.dataset.get_dataset_for_task(task_id=0)
        #self.train_model_at_task(t=0, train_dataset=task_dataset['train'])
        self.trainer.train_model_at_task(t=0, train_dataset=task_dataset['train'], actions=[])
        res = self.evaluate_model_at_task(current_task=0)
        self.ep_res.append(res) # used for getting state
        
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
        self.ep_info.append(info)
        #self.save_transition(cmd='reset', info=info)
        return state, info

    def step(self, t, action):
        """ return observation (state), reward, done (boolean), info (dict with training info)
        """
        # Get current and previous task proportions
        #print('in env, step: ', action)
        current_task_proportion, task_proportions, actions = self.get_task_proportions(t, action)

        if self.verbose > 1:
            print('task_props: ', task_proportions)
            print('actions: ', actions)
        task_dataset = self.dataset.get_dataset_for_task(task_id=t)
        #self.train_model_at_task(t, task_dataset['train'], action, current_task_proportion)
        self.trainer.train_model_at_task(t, task_dataset['train'], actions, current_task_proportion)
        res = self.evaluate_model_at_task(current_task=t)
        self.ep_res.append(res) # used for getting state

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
                'accs': res['accs'],
                'losses': res['losses'],
                'task': t, 
                } 
        self.ep_info.append(info)
        if self.verbose > 1:
            if t+1 == self.n_tasks:
                print(np.stack(res['accs']['test'], axis=0))
        #self.save_transition(cmd='step', info=info)
        return state, reward, done, info

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
                #max_acc = np.max(accs, axis=0)
                #task_forgetting = max_acc - accs[-1]
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
        return state

    def evaluate_model_at_task(self, current_task):
        # Evaluate model
        res = {}
        acc_test, acc_val = np.zeros(self.n_tasks), np.zeros(self.n_tasks)
        loss_test, loss_val = np.zeros(self.n_tasks), np.zeros(self.n_tasks) 
        for u in range(current_task+1):
            task_dataset = self.dataset.get_dataset_for_task(u)
            val_res = self.trainer.eval_task(u, task_dataset['valid']) # get task accuracy
            test_res = self.trainer.eval_task(u, task_dataset['test']) # get task accuracy
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

        if current_task > 0:
            accs_test = [info['test']['acc'].copy() for info in self.ep_info]
            accs_test.append(acc_test)
            accs_val = [info['val']['acc'].copy() for info in self.ep_info]
            accs_val.append(acc_val)
            res['accs'] = {'test': accs_test, 'val': accs_val}

            losses_test = [info['test']['loss'].copy() for info in self.ep_info]
            losses_test.append(loss_test)
            losses_val = [info['val']['loss'].copy() for info in self.ep_info]
            losses_val.append(loss_val)
            res['losses'] = {'test': losses_test, 'val': losses_val}
        return res

    """
    def train_model_at_task(self, t, train_dataset, action=0, task_proportion=None):

        # get actions
        if t > 1:
            actions = [info['action'] for info in self.ep_info[1:]]
            actions.append(action)
        elif t == 1:
            actions = [0]
        else:
            actions = []

        # Train on task t
        if t <= len(actions):
            print('actions: ', actions)
            path_exists, filename = self.model_checkpoint_exists(t, actions)  
            print('filename: ', filename)
            path_exists = path_exists if self.load_checkpoints else False
        else:
            # if all actions could not be stored becuase they are too many
            filename = 'model_rollout_id_{}_task_{}.pth.tar'.format(rollout_id, t+1)
            path_exists = False
        if path_exists:
            #print('loading checkpoint {}'.format(checkpoint_dir + '/' + filename))
            self.trainer.load_checkpoint(checkpoint_dir=self.checkpoint_dir, file_path=filename)
            self.trainer.update_coreset(train_dataset, t) # update memory since doesn't go in to train_single_task()
            #self.trainer.episodic_filled_counter += self.trainer.memories_per_class * self.trainer.classes_per_task
        else:
            self.trainer.train_single_task(t, train_dataset, task_proportion)
            # Save checkpoint
            self.trainer.save_checkpoint(task_id=t, folder=self.checkpoint_dir, file_name=filename)
        print('filename: ', filename)
        self.trainer.model = self.trainer.load_model_from_file(file_name=filename) # uses checkpoint_dir inside function

    def model_checkpoint_exists(self, task, actions=None):
        # Check if model checkpoint exists at task.
        if task == 0:
            model_path = 'model_0.pth.tar'
        else:    
            assert task == len(actions) # check that number of seen actions are the same as the task id
            indexing = '0-' + '-'.join([str(a) for a in actions])
            model_path = 'model_{}.pth.tar'.format(indexing)
        path = os.path.join(self.checkpoint_dir, model_path)
        return os.path.exists(path), model_path

    """

    def get_task_proportions(self, t, action):
        # get actions
        if t > 1:
            if isinstance(self.action_space, DiscreteActionSpace):
                actions = [info['action'] for info in self.ep_info[1:]]
                actions.append(action)
            else:
                actions = [np.round(act, 2) for act in [info['action'] for info in self.ep_info[1:]]]
                action = [round(a, 2) for a in action]
                actions.append(action)
        elif t == 1:
            if isinstance(self.action_space, DiscreteActionSpace):
                actions = [0]
            else:
                actions = [action]
        else:
            raise ValueError('Can only get task proportions for task {}>=1'.format(t))

        task_proportions = []
        for t, a in enumerate(actions):
            #print('a: ', a)
            if isinstance(self.action_space, ContinuousActionSpace):
                task_prop = self.action_space.get_task_proportion_from_action(a.copy(), t) # input action copy otherwise it gets changed
                #print(a)
                #task_prop = F.softmax(a, dim=-1).cpu().numpy()
            #else:
            elif isinstance(self.action_space, DiscreteActionSpace):
                task_prop = self.action_space.get_action_by_index(t, a)
            else:
                task_prop = a # the action is the actual proportion
            task_proportions.append(task_prop)
        #print(actions)
        return task_proportions[-1], task_proportions, actions
        

    def save_transition(self, cmd, info):

        if cmd == 'reset':
            t = info['task']
            index = '0'
            info['checkpoint'] = 'model_0.pth.tar'
            self.transitions[t][index] = info 
        elif cmd == 'step':   
            t = info['task']
            actions = info['actions'] 
            #print('actions: ', actions)
            assert t == len(actions) # check that number of seen actions are the same as the task id
            index = '0-' + '-'.join([str(a) for a in actions])
            info['checkpoint'] = 'model_{}.pth.tar'.format(index)
            self.transitions[t][index] = info 
        else:
            raise ValueError('Command {} does not exist.'.format(cmd))
        
    def save_transition_table(self):
        #print('table: ', self.transitions)
        #print('path: ', os.path.join(self.transition_table_dir, self.filename_transition_table))
        save_pickle(data=self.transitions, 
                    path=os.path.join(self.transition_table_dir, self.filename_transition_table))
