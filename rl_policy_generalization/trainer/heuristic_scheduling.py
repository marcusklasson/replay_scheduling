
import os
import time 
import numpy as np
import torch

from trainer.rs import ReplaySchedulingTrainer

class HeuristicSchedulingTrainer(ReplaySchedulingTrainer):
    """ Trainer for Heuristic Scheduling. 
    """

    def __init__(self, args):
        super().__init__(args)
        self.baseline_policy = args.baseline_policy
        self.tau = args['replay']['tau'] # used in all heuristics
        self.equal_replay_if_no_selected = True if ('pp' in self.baseline_policy) else False
        self.replay_schedule = []

    def get_next_task_proportions(self, task_id, val_accs):
        props = [0.0]*(self.n_tasks-1)
        replay_tasks = []
        current_val_accs = val_accs[task_id, :]
        print(current_val_accs)
        #prev_val_accs = val_accs[task_id-1, :]
        #print(val_accs)
        #print(current_val_accs)
        #print(max_val_accs)
        print('Select tasks for replay: ')
        if self.baseline_policy in ['heuristic1', 'heuristic1pp']:
            # Add replay task based on if current acc is below threshold based on top task acc
            max_val_accs = np.max(val_accs, axis=0)[:task_id+1]
            for t, acc in enumerate(current_val_accs[:task_id+1]):
                threshold = max_val_accs[t] * self.tau
                print('Task {} performance: {:.4f} < {:.4f}*{} = {}'.format(t+1, acc, max_val_accs[t], self.tau, acc <= threshold))
                if acc <= threshold:
                    replay_tasks.append(t)
        elif self.baseline_policy in ['heuristic2', 'heuristic2pp']:
            # Add replay task based on if current acc is below threshold based on previous task acc
            prev_val_accs = val_accs[task_id-1, :] if task_id>0 else np.zeros(self.n_tasks)
            for t, acc in enumerate(current_val_accs[:task_id+1]):
                threshold = prev_val_accs[t] * self.tau
                print('Task {} performance: {:.4f} < {:.4f}*{} = {}'.format(t+1, acc, prev_val_accs[t], self.tau, acc <= threshold))
                if acc <= threshold:
                    replay_tasks.append(t)
        elif self.baseline_policy in ['heuristic3', 'heuristic3pp']:
            # Add replay task based on if current acc is below threshold 
            #print(current_val_accs)
            for t, acc in enumerate(current_val_accs[:task_id+1]):
                #print(t, acc)
                threshold = self.tau
                print('Task {} performance: {:.4f} < {:.4f} = {}'.format(t+1, acc, self.tau, acc <= threshold))
                if acc <= threshold:
                    replay_tasks.append(t)
        else:
            raise ValueError('Baseline %s does not exist...' %(self.baseline_policy))

        # calculate proportion
        if len(replay_tasks) > 0:
            for t in replay_tasks:
                props[t] = 1/len(replay_tasks)
            assert np.isclose(np.sum(props), 1.0), 'proportion values = {} should sum to {}'.format(props, 1)
        elif (len(replay_tasks) == 0) and self.equal_replay_if_no_selected:
            print('replay all tasks equally since no selected replay tasks')
            for t in range(task_id+1):
                props[t] = 1/(task_id+1)
            assert np.isclose(np.sum(props), 1.0), 'proportion values = {} should sum to {}'.format(props, 1)
        # add task proportion to replay schedule
        self.replay_schedule.append(props)
        return props