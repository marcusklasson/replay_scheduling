
import torch
import torch.nn as nn
import numpy as np

from itertools import combinations_with_replacement

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class DiscreteActionSpace(object):

    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        self.action_space_dims = self._compute_action_space_dims(n_tasks)
        #self.action_space = dict()

    @property
    def n(self):
        return self.action_space_dims[-1]

    @property
    def shape(self):
        return 1

    def get_dim_at_task(self, task):
        if (task < 0) or (task >= self.n_tasks):
            raise ValueError('Task index {} is out of range for action space of {} tasks!'.format(task, self.n_tasks))
        return self.action_space_dims[task]

    def _compute_action_space_dims(self, n_tasks):
        if n_tasks > 7:
            raise ValueError('The number of actions for T={} tasks in total are too many!'.format(n_tasks))
        
        # Get replay schedules at each task
        action_space_dims = [1]
        a = []
        for t in range(1, n_tasks):
            a.append(t)
            x = list(combinations_with_replacement(a, t))
            #print(x)
            action_space_dims.append(len(x))
        return action_space_dims

# From stable baselines
# https://github.com/DLR-RM/stable-baselines3/blob/3efab0d267e74cb03264411d4500ddde0c163404/stable_baselines3/common/utils.py#L43
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y