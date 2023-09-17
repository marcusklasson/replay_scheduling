
import io
import os
import pickle
import random
import numpy as np
import torch
from itertools import combinations_with_replacement

def print_log_acc_bwt(acc, lss, output_path, file_name='logs.p', verbose=1):

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    avg_acc_full = np.sum(acc) / np.sum(acc > 0).astype(np.float32)
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
    # BWT calculated based on UCB paper (https://arxiv.org/abs/1906.02425)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()

    logs = {}
    # save results
    logs['name'] = output_path
    logs['acc'] = acc
    logs['loss'] = lss
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc) #Task accuracy after training
    logs['rij'] = acc[-1]  # Task accuracy after training on final task

    # pickle
    path = os.path.join(output_path, file_name)
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    if verbose > 0:
        print('*'*100)
        if acc.shape[0] <= 10: # looks awful if printing for 20 tasks
            print('Accuracies =')
            for i in range(acc.shape[0]):
                print('\t',end=',')
                for j in range(acc.shape[1]):
                    print('{:5.4f}% '.format(acc[i,j]),end=',')
                print()
        print ('ACC: {:5.4f}%'.format(avg_acc))
        print ('ACC (full): {:5.4f}%'.format(avg_acc_full))
        print ('BWT: {:5.2f}%'.format(gem_bwt))
        print()
        #print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))
        print('*'*100)
        print('Done!')

        print ("Log file saved in ", path)

    return avg_acc, gem_bwt

def set_random_seed(seed):
    # Set random seed
    #print('Set random seed %d...' %(seed))
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        # Faster run but not deterministic:
        #torch.backends.cudnn.benchmark = True

        # To get deterministic results that match with paper at cost of lower speed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)
    else:
        print('File {:s} does not exist'.format(path))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_percentage_for_memory_bin(x, num_tasks):
    p = dict((y, np.count_nonzero(x == y)/len(x)) for y in range(1, num_tasks))
    s = np.sum(list(p.values()))
    assert np.isclose(s, 1), 'values = {} should sum to {}'.format(s, 1)
    return p

def get_schedule_mapping(num_tasks):
    """ Get schedules in dictionary acquired by task and index.
    """
    
    # Get replay schedules at each task
    schedules = {}
    a = []
    for t in range(1, num_tasks):
        schedules[t] = {}
        a.append(t)
        x = list(combinations_with_replacement(a, t))
        for index, x1 in enumerate(x):
            tmp = np.array([int(u) for u in list(x1)])
            schedules[t][index] = tmp
    # Get action space
    action_space = {0:1}
    for t in range(1, num_tasks):
        action_space[t] = len(schedules[t])
        
    return schedules, action_space

class CPU_Unpickler(pickle.Unpickler):
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_pickle(filename):
    if not os.path.exists(filename):
        print('Warning: file "%s" does not exist!' % filename)
        return
    try:
        with open(filename, 'rb') as f:
            return CPU_Unpickler(f).load() #return pickle.load(f)
    except EOFError:
        print('Warning: log file corrupted!')

"""
def load_pickle(filename):
    if not os.path.exists(filename):
        print('Warning: file "%s" does not exist!' % filename)
        return
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except EOFError:
        print('Warning: log file corrupted!')
"""

def save_pickle(data, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            #print ('Saved %s..' %path)
    except:
        print('Could not save file %s' %(path))