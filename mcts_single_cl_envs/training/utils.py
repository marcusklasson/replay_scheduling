
import os
import pickle
import numpy as np

def print_log_acc_bwt(acc, lss, output_path, file_name='logs.p'):
    
    logs = {} # for saving all metrics
    print('*'*100)
    if acc.shape[0] <= 10: # looks awful if printing for 20 tasks
        print('Accuracies =')
        for i in range(acc.shape[0]):
            print('\t',end=',')
            for j in range(acc.shape[1]):
                print('{:5.4f}% '.format(acc[i,j]),end=',')
            print()

    n_tasks = acc.shape[1]
    if acc.shape[0] == acc.shape[1]+1:
        # FWT can only be computed if there is a baseline
        baseline = acc[0] # used for FWT
        acc = acc[1:]
        # FWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
        # fwt[t] equals result[t-1,t] - baseline[t]
        fwt = np.zeros(n_tasks)
        for t in range(1, n_tasks):
            fwt[t] = acc[t - 1, t] - baseline[t]
        gem_fwt = sum(fwt) / (len(acc[-1])-1)
        # save metrics in log
        logs['gem_fwt'] = gem_fwt
        logs['baseline'] = baseline
    else:
        gem_fwt = np.nan

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    print ('ACC: {:5.4f}%'.format(avg_acc))
    #print()
    #print()
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)


    # BWT calculated based on UCB paper (https://arxiv.org/abs/1906.02425)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT: {:5.2f}%'.format(gem_bwt))
    print ('FWT: {:5.2f}%'.format(gem_fwt))
    #print()
    #print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')

    
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

    print ("Log file saved in ", path)
    return avg_acc, gem_bwt, gem_fwt

def load_memory_partition(config):
    """ Loading memory partition and use it for training.
    """
    seed = config['session']['seed']
    log_dir = config['training']['log_dir']

    if config['search']['method'] in ('bfs'):
        filename = os.path.join(log_dir, 'bfs_res_seed%d.p' %(seed))
        results = load_pickle(filename)
        best_task_config = results['best_task_config'] 
        partition = best_task_config['partition']
        return partition

    elif config['search']['method'] in ('mcts'):
        filename = os.path.join(log_dir, 'mcts_res_seed%d.p' %(seed))
        results = load_pickle(filename)
        best_state = results['best_state']

        if config['data']['name'] in ('MNIST', 'FashionMNIST', 'notMNIST'):
            best_task_config = best_state.node_info[-1][best_state.current_index]
            partition = best_task_config['partition']
        elif config['data']['name'] in ('CIFAR100'):
            partition = best_state.memory_budgets
        return partition

    else:
        raise ValueError('Search method %s does not exist.' %(config['search']['method']))

def load_pickle(filename):
    if not os.path.exists(filename):
        print('Warning: file "%s" does not exist!' % filename)
        return
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except EOFError:
        print('Warning: log file corrupted!')

def save_pickle(data, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            #print ('Saved %s..' %path)
    except:
        print('Could not save file %s' %(path))

def compute_gem_bwt(accs):
    return sum(accs[-1]-np.diag(accs))/ (len(accs[-1])-1)