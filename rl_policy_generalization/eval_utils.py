
import os
import numpy as np 

from training.utils import load_pickle

def get_ets_baseline_res_from_logs(root_dir, dataset_seeds, val=False):
    res, res_val = {}, {}
    res['acc'], res['bwt'], res['acc_full'] = [], [], []
    res_val['acc'], res_val['bwt'], res_val['acc_full'] = [], [], []
    for dataset_seed in dataset_seeds:
        #path = os.path.join(root_dir, 'logs/ets/seed1/dataset_seed%d' %(dataset_seed), 'logs.p')
        path = os.path.join(root_dir, 'logs/ets/seed%d/dataset_seed%d' %(dataset_seed, dataset_seed), 'logs.p')
        log = load_pickle(path)
        res['acc'].append(np.mean(log['rij']))
        res['acc_full'].append(compute_acc_full(log['acc']))
        res['bwt'].append(log['gem_bwt'])

        if val:
            #path = os.path.join(root_dir, 'logs/ets/seed1/dataset_seed%d' %(dataset_seed), 'logs_val.p')
            path = os.path.join(root_dir, 'logs/ets/seed%d/dataset_seed%d' %(dataset_seed, dataset_seed), 'logs_val.p')
            log = load_pickle(path)
            res_val['acc'].append(np.mean(log['rij']))
            res_val['acc_full'].append(compute_acc_full(log['acc']))
            res_val['bwt'].append(log['gem_bwt'])
    return res, res_val

def get_random_policy_baseline_res_from_logs(root_dir, dataset_seeds, policy_seeds, metric='acc_last', val=False):
    res, res_val = {}, {}
    res['acc'], res['bwt'], res['acc_full'] = [], [], []
    res_val['acc'], res_val['bwt'], res_val['acc_full'] = [], [], []
    for dataset_seed in dataset_seeds:
        accs, accs_full, bwts = [], [], []
        accs_val, accs_full_val, bwts_val = [], [], []

        for policy_seed in policy_seeds:
            #path = os.path.join(root_dir, 'logs/random/seed1/dataset_seed%d/policy_seed%d' %(dataset_seed, policy_seed), 'logs.p')
            path = os.path.join(root_dir, 'logs/random/seed%d/dataset_seed%d/policy_seed%d' %(dataset_seed, dataset_seed, policy_seed), 'logs.p')
            log = load_pickle(path)
            accs.append(np.mean(log['rij']))
            accs_full.append(compute_acc_full(log['acc']))
            bwts.append(log['gem_bwt'])

            if val:
                #path = os.path.join(root_dir, 'logs/random/seed1/dataset_seed%d/policy_seed%d' %(dataset_seed, policy_seed), 'logs_val.p')
                path = os.path.join(root_dir, 'logs/random/seed%d/dataset_seed%d/policy_seed%d' %(dataset_seed, dataset_seed, policy_seed), 'logs_val.p')
                log = load_pickle(path)
                accs_val.append(np.mean(log['rij']))
                accs_full_val.append(compute_acc_full(log['acc']))
                bwts_val.append(log['gem_bwt'])
        # Compute mean over policy seeds
        res['acc'].append(accs)
        res['acc_full'].append(accs_full)
        res['bwt'].append(bwts)
        if val:
            res['acc'].append(accs_val)
            res['acc_full'].append(accs_full_val)
            res['bwt'].append(bwts_val)
    return res, res_val

def get_heuristic_baseline_res_from_logs(root_dir, dataset_seeds, metric='acc_last', val=False):
    res, res_val = {}, {}
    res['acc'], res['bwt'], res['acc_full'] = [], [], []
    res_val['acc'], res_val['bwt'], res_val['acc_full'] = [], [], []
    for dataset_seed in dataset_seeds:
        #path = os.path.join(root_dir, 'seed1/dataset_seed%d' %(dataset_seed), 'logs.p')
        path = os.path.join(root_dir, 'seed%d/dataset_seed%d' %(dataset_seed, dataset_seed), 'logs.p')
        log = load_pickle(path)
        res['acc'].append(np.mean(log['rij']))
        res['acc_full'].append(compute_acc_full(log['acc']))
        res['bwt'].append(log['gem_bwt'])

        if val:
            #path = os.path.join(root_dir, 'seed1/dataset_seed%d' %(dataset_seed), 'logs_val.p')
            path = os.path.join(root_dir, 'seed%d/dataset_seed%d' %(dataset_seed, dataset_seed), 'logs_val.p')
            log = load_pickle(path)
            res_val['acc'].append(np.mean(log['rij']))
            res_val['acc_full'].append(compute_acc_full(log['acc']))
            res_val['bwt'].append(log['gem_bwt'])
    return res, res_val

def get_dqn_res_from_logs(root_dir, dqn_seeds, dataset_seeds, metric='acc_last'):
    test_acc = []
    for dataset_seed in dataset_seeds:
        test_acc_seed = []
        for dqn_seed in dqn_seeds:
            path = os.path.join(root_dir, 'logs/dqn_seed%d/eval/logs_seed%d.p' %(dqn_seed, dataset_seed))
            res_seed = load_pickle(path)

            if metric == 'acc_last':
                test_acc_seed.append(np.mean(res_seed['rij']))
            elif metric == 'acc_full':
                test_acc_seed.append(compute_acc_full(res_seed['acc']))
            elif metric == 'bwt':
                test_acc_seed.append(res_seed['gem_bwt'])

        test_acc.append(test_acc_seed)
    return test_acc 

def compute_acc_full(accs):
    avg_accs_full = np.sum(accs) / np.sum(accs > 0).astype(np.float32)
    return avg_accs_full

def get_best_tau_for_heuristics(dataset_name, n_envs_train, metric=None):
    if dataset_name == 'MNIST':
        if n_envs_train == 10:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.9
            tau_heuristic3 = 0.9
        elif n_envs_train == 20:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
        elif n_envs_train == 30:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
        elif n_envs_train == 40:
            tau_heuristic1 = 0.95
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.95
    elif dataset_name == 'FashionMNIST':
        if n_envs_train == 10:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.95
            tau_heuristic3 = 0.9
        elif n_envs_train == 20:
            tau_heuristic1 = 0.95 
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.999
        elif n_envs_train == 30:
            tau_heuristic1 = 0.95
            tau_heuristic2 = 0.95
            tau_heuristic3 = 0.999
        elif n_envs_train == 40:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.999
    elif dataset_name == 'notMNIST':
        if n_envs_train == 10:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.999
        elif n_envs_train == 20:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.95
        elif n_envs_train == 30:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.95
        elif n_envs_train == 40:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.95
    elif dataset_name == 'CIFAR10':
        if n_envs_train == 10:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
        elif n_envs_train == 20:
            tau_heuristic1 = 0.95
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
        elif n_envs_train == 30:
            tau_heuristic1 = 0.95
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
        elif n_envs_train == 40:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
    return tau_heuristic1, tau_heuristic2, tau_heuristic3

def get_best_tau_for_heuristics_mixed_datasets(target_dataset_name, n_envs_train, metric=None):
    if target_dataset_name == 'notMNIST':
        if n_envs_train == 10:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.95
            tau_heuristic3 = 0.9
        elif n_envs_train == 20:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.95
            tau_heuristic3 = 0.9
        elif n_envs_train == 30:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.95
            tau_heuristic3 = 0.9
        elif n_envs_train == 40:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.95
            tau_heuristic3 = 0.9
        elif n_envs_train == 60:
            tau_heuristic1 = 0.95
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
        elif n_envs_train == 80:
            tau_heuristic1 = 0.95
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
    if target_dataset_name == 'FashionMNIST':
        if n_envs_train == 10:
            tau_heuristic1 = 0.9
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.95
        elif n_envs_train == 20:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.95
        elif n_envs_train == 30:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.999
        elif n_envs_train == 40:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
        elif n_envs_train == 60:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.9
        elif n_envs_train == 80:
            tau_heuristic1 = 0.999
            tau_heuristic2 = 0.999
            tau_heuristic3 = 0.95
    return tau_heuristic1, tau_heuristic2, tau_heuristic3

def get_dict_with_seed_to_task_ids():
    seed_to_task_ids = {}
    seed_to_task_ids[0] = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]] 
    seed_to_task_ids[1] = [[2, 9], [6, 4], [0, 3], [1, 7], [8, 5]] 
    seed_to_task_ids[2] = [[4, 1], [5, 0], [7, 2], [3, 6], [9, 8]] 
    seed_to_task_ids[3] = [[5, 4], [1, 2], [9, 6], [7, 0], [3, 8]] 
    seed_to_task_ids[4] = [[3, 8], [4, 9], [2, 6], [0, 1], [5, 7]] 
    seed_to_task_ids[5] = [[9, 5], [2, 4], [7, 1], [0, 8], [6, 3]] 
    seed_to_task_ids[6] = [[8, 1], [7, 0], [6, 5], [2, 4], [3, 9]] 
    seed_to_task_ids[7] = [[8, 5], [0, 2], [1, 9], [7, 3], [6, 4]] 
    seed_to_task_ids[8] = [[8, 6], [9, 0], [2, 5], [7, 1], [4, 3]] 
    seed_to_task_ids[9] = [[8, 4], [7, 2], [1, 9], [3, 0], [6, 5]] 
    seed_to_task_ids[10] = [[8, 2], [5, 6], [3, 1], [0, 7], [4, 9]] 
    seed_to_task_ids[11] = [[7, 8], [2, 6], [4, 5], [1, 3], [0, 9]] 
    seed_to_task_ids[12] = [[5, 8], [7, 0], [4, 9], [3, 2], [1, 6]] 
    seed_to_task_ids[13] = [[3, 5], [6, 1], [4, 7], [8, 9], [0, 2]] 
    seed_to_task_ids[14] = [[3, 9], [0, 5], [4, 2], [1, 7], [6, 8]] 
    seed_to_task_ids[15] = [[2, 6], [1, 3], [7, 0], [9, 4], [5, 8]] 
    seed_to_task_ids[16] = [[6, 2], [0, 7], [8, 4], [3, 1], [5, 9]] 
    seed_to_task_ids[17] = [[7, 2], [5, 3], [4, 0], [9, 8], [6, 1]] 
    seed_to_task_ids[18] = [[7, 9], [0, 4], [2, 1], [6, 5], [8, 3]] 
    seed_to_task_ids[19] = [[1, 7], [9, 6], [8, 4], [3, 0], [2, 5]] 
    return seed_to_task_ids