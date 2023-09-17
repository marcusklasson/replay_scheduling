
import os
import itertools 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as ss

from matplotlib import cm
from matplotlib import ticker
import tikzplotlib as tpl

from training.utils import load_pickle
from training.vis_utils import plot_grouped_bar_chart, plot_return_over_episodes
from eval_utils import get_best_tau_for_heuristics, get_best_tau_for_heuristics_mixed_datasets
from eval_utils import get_dict_with_seed_to_task_ids
from eval_utils import get_ets_baseline_res_from_logs, get_random_policy_baseline_res_from_logs
from eval_utils import get_heuristic_baseline_res_from_logs

def get_algo_res_from_logs(root_dir, algo_seeds, dataset_seeds, algo='a2c', metric='acc_last'):
    test_acc = []
    for dataset_seed in dataset_seeds:
        test_acc_seed = []
        for algo_seed in algo_seeds:
            path = os.path.join(root_dir, 'logs/%s_seed%d/eval/logs_seed%d.p' %(algo, algo_seed, dataset_seed))
            res_seed = load_pickle(path)
            if metric == 'acc_last':
                test_acc_seed.append(np.mean(res_seed['rij']))
            elif metric == 'bwt':
                test_acc_seed.append(res_seed['gem_bwt'])
        test_acc.append(test_acc_seed)
    return test_acc 

def get_res_from_algo_logs(root_dir, seeds, algo):
    return_per_episode = []
    best_return_eps_greedy = []
    return_policy_net = []
    return_policy_net_seeds = {}
    parsed_res = []
    for seed in seeds:
        path = os.path.join(root_dir, 'logs/%s_seed%d' %(algo, seed), '%s_stats.p' %(algo)) #os.path.join(root_dir, 'logs/dqn_seed%d' %(seed), 'dqn_stats.p')
        res = load_pickle(path)
        parsed_res1 = get_parse_results(res, algo=algo)
        #print(parsed_res1['ACC'].shape)
        parsed_res.append(parsed_res1) # list with dicts
    return parsed_res #{'return_policy_net': return_policy_net, 'return_policy_net_seeds': return_policy_net_seeds}

def get_parse_results(res, algo):
    accs = []
    returns = []
    for k in res.keys():
        #print(res.keys())
        if ('seed' in k) and (algo == 'dqn'):
            v = res[k]
            accs.append(v['ACC'])
            returns.append(v['return'])
        elif ('valid_env' in k) and (algo in ['a2c', 'ppo']):
            v = res[k]
            accs.append(v['ACC'])
            returns.append(v['return'])
        else:
            continue
    parsed_res = {'ACC': np.stack(accs, axis=0), 'return': np.stack(returns, axis=0)}
    return parsed_res

def plot_rewards_for_environment(algos_accs_envs, random_envs, ets_env, 
                                heuristic1_env, heuristic2_env, heuristic3_env,
                                label_seq=None, save_dir='./', fname='results'):
    fig, ax = plt.subplots()
    # DQN res
    N = 100 # number of eval steps for x-axis
    for k, accs in algos_accs_envs.items():
        idx = np.round(np.linspace(0, len(accs[0]) - 1, N)).astype(int)
        accs1 = accs[:, idx]
        x = np.linspace(1, N, N)
        ax.plot(x, np.mean(accs1, axis=0), label=k, linestyle='solid')
        #ax.fill_between(x, np.mean(accs, axis=0)+np.std(accs, axis=0), 
        #                np.mean(accs, axis=0)-np.std(accs, axis=0), alpha=0.25)
    # plot baselines
    ax.plot([1, N], [np.mean(random_policy_env), np.mean(random_policy_env)], label='Random')
    #ax.fill_between(np.arange(1, n_episodes), np.mean(random_policy_env)+np.std(random_policy_env), 
    #                np.mean(random_policy_env)-np.std(random_policy_env), alpha=0.25)
    ax.plot([1, N], [np.mean(ets_env), np.mean(ets_env)], label='ETS')
    ax.plot([1, N], [np.mean(heuristic1_env), np.mean(heuristic1_env)], label='Heur-GD')
    ax.plot([1, N], [np.mean(heuristic2_env), np.mean(heuristic2_env)], label='Heur-LD', linestyle='dashdot')
    ax.plot([1, N], [np.mean(heuristic3_env), np.mean(heuristic3_env)], label='Heur-AT', linestyle='dashed')

    ax.grid(alpha=0.5)
    ax.set_title('Labels: %s' %(label_seq), fontsize=10) 
    ax.set_ylabel('ACC')
    ax.set_xlabel('Eval. Steps')
    ax.legend()
    plt.tight_layout()
    """
    tpl.save(
        os.path.join(save_dir, fname+'.tex' ),  # this is name of the file where your code will lie
        axis_width="\\figwidth",  # we want LaTeX to take care of the width
        axis_height="\\figheight",  # we want LaTeX to take care of the height
        # if the figure contains an image in the background (this one doesn't), this is where LaTeX (!) should search for the png.
        tex_relative_path_to_data="./",
        # we want the plot to look *exactly* like here (e.g. axis limits, axis ticks, etc.)
        strict=True,
    )
    """
    plt.savefig(os.path.join(save_dir, fname+'.png' ))
    plt.close()

def compute_rankings(scores, names, seed):
    ranks = []
    for j in range(scores.shape[-1]):
        scores_j = scores[:, j]
        rank = ss.rankdata(1-scores_j, method='average') # use average to handle ties
        ranks.append(rank)
    ranks = np.array(ranks).T
    return ranks

## Settings
dataset_name = 'MNIST' # Change dataset in test envs: ['MNIST' 'FashionMNIST', 'notMNIST', 'CIFAR10']
experiment = 'new_task_order' # Change experiment type: [new_task_order, new_dataset]
###
algo_seeds = list(range(1,6)) #list(range(1,4)) 
seed_to_task_ids = get_dict_with_seed_to_task_ids()
print_environment_tables = False # for printing ACCs and ranks per environment for all methods and seeds 
print_welch_ttest_results = False # for printing Welch's t-tests for every test environment 
plot_environment_rewards = False # for printing the progress of ACCs in test environments  
if plot_environment_rewards:
    save_dir = './results/rewards_%s_%s' %(experiment, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

if experiment == 'new_task_order':
    dataset_seeds = list(range(10,20))
    if dataset_name in ['MNIST', 'notMNIST', 'CIFAR10']:
        n_train_envs_baselines = 10 
    elif dataset_name == 'FashionMNIST':
        n_train_envs_baselines = 20 

    tau_heuristic1, tau_heuristic2, tau_heuristic3 = get_best_tau_for_heuristics(dataset_name, n_envs_train=n_train_envs_baselines)
else: # new_dataset
    dataset_seeds = list(range(0,10))
    if dataset_name in ['notMNIST',]:
        n_train_envs_baselines = 10 
    elif dataset_name == 'FashionMNIST':
        n_train_envs_baselines = 10 
    
    tau_heuristic1, tau_heuristic2, tau_heuristic3 = get_best_tau_for_heuristics_mixed_datasets(dataset_name, 
                                                                n_envs_train=n_train_envs_baselines)
                                                        
print('Heuristic hyperparams - tau1 = {}, tau2 = {}, tau3 = {}'.format(tau_heuristic1, tau_heuristic2, tau_heuristic3))

# Get baseline results
root_dir = './results/baselines/%s_baselines' %(dataset_name)
baseline_res = {}
random_policy_res, _ = get_random_policy_baseline_res_from_logs(root_dir, dataset_seeds, policy_seeds=algo_seeds)

baseline_res['random'] = {'mean': np.mean(random_policy_res['acc']), 'std': np.std(random_policy_res['acc'])} 
ets_baseline_res, _ = get_ets_baseline_res_from_logs(root_dir, dataset_seeds)
baseline_res['ets'] = {'mean': np.mean(ets_baseline_res['acc']), 'std': np.std(ets_baseline_res['acc'])}

heuristic1_res, _ = get_heuristic_baseline_res_from_logs(root_dir + '/logs/heuristic1/threshold_%s' %(tau_heuristic1), dataset_seeds)
baseline_res['heuristic1'] = {'mean': np.mean(heuristic1_res['acc']), 'std': np.std(heuristic1_res['acc'])}

heuristic2_res, _ = get_heuristic_baseline_res_from_logs(root_dir + '/logs/heuristic2/threshold_%s' %(tau_heuristic2), dataset_seeds)
baseline_res['heuristic2'] = {'mean': np.mean(heuristic2_res['acc']), 'std': np.std(heuristic2_res['acc'])}

heuristic3_res, _ = get_heuristic_baseline_res_from_logs(root_dir + '/logs/heuristic3/threshold_%s' %(tau_heuristic3), dataset_seeds)
baseline_res['heuristic3'] = {'mean': np.mean(heuristic3_res['acc']), 'std': np.std(heuristic3_res['acc'])}

algo_test_accs, algo_all_accs_progress = {}, {}
algo_all_accs, algo_all_bwts = {}, {}

# Get DQN results
algo = 'dqn'
n_train_envs = 10
n_val_envs = 10
dataset_seeds_algo = [x+1 for x in list(range(10,20))]
if experiment == 'new_task_order':
    if dataset_name == 'MNIST':
        n_train_envs = 30
        root_dir = './results/%s/mnist/%s' %(experiment, algo)
    elif dataset_name == 'FashionMNIST':
        n_train_envs = 20
        root_dir = './tmlr_results/%s/fashionmnist/%s' %(experiment, algo)
    elif dataset_name == 'notMNIST':
        n_train_envs = 30
        root_dir = './tmlr_results/%s/notmnist/%s' %(experiment, algo)
    elif dataset_name == 'CIFAR10':
        n_train_envs = 10
        root_dir = './tmlr_results/%s/cifar10/%s' %(experiment, algo)
    algo_dir = root_dir + '/train_envs_%d' %(n_train_envs)
else:
    if dataset_name == 'notMNIST':
        n_train_envs = 30
        root_dir = './tmlr_results/%s/notmnist/%s' %(experiment, algo)
    elif dataset_name == 'FashionMNIST':
        n_train_envs = 20
        root_dir = './tmlr_results/%s/fashionmnist/%s' %(experiment, algo)
        dataset_seeds_algo = [x for x in list(range(10,20))]
    algo_dir = root_dir + '/train_envs_%d' %(n_train_envs)

key = 'DQN'
algo_all_accs[key] = get_algo_res_from_logs(algo_dir, algo_seeds, dataset_seeds_algo, algo=algo)
algo_all_bwts[key] = get_algo_res_from_logs(algo_dir, algo_seeds, dataset_seeds_algo, algo=algo, metric='bwt')

# Get A2C results
algo = 'a2c'
dataset_seeds_algo = [x+1 for x in list(range(10,20))]
if experiment == 'new_task_order':
    a2c_n_steps = 5
    if dataset_name == 'MNIST':
        a2c_lr = 0.0001
        n_train_envs = 10
        root_dir = './results/%s/mnist/%s' %(experiment, algo)
    elif dataset_name == 'FashionMNIST':
        a2c_lr = 0.0001
        a2c_n_steps = 4
        n_train_envs = 40
        root_dir = './tmlr_results/%s/fashionmnist/%s' %(experiment, algo)
        dataset_seeds_algo = [x for x in list(range(10,20))]
    elif dataset_name == 'notMNIST':
        a2c_lr = 0.0001
        n_train_envs = 10
        root_dir = './tmlr_results/%s/notmnist/%s' %(experiment, algo)
    elif dataset_name == 'CIFAR10':
        a2c_lr = 0.0003
        n_train_envs = 10
        root_dir = './tmlr_results/%s/cifar10/%s' %(experiment, algo)
    algo_dir = root_dir + '/n_steps_%d_lr_%s' %(a2c_n_steps, str(a2c_lr))
else:
    a2c_n_steps = 5
    if dataset_name == 'notMNIST':
        a2c_lr = 0.0001
        n_train_envs = 10
        root_dir = './tmlr_results/%s/notmnist/%s' %(experiment, algo)
    elif dataset_name == 'FashionMNIST':
        a2c_lr = 0.0003
        a2c_n_steps = 4
        n_train_envs = 30
        root_dir = './tmlr_results/%s/fashionmnist/%s' %(experiment, algo)
        dataset_seeds_algo = [x for x in list(range(10,20))]
    algo_dir = root_dir + '/n_steps_%d_lr_%s' %(a2c_n_steps, str(a2c_lr))
    
key = 'A2C'
algo_all_accs[key] = get_algo_res_from_logs(algo_dir, algo_seeds, dataset_seeds_algo, algo=algo)
algo_all_bwts[key] = get_algo_res_from_logs(algo_dir, algo_seeds, dataset_seeds_algo, algo=algo, metric='bwt')

## Compute ranking
rankings, table = {}, {}
for i, seed in enumerate(dataset_seeds): #for i, (seed, label_seq) in enumerate(seed_to_task_ids.items()): #range(n_val_envs):
    label_seq = seed_to_task_ids[seed]   
    names, scores, forgetting = [], [], []
    # Get baseline env results
    random_policy_env = random_policy_res['acc'][i]
    names.append('Random')
    scores.append(random_policy_env)
    forgetting.append(random_policy_res['bwt'][i])

    ets_env = ets_baseline_res['acc'][i]
    ets_env = [ets_env]*len(algo_seeds)
    names.append('ETS')
    scores.append(ets_env)
    forgetting.append([ets_baseline_res['bwt'][i]]*len(algo_seeds))

    heuristic1_env = heuristic1_res['acc'][i]
    heuristic1_env = [heuristic1_env]*len(algo_seeds)
    names.append('Heuristic1')
    scores.append(heuristic1_env)
    forgetting.append([heuristic1_res['bwt'][i]]*len(algo_seeds))

    heuristic2_env = heuristic2_res['acc'][i]
    heuristic2_env = [heuristic2_env]*len(algo_seeds)
    names.append('Heuristic2')
    scores.append(heuristic2_env)
    forgetting.append([heuristic2_res['bwt'][i]]*len(algo_seeds))
    

    heuristic3_env = heuristic3_res['acc'][i]
    heuristic3_env = [heuristic3_env]*len(algo_seeds)
    names.append('Heuristic3')
    scores.append(heuristic3_env)
    forgetting.append([heuristic3_res['bwt'][i]]*len(algo_seeds))

    # get DQN and A2C results
    algo_accs_progress_env = {}
    algo_accs_env, algo_bwts_env = {}, {}
    for k, accs in algo_all_accs.items():
        algo_accs_env[k] = accs[i]
    for k, accs in algo_accs_env.items():
        names.append(k)
        scores.append(accs)
    
    for k, bwts in algo_all_bwts.items():
        algo_bwts_env[k] = bwts[i]
    for k, bwts in algo_bwts_env.items():
        forgetting.append(bwts)

    # Get rankings 
    scores = np.array(scores)
    names = np.array(names)
    forgetting = np.array(forgetting)

    ranks = compute_rankings(scores, names, seed)
    rankings[seed] = ranks 

    # create massive table
    table[seed] = {}
    for name, rank, score, bwt in zip(names.tolist(), ranks.tolist(), scores.tolist(), forgetting.tolist()):
        table[seed][name] = {}
        table[seed][name]['ranks'] = rank
        table[seed][name]['acc'] = score
        table[seed][name]['bwt'] = bwt
        table[seed][name]['avg_rank'] = np.mean(rank)
    # Plotting
    if plot_environment_rewards:
        plot_rewards_for_environment(algo_accs_progress_env, random_policy_env, ets_env,
                            heuristic1_env, heuristic2_env, heuristic3_env,
                            label_seq=label_seq, save_dir=save_dir, fname='rewards_env%d' %(i+1))
    
if print_environment_tables:
    print('\n**** ENVIRONMENT TABLES ****')
    #print(table)
    for seed, table_seed in table.items():
        print('Test Env. Seed %d' %(seed))
        print('Method, ACC (\%), BWT (\%), Rank')
        for name, table_name in table_seed.items(): 
            if 'heuristic1' in name.lower():
                name = 'Heur-GD'
            elif 'heuristic2' in name.lower():
                name = 'Heur-LD'
            elif 'heuristic3' in name.lower():
                name = 'Heur-AT'
            
            s = '%s, ' %(name)
            s += '%.2f $\pm$ %.2f, ' %(100*np.mean(table_name['acc']), 100*np.std(table_name['acc']))
            s += '%.2f $\pm$ %.2f, ' %(100*np.mean(table_name['bwt']), 100*np.std(table_name['bwt']))
            s += '%.2f' %(table_name['avg_rank'])
            print(s)
        print()

if print_welch_ttest_results:
    print('Statistical significance: Welch t-tests')
    for seed, table_name_metric in table.items():
        print('Test Env. Seed %d' %(seed))
        welch_ttest = ss.ttest_ind(table_name_metric['DQN']['acc'], table_name_metric['Random']['acc'], equal_var=False)
        print('DQN vs Random, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['DQN']['acc'], table_name_metric['ETS']['acc'], equal_var=False)
        print('DQN vs ETS, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['DQN']['acc'], table_name_metric['Heuristic1']['acc'], equal_var=False)
        print('DQN vs Heur-GD, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['DQN']['acc'], table_name_metric['Heuristic2']['acc'], equal_var=False)
        print('DQN vs Heur-LD, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['DQN']['acc'], table_name_metric['Heuristic3']['acc'], equal_var=False)
        print('DQN vs Heur-AT, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['DQN']['acc'], table_name_metric['A2C']['acc'], equal_var=False)
        print('DQN vs A2C, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        print()

        welch_ttest = ss.ttest_ind(table_name_metric['A2C']['acc'], table_name_metric['Random']['acc'], equal_var=False)
        print('A2C vs Random, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['A2C']['acc'], table_name_metric['ETS']['acc'], equal_var=False)
        print('A2C vs ETS, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['A2C']['acc'], table_name_metric['Heuristic1']['acc'], equal_var=False)
        print('A2C vs Heur-GD, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['A2C']['acc'], table_name_metric['Heuristic2']['acc'], equal_var=False)
        print('A2C vs Heur-LD, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['A2C']['acc'], table_name_metric['Heuristic3']['acc'], equal_var=False)
        print('A2C vs Heur-AT, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))

        welch_ttest = ss.ttest_ind(table_name_metric['A2C']['acc'], table_name_metric['DQN']['acc'], equal_var=False)
        print('A2C vs DQN, t=%.2f, pval=%.3f' %(welch_ttest.statistic, welch_ttest.pvalue))
        print()

# Get ranking table
print('\n***** Average Ranking Table *****')
# Get Rank averages
avg_ranks = []
for seed, ranks in rankings.items():
    avg_ranks.append(np.mean(ranks, axis=-1))
avg_ranks = np.array(avg_ranks).T
mean_ranks = np.mean(avg_ranks, axis=-1)

# Get ACC averages
avg_accs, std_accs = [], []
avg_accs_name = {}
for seed, table_seed in table.items(): 
    for name, table_name in table_seed.items(): 
        if name not in avg_accs_name.keys():
            avg_accs_name[name] = [] # 0.0
        avg_accs_name[name].append(table_name['acc'])
for name, accs in avg_accs_name.items():
    avg_accs.append(np.mean(accs))
    std_accs.append(np.std(accs))

# Get BWT averages
avg_bwts, std_bwts = [], []
avg_bwts_name = {}
for seed, table_seed in table.items(): 
    for name, table_name in table_seed.items(): 
        if name not in avg_bwts_name.keys():
            avg_bwts_name[name] = [] # 0.0
        avg_bwts_name[name].append(table_name['bwt'])
for name, bwts in avg_bwts_name.items():
    avg_bwts.append(np.mean(bwts))
    std_bwts.append(np.std(bwts))

print('Method: Rank, ACC, BWT')
for name, mean, avg_acc, std_acc, avg_bwt, std_bwt in zip(names, mean_ranks, avg_accs, std_accs, 
                                                          avg_bwts, std_bwts):
    if 'heuristic1' in name.lower():
        name = 'Heur-GD'
    elif 'heuristic2' in name.lower():
        name = 'Heur-LD'
    elif 'heuristic3' in name.lower():
        name = 'Heur-AT'
    print('%s: %.2f, %2.1f $\pm$ %2.1f, %2.1f $\pm$ %2.1f  ' %(name, mean, 
                                          100*avg_acc, 100*std_acc,
                                          100*avg_bwt, 100*std_bwt))
print()   

