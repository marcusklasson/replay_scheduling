
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
import tikzplotlib as tpl

from scipy import stats

from training.utils import load_pickle



def get_res_for_method(root_dir, memory_size, seeds):
    root_dir1 = os.path.join(root_dir)
    res = {}
    memory_dir = root_dir1 + '/M%d/logs' %(memory_size) 
    avg_accs_test = []
    gem_bwt_test, gem_fwt_test = [], []
    for seed in seeds:
        seed_dir = memory_dir + '/seed%d' %(seed)
        #print(seed_dir)
        log = load_pickle(seed_dir + '/logs_with_fwt.p') #'/%s_res_seed%d.p' %(method, seed))
        #accs = np.loadtxt(seed_dir + '/accs.txt')
        #avg_accs_test.append(np.mean(accs[-1, :]))
        avg_accs_test.append(np.mean(log['rij']))
        gem_bwt_test.append(log['gem_bwt'])
        gem_fwt_test.append(log['gem_fwt'])
    #print()
    res['acc'] = {'mean': np.mean(avg_accs_test), 'std': np.std(avg_accs_test)}
    res['gem_bwt'] = {'mean': np.mean(gem_bwt_test), 'std': np.std(gem_bwt_test)}
    res['gem_fwt'] = {'mean': np.mean(gem_fwt_test), 'std': np.std(gem_fwt_test)}
    res['test_accs'] = avg_accs_test
    return res

def plot_acc_over_memory(mcts_res, ets_res, random_res, heuristic_gd_res, heuristic_ld_res=None, heuristic_at_res=None, 
        ytick=0.01, save_dir='./', fname='acc_over_memory'):
    """ Plot the individual task accuarcies for all tasks.
        Args:
            accs (dict): Dictionary with all task accuracies' mean and std.
            save_dir (str): Directory where plots are saved.
            fname (str): The filename without .tex or .png extension, 
                which is added later.
    """
    # Gather accs
    memory_sizes = list(mcts_res.keys())
    print(memory_sizes)
    x = np.arange(1,len(memory_sizes)+1)
    mcts_mean = np.array([mcts_res[m]['test']['mean'] for m in memory_sizes])
    mcts_std = np.array([mcts_res[m]['test']['std'] for m in memory_sizes])
    ets_mean = np.array([ets_res[m]['test']['mean'] for m in memory_sizes])
    ets_std = np.array([ets_res[m]['test']['std'] for m in memory_sizes])
    fig, ax = plt.subplots()
    # Plot random baseline
    random_mean = np.array([random_res[m]['test']['mean'] for m in memory_sizes])
    random_std = np.array([random_res[m]['test']['std'] for m in memory_sizes])
    ax.plot(x, random_mean, '-o', label='Random')
    ax.fill_between(x, random_mean+random_std, random_mean-random_std, alpha=0.2,)

    # Plot ETS baseline
    ax.plot(x, ets_mean, '-o', label='ETS')
    ax.fill_between(x, ets_mean+ets_std, ets_mean-ets_std, alpha=0.2,)
    # Plot heuristic baseline
    heuristic_mean = np.array([heuristic_gd_res[m]['test']['mean'] for m in memory_sizes])
    heuristic_std = np.array([heuristic_gd_res[m]['test']['std'] for m in memory_sizes])
    ax.plot(x, heuristic_mean, '-o', label='Heuristic GD')
    ax.fill_between(x, heuristic_mean+heuristic_std, heuristic_mean-heuristic_std, alpha=0.2,)
    if heuristic_ld_res is not None:
        heuristic_mean = np.array([heuristic_ld_res[m]['test']['mean'] for m in memory_sizes])
        heuristic_std = np.array([heuristic_ld_res[m]['test']['std'] for m in memory_sizes])
        ax.plot(x, heuristic_mean, '-o', label='Heuristic LD')
        ax.fill_between(x, heuristic_mean+heuristic_std, heuristic_mean-heuristic_std, alpha=0.2,)
    if heuristic_at_res is not None:
        heuristic_mean = np.array([heuristic_at_res[m]['test']['mean'] for m in memory_sizes])
        heuristic_std = np.array([heuristic_at_res[m]['test']['std'] for m in memory_sizes])
        ax.plot(x, heuristic_mean, '-o', label='Heuristic AT')
        ax.fill_between(x, heuristic_mean+heuristic_std, heuristic_mean-heuristic_std, alpha=0.2,)

    # Plot MCTS rewards
    ax.plot(x, mcts_mean, '-o', label='RS-MCTS')
    ax.fill_between(x, mcts_mean+mcts_std, mcts_mean-mcts_std, alpha=0.2,)

    # Other stuff
    ax.set_xlabel('Memory size $M$')
    ax.set_ylabel('ACC')
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in memory_sizes])
    #ax.set_ylim(top=1.01)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick))
    ax.legend(loc='lower right')
    #plt.grid(alpha=0.5, linestyle='-')
    fig.tight_layout()
    tpl.save(
        osp.join(save_dir, fname+'.tex' ),  # this is name of the file where your code will lie
        axis_width="\\figwidth",  # we want LaTeX to take care of the width
        axis_height="\\figheight",  # we want LaTeX to take care of the height
        # if the figure contains an image in the background (this one doesn't), this is where LaTeX (!) should search for the png.
        tex_relative_path_to_data="./",
        # we want the plot to look *exactly* like here (e.g. axis limits, axis ticks, etc.)
        strict=True,
    )
    plt.savefig(os.path.join(save_dir, fname+'.png'), format='png')
    plt.close()

# Root directory
dataset = 'CIFAR100' # ['MNIST', 'FashionMNIST', 'notMNIST', 'PermutedMNIST', 'CIFAR100', 'miniImagenet']
seeds = [1, 2, 3, 4, 5]
memory_selection = 'uniform'
scenario = 'task'

val_thresholds = {'MNIST': 0.96, 'FashionMNIST': 0.97, 'notMNIST': 0.98, 
                'PermutedMNIST': 0.75, 'CIFAR100': 0.5, 'miniImagenet': 0.75,}

if dataset in ['MNIST', 'FashionMNIST', 'notMNIST']:
    classes_per_task = 2
    memory_sizes = [10] #[8, 24, 80, 120, 200, 400, 800] 
    if scenario == 'class':
        memory_sizes = [10, 20, 40, 100, 200]
        val_thresholds['MNIST'] = 0.75
        val_thresholds['FashionMNIST'] = 0.5
        val_thresholds['notMNIST'] = 0.5
elif dataset in ['PermutedMNIST']:
    classes_per_task = 10
    memory_sizes = [100] #[90, 270, 450, 900, 2250] 
elif dataset in ['CIFAR100', 'miniImagenet']:
    classes_per_task = 5
    memory_sizes = [100] #[ 95, 285, 475, 950, 1900]
    if scenario == 'class':
        memory_sizes = [100, 200, 400, 800, 1600]
        val_thresholds['CIFAR100'] = 0.01
        val_thresholds['miniImagenet'] = 0.01
"""
if dataset in ['MNIST', 'FashionMNIST', 'notMNIST']:
    classes_per_task = 2
    memory_sizes = [8, 24, 80, 120, 200, 400, 800] 
    if scenario == 'class':
        memory_sizes = [10, 20, 40, 100, 200]
        val_thresholds['MNIST'] = 0.75
        val_thresholds['FashionMNIST'] = 0.5
        val_thresholds['notMNIST'] = 0.5
elif dataset in ['PermutedMNIST']:
    classes_per_task = 10
    memory_sizes = [90, 270, 450, 900, 2250] 
elif dataset in ['CIFAR100', 'miniImagenet']:
    classes_per_task = 5
    memory_sizes = [ 95, 285, 475, 950, 1900]
    if scenario == 'class':
        memory_sizes = [100, 200, 400, 800, 1600]
        val_thresholds['CIFAR100'] = 0.01
        val_thresholds['miniImagenet'] = 0.01
"""

# directories
if scenario == 'class':
    root_dir = './experiments/iclr23/class_il_scenario_iclr' 
    save_dir = './experiments/iclr23/figs_acc_over_memory_with_random_class' 
else:
    root_dir = '../../mcts_results_with_fwt' #/memory_selection_iclr' 
    #root_dir = './experiments/tmlr_submission/mcts_results_with_fwt/memory_selection_iclr' 
    save_dir = './experiments/tmlr_submission/memory_selection_results_with_fwt' 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ets_res, random_res, mcts_res = {}, {}, {}
heuristic_gd_res, heuristic_ld_res, heuristic_at_res = {}, {}, {}
random_dir = root_dir + '/{}_random/{}'.format(dataset, memory_selection)
ets_dir = root_dir + '/{}_ets/{}'.format(dataset, memory_selection)
heuristic_gd_dir = root_dir + '/{}_heuristic/{}/val_threshold_{}'.format(dataset, memory_selection, val_thresholds[dataset])
if scenario == 'class':
    heuristic_gd_dir = root_dir + '/{}_heuristic/val_threshold_{}/{}'.format(dataset, val_thresholds[dataset], memory_selection)

mcts_dir = root_dir + '/{}_mcts/{}'.format(dataset, memory_selection)
for m in memory_sizes:
    random_res[m] = get_res_for_method(random_dir, memory_size=m, seeds=seeds)
    ets_res[m] = get_res_for_method(ets_dir, memory_size=m, seeds=seeds)
    heuristic_gd_res[m] = get_res_for_method(heuristic_gd_dir, memory_size=m, seeds=seeds)
    mcts_res[m] = get_res_for_method(mcts_dir, memory_size=m, seeds=seeds)

## Plot ACCs
#plot_acc_over_memory(mcts_res, ets_res, random_res, heuristic_gd_res, 
#                        ytick=0.02, save_dir=save_dir, fname='%s_acc_over_memory' %(dataset))

# Print table with metrics
lines = []
print('\nPrint tables with ACC, BWT and FWT metrics')
lines.append('-'*100)
lines.append('\nMem size      ,     ACC(%)    ,   GEM BWT(%)   ,   GEM FWT(%)   ')
for m in memory_sizes:
    lines.append('\nRandom M={:d}     , {:.2f} $\pm$ {:.2f} , {:.2f} ±$\pm$ {:.2f} , {:.2f} $\pm$ {:.2f} '.format(m, 
        random_res[m]['acc']['mean']*100, random_res[m]['acc']['std']*100, 
        random_res[m]['gem_bwt']['mean']*100, random_res[m]['gem_bwt']['std']*100,
        random_res[m]['gem_fwt']['mean']*100, random_res[m]['gem_fwt']['std']*100))
    lines.append('\nETS M={:d}, {:.2f} $\pm$ {:.2f} , {:.2f} $\pm$ {:.2f} , {:.2f} $\pm$ {:.2f} '.format(m, 
        ets_res[m]['acc']['mean']*100, ets_res[m]['acc']['std']*100, 
        ets_res[m]['gem_bwt']['mean']*100, ets_res[m]['gem_bwt']['std']*100,
        ets_res[m]['gem_fwt']['mean']*100, ets_res[m]['gem_fwt']['std']*100))
    lines.append('\nHeuristic GD tau={:.2f} M={:d}, {:.2f} $\pm$ {:.2f} , {:.2f} $\pm$ {:.2f} , {:.2f} $\pm$ {:.2f} '.format(
        val_thresholds[dataset], m, 
        heuristic_gd_res[m]['acc']['mean']*100, heuristic_gd_res[m]['acc']['std']*100, 
        heuristic_gd_res[m]['gem_bwt']['mean']*100, heuristic_gd_res[m]['gem_bwt']['std']*100,
        heuristic_gd_res[m]['gem_fwt']['mean']*100, heuristic_gd_res[m]['gem_fwt']['std']*100))
    lines.append('\nMCTS M={:d} , {:.2f} $\pm$ {:.2f} , {:.2f} $\pm$ {:.2f} , {:.2f} $\pm$ {:.2f} '.format(m, 
        mcts_res[m]['acc']['mean']*100, mcts_res[m]['acc']['std']*100, 
        mcts_res[m]['gem_bwt']['mean']*100, mcts_res[m]['gem_bwt']['std']*100,
        mcts_res[m]['gem_fwt']['mean']*100, mcts_res[m]['gem_fwt']['std']*100))
    #lines.append('\n')
    lines.append('-'*100)
print('-'*50)

f = open(osp.join(save_dir, '%s_metrics_table.txt' %(dataset)), 'w')
f.writelines(lines)
f.close()

for l in lines:
    print(l)
print()

"""
memory_size = 8
print('-'*50)

print('Statistical significance: Paired t-tests')
paired_ttest = stats.ttest_rel(mcts_res['test_accs'], random_res['test_accs'])
print('MCTS vs Random, t=%.2f, pval=%.3f' %(paired_ttest.statistic, paired_ttest.pvalue))

paired_ttest = stats.ttest_rel(mcts_res['test_accs'], ets_res['test_accs'])
print('MCTS vs ETS, t=%.2f, pval=%.3f' %(paired_ttest.statistic, paired_ttest.pvalue))

paired_ttest = stats.ttest_rel(mcts_res['test_accs'], heur_gd_res['test_accs'])
print('MCTS vs Heur-GD, t=%.2f, pval=%.3f' %(paired_ttest.statistic, paired_ttest.pvalue))

paired_ttest = stats.ttest_rel(mcts_res['test_accs'], heur_ld_res['test_accs'])
print('MCTS vs Heuristic LD: ', paired_ttest)

paired_ttest = stats.ttest_rel(mcts_res['test_accs'], heur_at_res['test_accs'])
print('MCTS vs Heuristic AT: ', paired_ttest)
"""
print()


"""
print('CL performance metrics')
for m in memory_sizes:
    print('M={:d}, Random, {:.2f} +/- {:.2f}, {:.2f} +/- {:.2f}'.format(m, 
        random_res[m]['test']['mean']*100, random_res[m]['test']['std']*100, 
        random_res[m]['gem_bwt']['mean']*100, random_res[m]['gem_bwt']['std']*100))

    print('M={:d}, ETS, {:.2f} +/- {:.2f}, {:.2f} +/- {:.2f}'.format(m, 
        ets_res[m]['test']['mean']*100, ets_res[m]['test']['std']*100, 
        ets_res[m]['gem_bwt']['mean']*100, ets_res[m]['gem_bwt']['std']*100))

    print('M={:d}, Heur-GD, {:.2f} +/- {:.2f}, {:.2f} +/- {:.2f}'.format(m, 
        heuristic_gd_res[m]['test']['mean']*100, heuristic_gd_res[m]['test']['std']*100, 
        heuristic_gd_res[m]['gem_bwt']['mean']*100, heuristic_gd_res[m]['gem_bwt']['std']*100))

    print('M={:d}, MCTS, {:.2f} +/- {:.2f}, {:.2f} +/- {:.2f}'.format(m, 
        mcts_res[m]['test']['mean']*100, mcts_res[m]['test']['std']*100, 
        mcts_res[m]['gem_bwt']['mean']*100, mcts_res[m]['gem_bwt']['std']*100))
print()

print('Statistical significance: Welch t-tests')
for m in memory_sizes:
    welch_ttest = stats.ttest_ind(mcts_res[m]['test_accs'], random_res[m]['test_accs'], equal_var=False)
    print('M=%d, MCTS vs Random, t=%.2f, pval=%.3f' %(m, welch_ttest.statistic, welch_ttest.pvalue))

    welch_ttest = stats.ttest_ind(mcts_res[m]['test_accs'], ets_res[m]['test_accs'], equal_var=False)
    print('M=%d, MCTS vs ETS, t=%.2f, pval=%.3f' %(m, welch_ttest.statistic, welch_ttest.pvalue))

    welch_ttest = stats.ttest_ind(mcts_res[m]['test_accs'], heuristic_gd_res[m]['test_accs'], equal_var=False)
    print('M=%d, MCTS vs Heur-GD, t=%.2f, pval=%.3f' %(m, welch_ttest.statistic, welch_ttest.pvalue))
print()
"""


"""
welch_ttest = stats.ttest_ind(mcts_res['test_accs'], heur_ld_res['test_accs'], equal_var=False)
print('MCTS vs Heuristic LD: ', welch_ttest)

welch_ttest = stats.ttest_ind(mcts_res['test_accs'], heur_at_res['test_accs'], equal_var=False)
print('MCTS vs Heuristic AT: ', welch_ttest)
"""
print()


"""
from utils import compute_ranks

memory_size = 8
results = {'Random': random_res[memory_size], 'ETS': ets_res[memory_size], 'Heuristic GD': heuristic_gd_res[memory_size],
            'Heuristic LD': heuristic_ld_res[memory_size], 'Heuristic AT': heuristic_at_res[memory_size], 'MCTS': mcts_res[memory_size]}
print(results)
ranks = compute_ranks(results, split='test')
print('Print ranks')
for model, rank in ranks.items():
    print('model: {}, rank: {}'.format(model, rank))
"""