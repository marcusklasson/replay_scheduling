
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import cm
from matplotlib import ticker
import tikzplotlib as tpl

def plot_task_accuracy(accs, save_dir, fname='task_accuracy', make_tikz_plot=False):
    """ Plot the individual task accuarcies for all tasks.
        Args:
            accs (dict): Dictionary with all task accuracies' mean and std.
            save_dir (str): Directory where plots are saved.
            fname (str): The filename without .tex or .png extension, 
                which is added later.
    """
    n_tasks = accs.shape[0]
    fig, ax = plt.subplots()
    for i in range(n_tasks):
        x = np.arange(i+1, n_tasks+1)
        acc = accs[i:, i]
        ax.plot(x, acc, marker='o', linestyle='-', label='Task %d'%(i+1))

    gem_bwt = sum(accs[-1]-np.diag(accs))/ (len(accs[-1])-1)

    # Other stuff
    ax.set_xlabel('Task Number')
    ax.set_ylabel('Accuracy')
    ax.set_title('ACC: %.3f, BWT: %.3f' %(100*np.mean(accs[-1]), 100*gem_bwt))
    ax.set_xticks(range(1, n_tasks+1))
    ax.set_ylim(bottom=0.5, top=1.01)
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick))
    ax.legend(loc='lower left')
    ax.grid(alpha=0.5, linestyle='-')
    fig.tight_layout()
    if make_tikz_plot:
        if not os.path.exists(save_dir + '/tikz'):
            os.makedirs(save_dir + '/tikz', exist_ok=True)
        tpl.save(
            os.path.join(save_dir + '/tikz', fname+'.tex' ),  # this is name of the file where your code will lie
            axis_width="\\figwidth",  # we want LaTeX to take care of the width
            axis_height="\\figheight",  # we want LaTeX to take care of the height
            # if the figure contains an image in the background (this one doesn't), this is where LaTeX (!) should search for the png.
            tex_relative_path_to_data="./",
            # we want the plot to look *exactly* like here (e.g. axis limits, axis ticks, etc.)
            strict=True,
        )
    plt.savefig(os.path.join(save_dir, fname+'.png'), format='png')
    plt.close()

def plot_task_proportions(task_proportions, save_dir, fname='task_proportions', make_tikz_plot=False):
    # parse task proportions
    n_tasks = len(task_proportions)#len(task.proportions[0].keys())
    width = 0.15
    spacing = 0.02
    """
    props = []
    for p in task_proportions:
        print(p)
        props.append(list(p.values()))
    """
    props = np.array(task_proportions, dtype=np.float32)
    ind = np.arange(len(props))
    fig, ax = plt.subplots()
    ax.bar(ind - 3*width/2, props[:, 0], width-spacing, label='Task 1', edgecolor='black') 
    ax.bar(ind - width/2, props[:, 1], width-spacing, label='Task 2', edgecolor='black') 
    ax.bar(ind + width/2, props[:, 2], width-spacing, label='Task 3', edgecolor='black') 
    ax.bar(ind + 3*width/2, props[:, 3], width-spacing, label='Task 4', edgecolor='black') 

    ax.set_xticks(ind)
    ax.set_xticklabels(np.arange(2, n_tasks+2))
    ax.set_xlabel('Task Number')
    ax.set_ylabel('Proportion')
    ax.legend()
    fig.tight_layout()
    #plt.show()
    
    if make_tikz_plot:
        if not os.path.exists(save_dir + '/tikz'):
            os.makedirs(save_dir + '/tikz', exist_ok=True)
        tpl.save(
            os.path.join(save_dir + '/tikz', fname+'.tex' ),  # this is name of the file where your code will lie
            axis_width="\\figwidth",  # we want LaTeX to take care of the width
            axis_height="\\figheight",  # we want LaTeX to take care of the height
            # if the figure contains an image in the background (this one doesn't), this is where LaTeX (!) should search for the png.
            tex_relative_path_to_data="./",
            # we want the plot to look *exactly* like here (e.g. axis limits, axis ticks, etc.)
            strict=True,
        )
    
    plt.savefig(os.path.join(save_dir, fname+'.png'), format='png')
    plt.close()

def plot_replay_schedule_bubble_plot(replay_schedule, save_dir, fname, make_tikz_plot=False):
    """ Plot nubmer of examples per task in the memory for a given replay schedule 
        at every trained task.
        Args:
            replay_schedule (np.array): Proportions per task at every time step.
            save_dir (str): Directory where plots are saved.
            fname (str): The filename without .tex or .png extension, 
                which is added later.
    """
    if isinstance(replay_schedule, list):
        replay_schedule = np.array(replay_schedule)
    n_tasks = replay_schedule.shape[0]+1
    n_memory_tasks = n_tasks-1 
    task_curr = np.arange(2, n_tasks+1) # current tasks
    task_memory = np.arange(1, n_tasks) # tasks in memory

    #print(cm.get_cmap('tab10'))
    cmap_name = 'tab20' if (n_tasks > 10) else 'tab10'
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots()
    for i, t_curr in enumerate(task_curr):
        #print(task_memory)
        #print([t_curr]*n_memory_tasks)
        #ax.scatter([t_curr]*n_memory_tasks, task_memory, s=replay_schedule[i, :]*1000, c=colors[:n_memory_tasks], alpha=0.75)
        n = len(task_memory[i:])
        scatter = ax.scatter(task_curr[i:], [task_memory[i]]*n, s=replay_schedule[i:, i]*1000, c=[colors[i]]*n, alpha=0.75)

    ax.set_ylabel('Replayed Task')
    ax.set_xlabel('Current Task')
    #ax.set_title('Split {}, Memory size {}, Seed {}'.format(dataset, m, seed))
    ax.set_xlim(1.5, n_tasks+0.5)
    ax.set_ylim(0.5, n_tasks-0.5)
    #ax.invert_yaxis()
    ax.set_xticks(task_curr)
    ax.set_yticks(task_memory)
    ax.grid(alpha=0.5)
    #plt.show()
    fig.tight_layout()
    if make_tikz_plot:
        if not os.path.exists(save_dir + '/tikz'):
            os.makedirs(save_dir + '/tikz', exist_ok=True)
        tpl.save(
            os.path.join(save_dir + '/tikz', fname+'.tex' ),  # this is name of the file where your code will lie
            axis_width="\\figwidth",  # we want LaTeX to take care of the width
            axis_height="\\figheight",  # we want LaTeX to take care of the height
            # if the figure contains an image in the background (this one doesn't), this is where LaTeX (!) should search for the png.
            tex_relative_path_to_data="./",
            # we want the plot to look *exactly* like here (e.g. axis limits, axis ticks, etc.)
            strict=True,
        )
    plt.savefig(os.path.join(save_dir, fname+'.png'), format='png')
    plt.close()

def plot_task_proportions_points(task_proportions, save_dir, fname='task_proportions'):
    # parse task proportions
    n_tasks = len(task_proportions)
    task_proportions = np.array(task_proportions)
    fig, ax = plt.subplots()
    for i in range(task_proportions.shape[0]):
        x = np.arange(i, n_tasks)
        props = task_proportions[i:, i]
        ax.plot(x, props, 'o', label='T%i' %(i+1), markersize=8)

    ax.set_xticks(np.arange(n_tasks))
    ax.set_xticklabels(np.arange(2, n_tasks+2))
    ax.set_xlabel('Task Number')
    ax.set_ylabel('Proportion')
    ax.legend(ncol=2)
    ax.grid(axis='y')
    fig.tight_layout()

    plt.savefig(os.path.join(save_dir, fname+'.png'), format='png')
    plt.close()

def plot_return_over_episodes(dqn_res, upper_bound=None, baseline=None, baseline_random=None, n_episodes=None, save_dir='./', title='Return over Episodes'):
    fig, ax = plt.subplots()
    x = np.arange(1, n_episodes+1)
    #x = np.linspace(1, n_episodes, len(dqn_res['mean']))

    if baseline:
        ax.plot([1, n_episodes], [baseline['mean'], baseline['mean']], color='k', label='ETS')
        ax.fill_between(x, baseline['mean']+baseline['std'], baseline['mean']-baseline['std'], color='k', alpha=0.2)
    if baseline_random:
        ax.plot([1, n_episodes], [baseline_random['mean'], baseline_random['mean']], color='b', label='Random')
        ax.fill_between(x, baseline_random['mean']+baseline_random['std'], baseline_random['mean']-baseline_random['std'], color='b', alpha=0.2)
    if upper_bound:
        ax.plot([1, n_episodes], [upper_bound['mean'], upper_bound['mean']], label='Upper (BFS)')
        ax.fill_between(x, upper_bound['mean']+upper_bound['std'], upper_bound['mean']-upper_bound['std'], alpha=0.2) 
    # DQN
    for name, res in dqn_res.items():
        x = np.linspace(1, n_episodes, len(res['mean']))
        ax.plot(x, res['mean'], label=name)
        ax.fill_between(x, res['mean']+res['std'], res['mean']-res['std'], alpha=0.2)

    ax.legend(loc='lower right')
    #ax.set_ylim(0.9, 1.0)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir + '/' + title + '.png')
    plt.close()
    #plt.show()

def plot_grouped_bar_chart(dqn_res, baseline=None, baseline_random=None, save_dir='./', fname='gridsearch_bar_chart'):
    # parse task proportions
    num_xticks = len(dqn_res) #len(task.proportions[0].keys())
    width = 0.15
    spacing = 0.04

    num_bars = len(dqn_res)
    ind = np.arange(num_bars)
    offset = ind - np.max(ind)/2
    fig, ax = plt.subplots()
    i = 0
    for expfrac_label, res_buffer in dqn_res.items():
        ind = np.arange(len(res_buffer))
        means_buffers = [v['mean'] for k, v in res_buffer.items()]
        stds_buffers = [v['std'] for k, v in res_buffer.items()]

        ax.bar(ind + offset[i]*width, means_buffers, width-spacing, 
                yerr=stds_buffers,
                label='explore={}'.format(expfrac_label), 
                edgecolor='black') 
        xtick_labels = list(res_buffer.keys())
        i += 1

    if baseline:
        ax.plot([-1, num_bars], [baseline['mean'], baseline['mean']], 'k-', alpha=0.5, label='ETS')
        ax.plot([-1, num_bars], [baseline['mean']+baseline['std'], baseline['mean']+baseline['std']], 'k--', alpha=0.5)
        ax.plot([-1, num_bars], [baseline['mean']-baseline['std'], baseline['mean']-baseline['std']], 'k--', alpha=0.5)
    if baseline_random:
        ax.plot([-1, num_bars], [baseline_random['mean'], baseline_random['mean']], 'b-', alpha=0.5, label='Random')
        ax.plot([-1, num_bars], [baseline_random['mean']+baseline_random['std'], baseline_random['mean']+baseline_random['std']], 'b--', alpha=0.5)
        ax.plot([-1, num_bars], [baseline_random['mean']-baseline_random['std'], baseline_random['mean']-baseline_random['std']], 'b--', alpha=0.5)

    ax.set_xticks(np.arange(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel('Buffer Size')  #ax.set_xlabel('Num steps') 
    ax.set_ylim(0.92, 1.0)
    ax.set_ylabel('ACC')
    ax.legend(ncol=3, loc='lower center')
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(save_dir, fname+'.png'), format='png')
    plt.close()