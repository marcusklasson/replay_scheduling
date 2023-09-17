
import os
import shutil
import argparse
import numpy as np
import torch

from mcts.search import ReplaySchedulingMCTS
from mcts.nodes import ReplaySchedulingNode, LongTaskHorizonNode
from mcts.state import State
from mcts.action_space import DiscreteActionSpace, TaskLimitedActionSpace

from training.data import get_multitask_experiment
from training.config import load_config
from training.utils import print_log_acc_bwt, save_pickle, compute_gem_bwt

# Arguments
parser = argparse.ArgumentParser(
    description='Training model for Continual Learning.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')

def run_mcts(config):
    # Shorthands
    out_dir = config['training']['out_dir']
    checkpoint_dir = config['training']['checkpoint_dir']
    log_dir = config['training']['log_dir']
    scenario = config['training']['scenario']
    n_tasks = config['data']['n_tasks']

    # Set random seed
    seed = config['session']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parameters for MCTS
    memory_limit = config['replay']['memory_limit']
    
    action_space_type = config['search']['action_space']
    if action_space_type == 'seen_tasks':
        action_space = DiscreteActionSpace(n_tasks) 
    elif action_space_type == 'task_limit':
        # used for limiting task proportions to specific number of tasks
        action_space = TaskLimitedActionSpace(n_tasks, 
                            task_sample_limit=config['search']['task_sample_limit']) 
    else:
        raise ValueError('Action space type {} does nto exist.'.format(action_space_type))

    # Create root node for MCTS
    initial_state = State(n_tasks=n_tasks, task=1, action_space=action_space)
    if n_tasks < 7:
        root = ReplaySchedulingNode(state=initial_state)
    else:
        root = LongTaskHorizonNode(state=initial_state) # inherits from ReplaySchedulingNode

    # Get datasets
    train_datasets, valid_datasets, test_datasets, classes_per_task = get_multitask_experiment(config)
    config['training']['classes_per_task'] = classes_per_task

    # Run MCTS search 
    mcts_iters = config['search']['iters']
    print('mcts_iters: ', mcts_iters)
    c_param = config['search']['c_param']
    mcts = ReplaySchedulingMCTS(config, 
                                node=root, 
                                datasets={'train': train_datasets, 
                                        'valid': valid_datasets,
                                        'test': test_datasets,})
    res = mcts.run_search(mcts_iters, c_param=c_param)
    return res

def main(args):
    # Load config
    config = load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")
    config['session']['device'] = device

    # Create directory for results
    out_dir = config['training']['out_dir']
    if config['training']['extension'] in ['hal', 'mer', 'der', 'derpp']:
        out_dir = os.path.join(out_dir, 'rs_mcts') # append scheduling method 
    out_dir = os.path.join(out_dir, '%s' %(config['replay']['sample_selection'])) # append memory selection method
    out_dir = os.path.join(out_dir, 'M%d' %(config['replay']['memory_limit'])) # append replay memory size 
    config['training']['out_dir'] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    accuracies, forgetting, rewards = [], [], []
    n_runs = config['session']['n_runs'] #2 #5 # hard coded
    seed_start = config['session']['seed']
    for seed in range(seed_start, seed_start+n_runs):
        print('*' * 100)
        print('\nRun with seed %d: ' %(seed))
        config['session']['seed'] = seed

        # Create log and checkpoint dirs
        checkpoint_dir = os.path.join(out_dir, 'checkpoints', 'seed%d' %(seed))
        config['training']['checkpoint_dir'] = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        log_dir = os.path.join(out_dir, 'logs', 'seed%d' %(seed))
        config['training']['log_dir'] = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Run code
        res = run_mcts(config)
        rewards.append(res['best_reward'])
        best_acc = res['best_acc'] # test acc matrix from replay schedule with best reward
        accuracies.append(np.mean(best_acc[-1, :]))
        forgetting.append(compute_gem_bwt(best_acc))

        # Save results
        save_pickle(res, os.path.join(log_dir, 'mcts_res_seed%s.p' %(seed)))
        np.savetxt(os.path.join(log_dir, 'accs.txt'), res['best_acc'], '%.6f') 
        np.savetxt(os.path.join(log_dir, 'best_rs.txt'), np.stack(res['best_rs'], axis=0), '%.3f') 
        print()

        # Print results
        print('Avg. val. accuracy after training on final task: ', res['best_reward'])
        print('Best replay schedule: ')
        print(np.stack(res['best_rs'], axis=0))

        t_elapsed = res['time_elapsed'][-1] # in seconds
        print('Total elapsed time: {:.2f} sec, or {:.2f} mins, or {:.2f} hours'.format(t_elapsed, 
                t_elapsed / 60.0, t_elapsed / (60.0**2)))

        # Remove checkpoint dir to save space
        if not config['session']['keep_checkpoints']:
            print('Removing checkpoint dir {} to save space'.format(checkpoint_dir))
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))

    # Done with training 
    print('*' * 100)
    print ("Average over {} runs for RS-MCTS with M={}: ".format(n_runs, config['replay']['memory_limit']))
    print ('Avg ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean()*100, np.array(accuracies).std()*100))
    print ('Avg BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean()*100, np.array(forgetting).std()*100))
    print ('Avg reward (val acc.): {:5.2f}% \pm {:5.4f}'.format(np.array(rewards).mean()*100, np.array(rewards).std()*100))
    print('Done.')


if __name__ == '__main__':
    main(args)
