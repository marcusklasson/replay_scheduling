
import argparse
import os
import shutil
import time
import pickle
import numpy as np
import torch

from training.config import load_config
from training.data import get_multitask_experiment
from trainer.rs import ReplaySchedulingTrainer
from mcts.action_space import DiscreteActionSpace, TaskLimitedActionSpace
from training.utils import print_log_acc_bwt, save_pickle

# Arguments
parser = argparse.ArgumentParser(
    description='Training model for Continual Learning.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
args = parser.parse_args()

########################################################################################################################    

def remove_dir(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

def run_experiment(config):

    # Shorthands
    out_dir = config['training']['out_dir']
    checkpoint_dir = config['training']['checkpoint_dir']
    log_dir = config['training']['log_dir']
    n_tasks = config['data']['n_tasks']
    scenario = config['training']['scenario']
    verbose = config['session']['verbose']

    # Get tree
    memory_limit = config['replay']['memory_limit']
    action_space_type = config['search']['action_space']
    if action_space_type == 'seen_tasks':
        action_space = DiscreteActionSpace(n_tasks) 
    elif action_space_type == 'task_limit':
        action_space = TaskLimitedActionSpace(n_tasks, 
                            task_sample_limit=config['search']['task_sample_limit']) 

    # Set random seed
    seed = config['session']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get datasets
    train_datasets, valid_datasets, test_datasets, classes_per_task = get_multitask_experiment(config)
    config['training']['classes_per_task'] = classes_per_task

    # Get training approach
    approach = ReplaySchedulingTrainer(config)
    t_apply_replay = config['replay']['single_task_replay'] # key arg for the experiment
    # generate replay schedule with with only single task
    replay_schedule = []
    for t in range(1, n_tasks):
        action = action_space.get_action_with_single_task(current_task=t, wanted_task=0)
        replay_schedule.append(action)
    approach.set_replay_schedule(replay_schedule)
    # create arrays for storing results
    acc, loss = np.zeros([n_tasks, n_tasks], dtype=np.float32), np.zeros([n_tasks, n_tasks], dtype=np.float32)
    val_acc, val_loss = np.zeros([n_tasks, n_tasks], dtype=np.float32), np.zeros([n_tasks, n_tasks], dtype=np.float32)

    t0 = time.time()
    for t in range(n_tasks):
        # Disable replay, key step in this experiment
        approach.replay_enabled = (t_apply_replay == (t+1))
        # Train classifier on task
        approach.train_single_task(t+1, train_datasets[t])
        # save checkpoints for evaluation 
        approach.save_checkpoint(task_id=t+1, folder=checkpoint_dir, file_name='model_task{}.pth.tar'.format(t+1))
        # Evaluation after learning task
        test_model = approach.load_model_from_file(file_name='model_task{}.pth.tar'.format(t+1))
        for u in range(t+1):
            val_res = approach.test(u+1, valid_datasets[u], model=test_model)
            test_res = approach.test(u+1, test_datasets[u], model=test_model)
            print('>>> Test on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(u+1,
                                                                                test_res['loss_t'],
                                                                                100*test_res['acc_t']))
            acc[t, u], loss[t, u] = test_res['acc_t'], test_res['loss_t']
            val_acc[t, u], val_loss[t, u] = val_res['acc_t'], val_res['loss_t']
    # Save results
    print()
    np.savetxt(os.path.join(log_dir, 'accs.txt'), acc, '%.6f')
    np.savetxt(os.path.join(log_dir, 'accs_val.txt'), val_acc, '%.6f')

    avg_acc, gem_bwt = print_log_acc_bwt(acc, loss, output_path=log_dir, file_name='logs.p')
    avg_acc_val, _ = print_log_acc_bwt(val_acc, val_loss, output_path=log_dir, file_name='logs_val.p')

    t_elapsed = time.time() - t0
    print('Elapsed time: {:.2f} sec, or {:.2f} mins'.format(t_elapsed, t_elapsed / 60.0))
    # Save results
    res = {}
    res['reward'] = avg_acc_val
    res['rs'] = replay_schedule
    res['acc'] = acc
    res['avg_acc'] = avg_acc
    res['gem_bwt'] = gem_bwt
    return res

def main(args):
    # Load config
    config = load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")
    config['session']['device'] = device

    # Create directory for results
    out_dir = config['training']['out_dir']
    out_dir = os.path.join(out_dir, 'M%d' %(config['replay']['memory_limit'])) # append replay memory size 
    #config['training']['out_dir'] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    res_replay_at_task = {}
    for replay_at_task in range(2, config['data']['n_tasks']+1):
        print('*' * 100)
        print('\nRun experiment with Task 1 replay at task %d: ' %(replay_at_task))

        config['replay']['single_task_replay'] = replay_at_task
        out_dir_t = os.path.join(out_dir, 'T%d' %(replay_at_task))
        config['training']['out_dir'] = out_dir_t
        res_replay_at_task[replay_at_task] = {}

        accuracies, forgetting, rewards = [], [], []
        n_runs = 5
        for seed in range(1, n_runs+1):
            print('*' * 100)
            print('\nRun with seed %d: ' %(seed))
            config['session']['seed'] = seed

            # Create log and checkpoint dirs
            checkpoint_dir = os.path.join(out_dir_t, 'checkpoints', 'seed%d' %(seed))
            config['training']['checkpoint_dir'] = checkpoint_dir
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            log_dir = os.path.join(out_dir_t, 'logs', 'seed%d' %(seed))
            config['training']['log_dir'] = log_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Run code
            res = run_experiment(config)
            accuracies.append(res['avg_acc'])
            forgetting.append(res['gem_bwt'])
            rewards.append(res['reward'])

            # Save results
            save_pickle(res, os.path.join(log_dir, 'res_seed%s.p' %(seed)))
            np.savetxt(os.path.join(log_dir, 'rs.txt'), np.stack(res['rs'], axis=0), '%.3f') 
            print()

            # Remove checkpoint dir to save space
            if not config['session']['keep_checkpoints']:
                print('Removing checkpoint dir {} to save space'.format(checkpoint_dir))
                try:
                    shutil.rmtree(checkpoint_dir)
                except OSError as e:
                    print ("Error: %s - %s." % (e.filename, e.strerror))
        # Done with training for this T
        res_replay_at_task[replay_at_task]['accuracies'] = accuracies
        res_replay_at_task[replay_at_task]['forgetting'] = forgetting

    # Done with training 
    print('*' * 100)
    for replay_at_task, metrics in res_replay_at_task.items():
        print ("Average over {} runs when replaying at Task {}: ".format(n_runs, replay_at_task))
        print ('Avg ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(metrics['accuracies']).mean()*100, np.array(metrics['accuracies']).std()*100))
        print ('Avg BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(metrics['forgetting']).mean()*100, np.array(metrics['forgetting']).std()*100))
        print()
    print('Done.')

if __name__ == '__main__':
    main(args)