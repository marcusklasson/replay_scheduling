
import argparse
import os
import shutil
import time
import pickle
import numpy as np
import torch

from training.config import load_config
from training.data import get_multitask_experiment
from training.utils import print_log_acc_bwt, save_pickle

# Arguments
parser = argparse.ArgumentParser(
    description='Training model for Continual Learning.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--seed', type=int, default=1, help='Seed for random number generator.')
parser.add_argument('--scenario', type=str, default='task', choices=['task', 'class'], help='Continual learning scenario.')
parser.add_argument('--memory_size', type=int, default=24, help='Memory size.')
parser.add_argument('--selection_method', type=str, default='random', choices=['random', 'mean_of_features', 'k_center_coreset'],
                    help='Method for selecting samples to store in memory.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--load_partition', action='store_true', 
                    help='Load memory partition found from tree search.')
parser.add_argument('--agem', action='store_true', help='Use A-GEM model.')

args = parser.parse_args()

def run(config):

    # Shorthands
    out_dir = config['training']['out_dir']
    checkpoint_dir = config['training']['checkpoint_dir']
    model_file = config['training']['model_file']
    log_dir = config['training']['log_dir']
    verbose = config['session']['verbose']

    scenario = config['training']['scenario']
    replay_mode = config['replay']['method']

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

    if verbose > 0:
        print('Number of training datasets: ', len(train_datasets))
        for t, dataset in enumerate(train_datasets):
            print(t, len(dataset))
        print('Number of validation datasets: ', len(valid_datasets))
        for t, dataset in enumerate(valid_datasets):
            print(t, len(dataset))
        print('Number of test datasets: ', len(test_datasets))
        for t, dataset in enumerate(test_datasets):
            print(t, len(dataset))
        print()

    # Get training approach
    method = config['training']['method']
    if method == 'agem':
        from trainer.agem import AGEM
        approach = AGEM(config)
    elif method == 'er':
        from trainer.er import ER
        approach = ER(config)
    elif method == 'rs':
        from trainer.rs import ReplaySchedulingTrainer
        approach = ReplaySchedulingTrainer(config)
    elif method == 'coreset':
        from trainer.coreset import Coreset
        approach = Coreset(config)
    elif method == 'finetune':
        from trainer.finetune import Finetune
        approach = Finetune(config)
    elif method == 'joint':
        from trainer.joint import JointTrainer
        approach = JointTrainer(config)
    else:
        raise ValueError('Method {} is not implemented'.format(args.method))

    # Loop over all tasks
    n_tasks = config['data']['n_tasks']
    acc, loss = np.zeros([n_tasks, n_tasks], dtype=np.float32), np.zeros([n_tasks, n_tasks], dtype=np.float32)
    val_acc, val_loss = np.zeros([n_tasks, n_tasks], dtype=np.float32), np.zeros([n_tasks, n_tasks], dtype=np.float32)
    t0 = time.time() # for measuring elapsed time

    for t in range(n_tasks):
        
        if (method == 'joint') and (t > 0):
            train_dataset = torch.utils.data.ConcatDataset(train_datasets[:(t+1)])
        else:
            train_dataset = train_datasets[t]
        print('Training on dataset from task %d...' %(t+1))
        print('Number of training examples: ', len(train_dataset))
        print()

        # Train on task t
        approach.train_single_task(t+1, train_dataset, valid_datasets)
        train_res = approach.evaluate_task(t+1, train_dataset)
        print(' Training loss: {:.3f} Acc = {:.3f}, '.format(train_res['loss_t'], train_res['acc_t']))
        print('-'*250)
        print()

        # Save checkpoint
        approach.save_checkpoint(task_id=t+1, folder=checkpoint_dir, file_name='model_%d.pth.tar' %(t+1))
        # Testing model on all seen tasks
        test_model = approach.load_model_from_file(file_name='model_%d.pth.tar' %(t+1)) # uses checkpoint_dir inside function
        for u in range(t+1):
            test_res = approach.test(u+1, test_datasets[u], model=test_model)
            print('>>> Test on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(u+1,
                                                                                 test_res['loss_t'],
                                                                                 100*test_res['acc_t']))
            acc[t, u], loss[t, u] = test_res['acc_t'], test_res['loss_t']
            if len(valid_datasets[u]) > 0:
                val_res = approach.test(u+1, valid_datasets[u], model=test_model)
                val_acc[t, u], val_loss[t, u] = val_res['acc_t'], val_res['loss_t']
            else:
                val_acc[t, u], val_loss[t, u] = 0.0, 0.0 
        # Save accuracies
        print()
        np.savetxt(os.path.join(log_dir, 'accs.txt'), acc, '%.6f')
        np.savetxt(os.path.join(log_dir, 'accs_val.txt'), val_acc, '%.6f')

    avg_acc, gem_bwt = print_log_acc_bwt(acc, loss, output_path=log_dir, file_name='logs.p')
    avg_acc_val, _ = print_log_acc_bwt(val_acc, val_loss, output_path=log_dir, file_name='logs_val.p')

    # Print elapsed time
    t_elapsed = time.time() - t0 # in seconds
    t_string = 'Total elapsed time: {:.2f} sec, or {:.2f} mins, or {:.2f} hours'.format(t_elapsed, t_elapsed / 60.0, t_elapsed / (60.0**2))
    print(t_string)
    print()
    np.savetxt(os.path.join(log_dir, 'time_elapsed.txt'), [t_string], '%s') # t_string has to be in an array
    # Save results
    res = {}
    res['reward'] = avg_acc_val
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
    if config['training']['method'] not in ['joint', 'finetune']:
        out_dir = os.path.join(out_dir, 'M%d' %(config['replay']['examples_per_class'])) # append examples per class 
    config['training']['out_dir'] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    accuracies, forgetting, rewards = [], [], []
    n_runs = config['session']['n_runs'] 
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
        res = run(config)
        accuracies.append(res['avg_acc'])
        forgetting.append(res['gem_bwt'])
        rewards.append(res['reward'])

        # Save results
        save_pickle(res, os.path.join(log_dir, 'res_seed%s.p' %(seed)))
        print()

        # Remove checkpoint dir to save space
        if not config['session']['keep_checkpoints']:
            print('Removing checkpoint dir {} to save space'.format(checkpoint_dir))
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))

    # Done with training 
    print('*' * 100)
    print ('Average over {} runs for Baseline: '.format(n_runs))
    print ('Avg ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean()*100, np.array(accuracies).std()*100))
    print ('Avg BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean()*100, np.array(forgetting).std()*100))
    print ('Avg reward (val acc.): {:5.2f}% \pm {:5.4f}'.format(np.array(rewards).mean()*100, np.array(rewards).std()*100))
    print('Done.')

if __name__ == '__main__':
    main(args)