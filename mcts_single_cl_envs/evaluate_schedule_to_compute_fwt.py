
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
from trainer.heuristic_scheduling import HeuristicSchedulingTrainer
from mcts.action_space import DiscreteActionSpace, TaskLimitedActionSpace
from training.utils import print_log_acc_bwt, save_pickle, load_pickle

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

def select_trainer(config):
    # Get training approach
    trainer_fn = ReplaySchedulingTrainer
    if 'heuristic' in config['replay']['schedule']:
        trainer_fn = HeuristicSchedulingTrainer

    method = config['training']['method']
    if method == 'rs':
        if 'extension' in config['training'].keys():
            extension = config['training']['extension']
            if extension == 'der':
                print('Using Dark-ER trainer!')
                from trainer.rs_der import ReplaySchedulingTrainerDER
                trainer_fn = ReplaySchedulingTrainerDER
            elif extension == 'derpp':
                print('Using DER++ trainer!')
                from trainer.rs_der import ReplaySchedulingTrainerDERPP
                trainer_fn = ReplaySchedulingTrainerDERPP
            elif extension == 'mer':
                print('Use Meta-ER trainer!')
                from trainer.rs_mer import ReplaySchedulingTrainerMER
                trainer_fn = ReplaySchedulingTrainerMER
            elif extension == 'hal':
                print('Use Hindsight Anchor Learning trainer!')
                from trainer.rs_hal import ReplaySchedulingTrainerHAL
                trainer_fn = ReplaySchedulingTrainerHAL
            elif extension == 'coreset':
                print('Using Coreset trainer!')
                if 'heuristic' in config['replay']['schedule']:
                    from trainer.heuristic_scheduling import HeuristicSchedulingTrainerCoreset
                    trainer_fn = HeuristicSchedulingTrainerCoreset
                else:
                    from trainer.rs_coreset_buffer import ReplaySchedulingTrainerCoreset
                    trainer_fn = ReplaySchedulingTrainerCoreset
            else:
                print('Using standard trainer!')
    elif method == 'agem':
        from trainer.agem import AGEM
        trainer_fn = AGEM
    elif method == 'er':
        from trainer.er import ER
        trainer_fn = ER
    elif method == 'coreset':
        from trainer.coreset import Coreset
        trainer_fn = Coreset
    elif method == 'finetune':
        from trainer.finetune import Finetune
        trainer_fn = Finetune
    elif method == 'joint':
        from trainer.joint import JointTrainer
        trainer_fn = JointTrainer
    else:
        raise ValueError('Method {} is not implemented'.format(args.method))
    approach = trainer_fn(config)
    return approach

def load_replay_schedule(config):
    replay_schedule = []
    path = config['training']['log_dir']
    if config['replay']['schedule'] in ['random', 'ets', 'heuristic_global_drop']:
        # Baseline schedule
        res = load_pickle(path + '/res_seed%d.p' %(config['session']['seed']))
        replay_schedule = res['rs']
        #replay_schedule = np.loadtxt(path + '/rs.txt')
    else:
        # MCTS or BFS
        res = load_pickle(path + '/mcts_res_seed%d.p' %(config['session']['seed']))
        replay_schedule = res['best_rs']
        #replay_schedule = np.loadtxt(path + '/best_rs.txt')
    #replay_schedule = replay_schedule.tolist()
    return replay_schedule

def run_ets(config):

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
    torch.cuda.manual_seed_all(seed) # added
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ##torch.backends.cudnn.enabled = False # added

    # Get datasets
    train_datasets, valid_datasets, test_datasets, classes_per_task = get_multitask_experiment(config)
    config['training']['classes_per_task'] = classes_per_task

    # Get training approach
    approach = select_trainer(config)

    # load replay schedule using log from previous experiment
    replay_schedule = []
    if config['training']['method'] not in ['finetune', 'joint']:
        replay_schedule = load_replay_schedule(config)
    approach.set_replay_schedule(replay_schedule)
    # create arrays for storing results
    acc, loss = np.zeros([n_tasks, n_tasks], dtype=np.float32), np.zeros([n_tasks, n_tasks], dtype=np.float32)
    val_acc, val_loss = np.zeros([n_tasks, n_tasks], dtype=np.float32), np.zeros([n_tasks, n_tasks], dtype=np.float32)

    t0 = time.time()
    
    for t in range(n_tasks):
        
        if t == 0: # 
            approach.save_checkpoint(task_id=t+1, folder=checkpoint_dir, file_name='model_task{}.pth.tar'.format(t))
            test_model = approach.load_model_from_file(file_name='model_task{}.pth.tar'.format(t))
            baseline, _, baseline_val, _ = evaluate_model_on_all_tasks(approach, test_model, test_datasets, valid_datasets, n_tasks)
            """
            """
            # resetting seed for safety
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        # Train classifier on task
        approach.train_single_task(t+1, train_datasets[t])
        # save checkpoints for evaluation 
        approach.save_checkpoint(task_id=t+1, folder=checkpoint_dir, file_name='model_task{}.pth.tar'.format(t+1))
        # Evaluation after learning task
        test_model = approach.load_model_from_file(file_name='model_task{}.pth.tar'.format(t+1))
        acc[t, :], loss[t, :], val_acc[t, :], val_loss[t, :] = evaluate_model_on_all_tasks(approach, test_model, test_datasets, valid_datasets, n_tasks)
    # Save results
    print()

    acc = np.concatenate((baseline[np.newaxis, :], acc))
    val_acc = np.concatenate((baseline_val[np.newaxis, :], val_acc))

    np.savetxt(os.path.join(log_dir, 'accs_with_baseline.txt'), acc, '%.6f')
    np.savetxt(os.path.join(log_dir, 'accs_val_with_baseline.txt'), val_acc, '%.6f')

    avg_acc, gem_bwt, gem_fwt = print_log_acc_bwt(acc, loss, output_path=log_dir, file_name='logs_with_fwt.p')
    avg_acc_val, _, _ = print_log_acc_bwt(val_acc, val_loss, output_path=log_dir, file_name='logs_val_with_fwt.p')

    t_elapsed = time.time() - t0
    print('Elapsed time: {:.2f} sec, or {:.2f} mins'.format(t_elapsed, t_elapsed / 60.0))
    # Save results
    res = {}
    res['reward'] = avg_acc_val
    res['rs'] = replay_schedule
    res['acc'] = acc
    res['avg_acc'] = avg_acc
    res['gem_bwt'] = gem_bwt
    res['gem_fwt'] = gem_fwt
    return res

def evaluate_model_on_all_tasks(approach, test_model, test_datasets, valid_datasets, n_tasks):
    # evaluate model across every seen and future task
    test_acc, test_loss = np.zeros(n_tasks, dtype=np.float32), np.zeros(n_tasks, dtype=np.float32)
    val_acc, val_loss = np.zeros(n_tasks, dtype=np.float32), np.zeros(n_tasks, dtype=np.float32)
    for u in range(n_tasks):
        test_res = approach.test(u+1, test_datasets[u], model=test_model)
        print('>>> Test on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(u+1,
                                                                            test_res['loss_t'],
                                                                            100*test_res['acc_t']))
        test_acc[u], test_loss[u] = test_res['acc_t'], test_res['loss_t']
        if len(valid_datasets[u]) > 0:
            val_res = approach.test(u+1, valid_datasets[u], model=test_model)
            val_acc[u], val_loss[u] = val_res['acc_t'], val_res['loss_t']
        else:
            val_acc[u], val_loss[u] = 0.0, 0.0 
    return test_acc, test_loss, val_acc, val_loss


def get_dirname(config):
    out_dir = config['training']['out_dir']
    #out_dir = os.path.join(out_dir, config['replay']['schedule']) # append scheduling method
    schedule = 'heuristic' if 'heuristic' in config['replay']['schedule'] else config['replay']['schedule'] 
    out_dir = os.path.join(out_dir, '%s_%s' %(config['data']['name'], schedule)) # append dataset name and scheduling method
    out_dir = os.path.join(out_dir, '%s' %(config['replay']['sample_selection'])) # append memory selection method
    if schedule == 'heuristic':
        out_dir = os.path.join(out_dir, 'val_threshold_%s' %(str(config['replay']['val_threshold']))) # append threshold value
    out_dir = os.path.join(out_dir, 'M%d' %(config['replay']['memory_limit'])) # append replay memory size 
    return out_dir 

def main(args):
    # Load config
    config = load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")
    config['session']['device'] = device

    # Create directory for results
    out_dir = get_dirname(config)

    config['training']['out_dir'] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    accuracies, forgetting, forward_transfer, rewards = [], [], [], []
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
        res = run_ets(config)
        accuracies.append(res['avg_acc'])
        forgetting.append(res['gem_bwt'])
        forward_transfer.append(res['gem_fwt'])
        rewards.append(res['reward'])

        # Save results
        save_pickle(res, os.path.join(log_dir, 'res_seed%s_testing.p' %(seed)))
        #np.savetxt(os.path.join(log_dir, 'rs.txt'), np.stack(res['rs'], axis=0), '%.3f') 
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
    print ("Average over {} runs for {:s} schedule with M={}: ".format(n_runs, 
                                                                        config['replay']['schedule'], 
                                                                        config['replay']['memory_limit']))
    print ('Avg ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean()*100, np.array(accuracies).std()*100))
    print ('Avg BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean()*100, np.array(forgetting).std()*100))
    print ('Avg FWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forward_transfer).mean()*100, np.array(forward_transfer).std()*100))
    print ('Avg reward (val acc.): {:5.2f}% \pm {:5.4f}'.format(np.array(rewards).mean()*100, np.array(rewards).std()*100))
    print('Done.')

if __name__ == '__main__':
    main(args)
