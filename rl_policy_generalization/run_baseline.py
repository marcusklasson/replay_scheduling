
import os, argparse, time
import shutil
import numpy as np
import random
from omegaconf import OmegaConf

from training.logger import Logger
from envs.action_spaces import DiscreteActionSpace, CountingActionSpace
from trainer.rs import ReplaySchedulingTrainer 
from trainer.heuristic_scheduling import HeuristicSchedulingTrainer 
from training.utils import print_log_acc_bwt, set_random_seed, save_pickle, load_pickle
from training.vis_utils import plot_task_accuracy, plot_task_proportions
from dataloaders.cl_dataset import ContinualDataset

t_start = time.time()

# Arguments
parser = argparse.ArgumentParser(description='Coding...')
# Load the config file
parser.add_argument('--config',  type=str, default='./configs/baseline_mnist.yaml')
parser.add_argument('--baseline_policy', type=str)
parser.add_argument('--random_policy_seed',  type=int, default=1)
parser.add_argument('--n_runs_random_policy',  type=int, default=3)
parser.add_argument('--tau',  type=float, default=0.999)
flags =  parser.parse_args()
args = OmegaConf.load(flags.config)
args.baseline_policy = flags.baseline_policy
args.random_policy_seed = flags.random_policy_seed
args.n_runs_random_policy = flags.n_runs_random_policy
args.replay.tau = flags.tau

print()

########################################################################################################################    

def train(args, trainer, dataset, action_space, log_dir):
    # Run 
    n_tasks = args.n_tasks
    acc = np.zeros([n_tasks, n_tasks])
    loss = np.zeros([n_tasks, n_tasks])

    val_acc = np.zeros([n_tasks, n_tasks])
    val_loss = np.zeros([n_tasks, n_tasks])
    rs = []
    for t in range(n_tasks):

        print('*'*250)
        task_dataset = dataset.get_dataset_for_task(t)
        print(' '*105, 'Dataset {:2d} ({:s})'.format(t+1, task_dataset['name']))
        print('*'*250)

        # get task proportions for replay
        if t > 0:
            if args.baseline_policy=='ets':
                task_proportions = action_space.get_action_with_equal_proportions(t-1)
            elif args.baseline_policy=='random':
                action_index, task_proportions = action_space.sample(t-1)
            elif 'heuristic' in args.baseline_policy: #args.baseline_policy in ['heuristic1', 'heuristic2', 'heuristic3']:
                task_proportions = trainer.get_next_task_proportions(t-1, val_acc)
            else: # finetune
                task_proportions = None 
            print('task props: ', task_proportions)
            rs.append(task_proportions)
        else:
            task_proportions = None

        # Train model
        trainer.train_single_task(t, task_dataset['train'], task_proportions) # train model on current task

        # Evaluate model
        for u in range(t+1):
            #dataloader = dataset.get_dataloader(u)
            task_dataset = dataset.get_dataset_for_task(u)
            test_res = trainer.eval_task(u, task_dataset['test']) # get task accuracy
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, task_dataset['name'],
                                                                                          test_res['loss_t'],
                                                                                          test_res['acc_t']*100))
            acc[t, u] = test_res['acc_t']
            loss[t, u] = test_res['loss_t']
            if len(task_dataset['valid']) > 0:
                val_res = trainer.eval_task(u, task_dataset['valid']) # get task accuracy
                val_acc[t, u] = val_res['acc_t']
                val_loss[t, u] = val_res['loss_t']
            else:
                val_acc[t, u], val_loss[t, u] = 0.0, 0.0
    
    print('Replay schedule: ')
    for t, r in enumerate(rs):
        print(t, r)
    print()

    # Compute metrics
    print('\nSave validation metrics...')
    avg_acc, gem_bwt = print_log_acc_bwt(val_acc, val_loss, output_path=log_dir, file_name='logs_val.p')
    print('\nSave test metrics...')
    avg_acc, gem_bwt = print_log_acc_bwt(acc, loss, output_path=log_dir)
    plot_task_accuracy(accs=acc, save_dir=log_dir)
    print('Elapsed time: {:.2f} sec'.format(time.time() - t_start))

    if args.n_tasks == 5 and args.baseline_policy in ['ets', 'random', 'heuristic1', 'heuristic2', 'heuristic3']:
        plot_task_proportions(task_proportions=rs, save_dir=log_dir)
    save_pickle(rs, path=log_dir + '/rs.p')

    if not args.session.keep_checkpoints:
        print('Removing checkpoint dir {} to save space'.format(args.checkpoint_dir))
        try:
            shutil.rmtree(args.checkpoint_dir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
    return 

def main(args):
    
    # Add baseline names to log dirs
    if args.baseline_policy in ['ets', 'random']:
        log_dir_ = args.log_dir + '/%s' %(args.baseline_policy)
        checkpoint_dir_ = args.checkpoint_dir + '/%s' %(args.baseline_policy)
        args.data.pc_valid = 0.0 # ETS and Random uses no validation dataset
    elif args.baseline_policy in ['heuristic1', 'heuristic2', 'heuristic3',]:
        log_dir_ = args.log_dir + '/%s/threshold_%s' %(args.baseline_policy, str(args.replay.tau))
        checkpoint_dir_ = args.checkpoint_dir + '/%s/threshold_%s' %(args.baseline_policy, str(args.replay.tau))
        assert args.data.pc_valid > 0.0

    n_envs = args.n_envs 
    seed_start = args.seed
    for seed in range(seed_start, seed_start+n_envs):
        print('*' * 100)
        print('\nRun with seed %d: ' %(seed))
        # set seeds for dataset shuffle and cl network initialization
        args.data.seed = seed 
        args.cl.seed = seed 

        # Create log dir named with seed
        log_dir = log_dir_ + '/seed%d' %(seed)
        checkpoint_dir__ = checkpoint_dir_ + '/seed%d' %(seed)

        n_runs_policy = args.n_runs_random_policy if args.baseline_policy == 'random' else 1
        
        for run_idx in range(n_runs_policy):
            if args.baseline_policy == 'random':
                random_policy_seed = args.random_policy_seed + run_idx # seed starts at 1
                log_dir = log_dir + '/policy_seed%d' %(random_policy_seed)
                checkpoint_dir = checkpoint_dir__ + '/policy_seed%d' %(random_policy_seed)
            else:
                random_policy_seed = None
                checkpoint_dir = checkpoint_dir__ 
                
            #log_dir = log_dir + '/logs'
            #args.checkpoint_dir = checkpoint_dir + '/checkpoints'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir, exist_ok=True)

            # Set random seed
            set_random_seed(seed)
            # Data loader
            args.data.shuffle_labels = True if seed > 0 else False
            print('Instantiate data generators and model...')
            dataset = ContinualDataset(args)
            args.dataset_name = dataset.dataset # 
            print('task_ids: ', dataset.task_ids)
            args.n_tasks = dataset.n_tasks
            args.classes_per_task = dataset.classes_per_task
            args.input_size = dataset.input_size

            action_space = DiscreteActionSpace(args.n_tasks, seed=random_policy_seed)

            # Create trainer
            if args.baseline_policy in ['ets', 'random']:
                trainer = ReplaySchedulingTrainer(args)
            elif 'heuristic' in args.baseline_policy: #args.baseline_policy in ['heuristic1', 'heuristic2', 'heuristic3']:
                trainer = HeuristicSchedulingTrainer(args)
            else:
                raise ValueError('Baseline %s does not exist...' %(args.baseline_policy))

            # train baseline
            train(args, trainer, dataset, action_space, log_dir)
    

if __name__ == '__main__':
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)
    main(args)
    print('All done!')