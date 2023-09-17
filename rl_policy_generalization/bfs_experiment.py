

import os, argparse, time
import sys
import shutil
import numpy as np
from omegaconf import OmegaConf
import tracemalloc

import torch

from training.vis_utils import plot_task_accuracy 
from training.utils import print_log_acc_bwt, set_random_seed, save_pickle, load_pickle
from envs.utils import create_dir, get_observation_dim
from envs.action_spaces import DiscreteActionSpace, CountingActionSpace
from trainer.rs import ReplaySchedulingTrainer
from dataloaders.cl_dataset import ContinualDataset

t_start = time.time()

# Arguments
parser = argparse.ArgumentParser(description='Coding...')
# Load the config file
parser.add_argument('--config',  type=str, default='./configs/config_experiment1.yml')
flags =  parser.parse_args()
args = OmegaConf.load(flags.config)

print()

########################################################################################################################    

def remove_dir(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

def store_transitions_from_path(transitions, test_res, val_res, actions):
    # load transitions from file and add results
    #transitions = load_pickle(os.path.join(args.table_dir, transitions_fname))
    n_tasks = test_res['acc'].shape[0]
    for t in range(n_tasks):
        if t==0:
            index = '0'
        elif t>0:
            index = '0-' + '-'.join([str(a) for a in actions[:t]])

        if index in transitions[t].keys():
            continue
        else:
            acc_test, loss_test = test_res['acc'][t, :], test_res['loss'][t, :]
            acc_val, loss_val = val_res['acc'][t, :], val_res['loss'][t, :]
            if t > 0:
                done = (t+1==n_tasks)
                acts = actions[:t]
                transition = {
                            'task': t,
                            'acc': acc_val, 
                            'loss': loss_val,  
                            'val': {'acc': acc_val, 'loss': loss_val},
                            'test': {'acc': acc_test, 'loss': loss_test}, 
                            'action': acts[-1],
                            'actions': acts.copy(),
                            'done': done,
                            }
            else:
                transition = {
                            'task': t,
                            'acc': acc_val, 
                            'loss': loss_val,  
                            'val': {'acc': acc_val, 'loss': loss_val},
                            'test': {'acc': acc_test, 'loss': loss_test}, 
                            }
            # Store transition
            transitions[t][index] = transition
    #save_pickle(transitions, os.path.join(args.table_dir, transitions_fname))
    return transitions

# Represents a node of an n-ary tree
class Node:
    def __init__(self, key):
        self.key = key
        self.child = []
   
 # Utility function to create a new tree node
def newNode(key):    
    temp = Node(key)
    return temp

def initialize_tree(n_tasks, action_space):

    for t in range(n_tasks):
        if t > 1:
            all_children = [] # 
            num_actions = action_space.dims[t-1]
            for i, temp in enumerate(curr):
                for act in range(num_actions):
                    idx = act + i*num_actions # this gives unique second index in model_path
                    path_index = temp.key['path_index'] + [idx]
                    actions = temp.key['actions'] + [act]
                    parent_path = temp.key['model_path']
                    transition_key = '0-' + '-'.join([str(i) for i in actions]) # based on the actions
                    prev_transition_key = '0-' + '-'.join([str(i) for i in actions[:-1]]) # based on the actions
                    #model_path =  'model_%d_%d.pt' %(t+1, idx)
                    model_path =  'model_{}.pth.tar'.format(transition_key)
                
                    temp.child.append(newNode({'task_id': t,
                                        'path_index': path_index,
                                        'action': act,
                                        'actions': actions, 
                                        'parent_path': parent_path,
                                        'model_path': model_path,
                                        'model_paths': temp.key['model_paths'] + [model_path],
                                        'transition_key': transition_key,
                                        'prev_transition_key': prev_transition_key,
                                    }))
                all_children.extend(temp.child) # keep track of all children from parents on upper node level
            curr = all_children
        elif t == 0:
            root = newNode({'task_id': t,
                            'path_index': [0],
                            'action': None, 
                            'actions': None,
                            'parent_path': None,
                            #'model_path': 'model_%d_%d.pt' %(t+1, 0), #['model_%d_%d.pth.tar' %(t+1, 1)],
                            #'model_paths': ['model_%d_%d.pt' %(t+1, 0)], 
                            'model_path': 'model_{}.pth.tar'.format(0), 
                            'model_paths': ['model_{}.pth.tar'.format(0)], 
                            'transition_key': '0',
                            'prev_transition_key': None,
                            })
            curr = root
        elif t == 1:
            path_index = curr.key['path_index'] + [0]
            action = 0
            #model_path = 'model_%d_%d.pt' %(t+1, 0)
            model_path = 'model_{}-{}.pth.tar'.format(0,0) 
            
            curr.child.append(newNode({'task_id': t,
                                'path_index': path_index,
                                'action': action,
                                'actions': [action],
                                'parent_path': curr.key['model_path'],
                                'model_path': model_path,
                                'model_paths': curr.key['model_paths'] + [model_path],
                                'transition_key': '0-0',
                                'prev_transition_key': '0',
                            }))
            curr = curr.child # list with nodes (only one node at this step!)
    return root

# Prints the n-ary tree level wise
def LevelOrderTraversal(root, verbose=False):
 
    if (root == None):
        return;
   
    # Standard level order traversal code
    # using queue
    q = []  # Create a queue
    q.append(root); # Enqueue root 

    order = []
    while (len(q) != 0):
     
        n = len(q);
  
        # If this node has children
        while (n > 0):
         
            # Dequeue an item from queue and print it
            p = q[0]
            q.pop(0);
            order.append(p.key)
            if verbose:
                print(p.key, end=' ')
   
            # Enqueue all children of the dequeued item
            for i in range(len(p.child)):
             
                q.append(p.child[i]);
            n -= 1
   
        print() # Print new line between two levels
    return order

def get_results_from_path(approach, dataset, model_paths):
    n_tasks = approach.n_tasks 

    acc, loss = np.zeros([n_tasks, n_tasks], dtype=np.float32), np.zeros([n_tasks, n_tasks], dtype=np.float32)
    val_acc, val_loss = np.zeros([n_tasks, n_tasks], dtype=np.float32), np.zeros([n_tasks, n_tasks], dtype=np.float32)
    for t in range(n_tasks):
        test_model = approach.load_model_from_file(file_name=model_paths[t])
        #print('Test model trained on task {:d}...'.format(t+1))
        for u in range(t+1):
            val_res = approach.test(u, dataset.valid_set[u], model=test_model)
            test_res = approach.test(u, dataset.test_set[u], model=test_model)
            print('>>> Test on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(u+1,
                                                                                test_res['loss_t'],
                                                                                100*test_res['acc_t']))
            acc[t, u], loss[t, u] = test_res['acc_t'], test_res['loss_t']
            val_acc[t, u], val_loss[t, u] = val_res['acc_t'], val_res['loss_t']
    res = {}
    res['test'] = {'acc': acc, 'loss': loss}
    res['val'] = {'acc': val_acc, 'loss': val_loss}
    return res

def get_env_name(dataset, seed):
    if dataset.dataset in ['PermutedMNIST']:
        task_shuffle = dataset.task_shuffle
        env_name = '{}Task-{}'.format(dataset.n_tasks, dataset.dataset)
        for t in task_shuffle:
            env_name += '-{}'.format(t)
    else:
        env_name = '{}-5Split'.format(dataset.dataset)
        for i in range(len(dataset.task_ids)):
            env_name += '-{}'.format(dataset.task_ids[i])
    env_name += '-Seed-{}'.format(seed)
    env_name += '-DatasetSeed-{}'.format(dataset.dataset_seed)
    return env_name

def run_bfs(args):

    # Shorthands
    checkpoint_dir = args.checkpoint_dir # config['training']['checkpoint_dir']
    log_dir = args.log_dir # config['training']['log_dir']
    n_tasks = args.n_tasks #config['data']['n_tasks']
    verbose = args.verbose # config['session']['verbose']

    # set global seed
    set_random_seed(args.seed)
    # Get dataset and tree
    obs_dim = get_observation_dim(args)
    if args.action_space == 'counting':
        action_space = CountingActionSpace(n_tasks=args.n_tasks)
    else:
        action_space = DiscreteActionSpace(n_tasks=args.n_tasks)
    print('Instantiate data generators, env and model...')
    dataset = ContinualDataset(args)
    print('task ids: ', dataset.task_ids)
    args.classes_per_task = dataset.classes_per_task

    # Get breadth-first search order
    root = initialize_tree(n_tasks, action_space)
    bfs_order = LevelOrderTraversal(root, verbose=0)

    # Get training approach
    approach = ReplaySchedulingTrainer(args)
    approach.reset()

    # Create directory for storing results 
    transition_table_dir = args.table_dir
    if not os.path.exists(transition_table_dir):
        os.makedirs(transition_table_dir, exist_ok=True)
    #create_dir(transition_table_dir)
    transitions = {t:dict() for t in range(n_tasks)}
    transitions_fname = get_env_name(dataset, args.seed) + '.pkl' #dataset.get_dataset_name() + '.pkl'

    save_pickle(transitions, os.path.join(transition_table_dir, transitions_fname))

    # For saving results
    res = {}
    rewards = []
    accs = []
    best_accs = []
    visited_terminal_node = []
    best_rewards = []
    t_elapsed = []
    replay_schedules = []
    best_reward = 0.0
    best_path_index = None
    best_task_config = None
    best_acc = None
    best_rs = None
    best_actions = None

    t0 = time.time()
    
    # train on first task
    t = 0
    print('Train on task {:d}'.format(t+1))
    approach.train_single_task_in_bfs(t, dataset.train_set[t])
    #storing_transition(transitions, cmd='reset', task_id=t, trainer=approach, dataset=dataset)
    
    model_path = bfs_order[0]['model_path']
    approach.save_checkpoint(task_id=t+1, folder=checkpoint_dir, file_name=model_path)
    print()

    for i, task_config in enumerate(bfs_order[1:]):
        task_id = task_config['task_id']
        action = task_config['action']
        actions = task_config['actions']
        model_paths = task_config['model_paths']

        # Load checkpoint
        approach.load_checkpoint(checkpoint_dir=checkpoint_dir, file_path=model_paths[task_id-1])
        current_task_proportion = action_space.get_action_by_index(task=task_id-1, action_index=action)
        print('current_task_proportion: ', current_task_proportion)
        # Get replay schedule from action indices and set schedule in trainer
        replay_schedule = []
        for tt, act_index in enumerate(actions):
            task_prop = action_space.get_action_by_index(task=tt, action_index=act_index)
            replay_schedule.append(task_prop)

        # Train classifier on task
        approach.train_single_task_in_bfs(task_id, dataset.train_set[task_id], current_task_proportion)

        # save checkpoints for evaluation 
        approach.save_checkpoint(task_id=task_id+1, folder=checkpoint_dir, file_name=model_paths[task_id])
        # Evaluation after learning last task
        if (task_id+1) == n_tasks:
            res = get_results_from_path(approach, dataset, model_paths)
            
            # Parse results for storing transitions
            #store_transitions_from_path(transitions_fname, res['test'], res['val'], actions)
            transitions = store_transitions_from_path(transitions, res['test'], res['val'], actions)
            
            # Check best reward and store results
            avg_acc = np.mean(res['val']['acc'][-1, :])
            reward = avg_acc # computed from validation set
            if reward > best_reward:
                best_reward = reward
                #best_task_config = task_config
                best_acc = res['test']['acc'] # test acc matrix
                best_rs = replay_schedule
                best_actions = actions

            # remove final checkpoint after evaluation
            fname = os.path.join(checkpoint_dir, model_paths[n_tasks-1])
            #print(fname)
            if os.path.exists(fname):
                os.remove(fname)

            if (i+1) % 10 == 0:
                save_pickle(transitions, os.path.join(transition_table_dir, transitions_fname))

    print('Breadth-first search finished after {:d} iterations.\n'.format(i+1))

    save_pickle(transitions, os.path.join(transition_table_dir, transitions_fname))

    # Save results
    res = {}
    res['best_reward'] = best_reward
    res['best_acc'] = best_acc
    res['best_rs'] = best_rs
    res['best_actions'] = best_actions
    #res['best_task_config'] = best_task_config
    #res['actions'] = best_task_config['actions']
    #res['rewards'] = rewards
    #res['rs'] = replay_schedules
    #res['visited_terminal_node'] = visited_terminal_node
    #res['best_rewards'] = best_rewards
    res['time_elapsed'] = time.time() - t0 #t_elapsed
    return res

def main(args):
    
    create_dir(args.log_dir)
    log_dir = args.log_dir
    create_dir(args.checkpoint_dir)
    checkpoint_dir = args.checkpoint_dir
    
    # Load config
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")
    args.device = str(device)

    res = run_bfs(args)
    best_acc = res['best_acc'] # test acc matrix from replay schedule with best reward

    # Save results
    save_pickle(res, os.path.join(log_dir, 'res_seed%s.p' %(args.seed)))
    np.savetxt(os.path.join(log_dir, 'best_rs.txt'), np.stack(res['best_rs'], axis=0), '%.3f') 
    print('\nResults:')
    print('Avg. val accuracy after training on final task: ', res['best_reward'])
    print('Best replay schedule: {}'.format(res['best_rs']))
    t_elapsed = res['time_elapsed'] # in seconds
    print('Total elapsed time: {:.2f} sec, or {:.2f} mins, or {:.2f} hours'.format(t_elapsed, 
            t_elapsed / 60.0, t_elapsed / (60.0**2)))
    print('*' * 100)
    print()

    # Remove checkpoint dir to save space
    if not args.session.keep_checkpoints:
        print('Removing checkpoint dir {} to save space'.format(checkpoint_dir))
        try:
            shutil.rmtree(checkpoint_dir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

    print('Done.')

if __name__ == '__main__':
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)
    main(args)
    print('All done!')