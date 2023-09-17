
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
from training.utils import print_log_acc_bwt, save_pickle, compute_gem_bwt

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
    # initializing the search tree with nodes
    for t in range(n_tasks):
        if t > 1:
            all_children = [] # 
            num_actions = action_space[t]
            for i, temp in enumerate(curr):
                for act in range(num_actions):
                    idx = act + i*num_actions # this gives unique second index in model_path
                    path_index = temp.key['path_index'] + [idx]
                    actions = temp.key['actions'] + [act]
                    parent_path = temp.key['model_path']
                    transition_key = '0-' + '-'.join([str(i) for i in actions]) # based on the actions
                    prev_transition_key = '0-' + '-'.join([str(i) for i in actions[:-1]]) # based on the actions
                    #model_path =  'model_%d_%d.pt' %(t+1, idx)
                    model_path =  'model_{}.pt'.format(transition_key)
                
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
                            'model_path': 'model_{}.pt'.format(0), 
                            'model_paths': ['model_{}.pt'.format(0)], 
                            'transition_key': '0',
                            'prev_transition_key': None,
                            })
            curr = root
        elif t == 1:
            path_index = curr.key['path_index'] + [0]
            action = 0
            #model_path = 'model_%d_%d.pt' %(t+1, 0)
            model_path = 'model_{}-{}.pt'.format(0,0) 
            
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
    # creating the traversal path in breadth-first order
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

def run_bfs(config):

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
    num_actions = [action_space.get_dim_at_task(t) for t in range(n_tasks)]
    # Get breadth-first search order
    root = initialize_tree(n_tasks, num_actions)
    bfs_order = LevelOrderTraversal(root, verbose=verbose)

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
    trainer_fn = ReplaySchedulingTrainer
    if 'extension' in config['training'].keys():
        if config['training']['extension'] in ['coreset']:
            from trainer.rs_coreset_buffer import ReplaySchedulingTrainerCoreset
            trainer_fn = ReplaySchedulingTrainerCoreset
    approach = trainer_fn(config)
    for t in range(n_tasks):
        if config['training']['extension'] in ['coreset']:
            approach.update_coreset(train_datasets[t], t)
        else:
            approach.update_episodic_memory(train_datasets[t])
    approach.use_episodic_memory = False # prevents trainer to update memory after training
    
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
    approach.train_single_task(t+1, train_datasets[t])
    model_path = bfs_order[0]['model_path']
    approach.save_checkpoint(task_id=t+1, folder=checkpoint_dir, file_name=model_path)
    print()

    for i, task_config in enumerate(bfs_order[1:]):
        t = task_config['task_id']
        action = task_config['action']
        actions = task_config['actions']
        model_paths = task_config['model_paths']
        #print(model_paths)

        # Load checkpoint
        approach.load_checkpoint(checkpoint_dir=checkpoint_dir, file_path=model_paths[t-1])
        # Get replay schedule from action indices and set schedule in trainer
        replay_schedule = []
        for t, action_index in enumerate(actions, start=1):
            action = action_space.get_action_by_index(task=t, action_index=action_index)
            replay_schedule.append(action)
        
        approach.set_replay_schedule(replay_schedule)
        approach.n_replays = len(replay_schedule)-1 # set the indexing variable for getting proportion for current task
        # Train classifier on task
        approach.train_single_task(t+1, train_datasets[t])
        # save checkpoints for evaluation 
        approach.save_checkpoint(task_id=t+1, folder=checkpoint_dir, file_name=model_paths[t])
        # Evaluation after learning last task
        if (t+1) == n_tasks:
            acc = np.zeros([n_tasks, n_tasks], dtype=np.float32)
            loss = np.zeros([n_tasks, n_tasks], dtype=np.float32)
            val_acc, val_loss = np.zeros([n_tasks, n_tasks], dtype=np.float32), np.zeros([n_tasks, n_tasks], dtype=np.float32)
            for t in range(n_tasks):
                test_model = approach.load_model_from_file(file_name=model_paths[t])
                #print('Test model trained on task {:d}...'.format(t+1))
                for u in range(t+1):
                    val_res = approach.test(u+1, valid_datasets[u], model=test_model)
                    test_res = approach.test(u+1, test_datasets[u], model=test_model)
                    print('>>> Test on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(u+1,
                                                                                        test_res['loss_t'],
                                                                                        100*test_res['acc_t']))
                    acc[t, u], loss[t, u] = test_res['acc_t'], test_res['loss_t']
                    val_acc[t, u], val_loss[t, u] = val_res['acc_t'], val_res['loss_t']

            # Save
            folder = os.path.join(log_dir, 'results', '_'.join([str(a) for a in actions]))
            print(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)

            print()
            fname = 'accs.txt'
            print('Saved accuracies at '+ os.path.join(folder, fname))
            np.savetxt(os.path.join(folder, fname), acc, '%.6f')
            np.savetxt(os.path.join(folder, 'accs_val.txt'), val_acc, '%.6f')

            _, _ = print_log_acc_bwt(acc, loss, output_path=folder, file_name='logs.p')
            avg_acc, gem_bwt = print_log_acc_bwt(val_acc, val_loss, output_path=folder, file_name='logs_val.p')

            # Check best reward and store results
            reward = avg_acc # computed from validation set
            if reward > best_reward:
                best_reward = reward
                best_task_config = task_config
                best_acc = acc # test acc matrix
                best_rs = replay_schedule
                best_actions = actions
            # Save results from iteration
            rewards.append(reward)
            best_rewards.append(best_reward)
            accs.append(acc)
            best_accs.append(best_acc)
            visited_terminal_node.append(task_config['path_index'][-1])
            t_elapsed.append(time.time() - t0) # in seconds
            replay_schedules.append(replay_schedule)

            # remove final checkpoint after evaluation
            fname = os.path.join(checkpoint_dir, model_paths[n_tasks-1])
            #print(fname)
            if os.path.exists(fname):
                os.remove(fname)

    print('Breadth-first search finished after {:d} iterations.\n'.format(i+1))

    # Save results
    res = {}
    res['best_reward'] = best_reward
    res['best_task_config'] = best_task_config
    res['actions'] = best_task_config['actions']
    res['rewards'] = rewards
    res['rs'] = replay_schedules
    res['visited_terminal_node'] = visited_terminal_node
    res['best_rewards'] = best_rewards
    res['time_elapsed'] = t_elapsed
    res['best_acc'] = best_acc
    res['best_rs'] = best_rs
    res['best_actions'] = best_actions
    return res

def main(args):
    # Load config
    config = load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")
    config['session']['device'] = device

    # Create directory for results
    out_dir = config['training']['out_dir']
    out_dir = os.path.join(out_dir, '%s' %(config['replay']['sample_selection'])) # append memory selection method
    out_dir = os.path.join(out_dir, 'M%d' %(config['replay']['memory_limit'])) # append replay memory size 
    config['training']['out_dir'] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    accuracies, forgetting, rewards = [], [], []
    n_runs = 5
    for seed in range(1, n_runs+1):
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
        res = run_bfs(config)
        rewards.append(res['best_reward'])
        best_acc = res['best_acc'] # test acc matrix from replay schedule with best reward
        accuracies.append(np.mean(best_acc[-1, :]))
        forgetting.append(compute_gem_bwt(best_acc))

        # Save results
        save_pickle(res, os.path.join(log_dir, 'res_seed%s.p' %(seed)))
        np.savetxt(os.path.join(log_dir, 'best_rs.txt'), np.stack(res['best_rs'], axis=0), '%.3f') 
        print('\nResults:')
        print('Avg. val accuracy after training on final task: ', res['best_reward'])
        print('Best replay schedule: {}'.format(res['best_rs']))
        t_elapsed = res['time_elapsed'][-1] # in seconds
        print('Total elapsed time: {:.2f} sec, or {:.2f} mins, or {:.2f} hours'.format(t_elapsed, 
                t_elapsed / 60.0, t_elapsed / (60.0**2)))
        print('*' * 100)
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
    print ("Average over {} runs for BFS with M={}: ".format(n_runs, config['replay']['memory_limit']))
    print ('Avg reward (val acc.): {:5.2f}% \pm {:5.4f}'.format(np.array(rewards).mean()*100, np.array(rewards).std()*100))
    print ('Avg ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean()*100, np.array(accuracies).std()*100))
    print ('Avg BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean()*100, np.array(forgetting).std()*100))
    print('Done.')

if __name__ == '__main__':
    main(args)
