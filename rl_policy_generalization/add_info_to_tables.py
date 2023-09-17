
import os 
import shutil
import ast
import numpy as np 
import torch 
from training.utils import load_pickle, save_pickle

def get_shuffled_labels(n_classes, classes_per_task, seed):
    #labels = torch.randperm(n_classes, generator=torch.Generator().manual_seed(seed)).tolist()
    rs = np.random.RandomState(seed)
    labels = list(rs.permutation(n_classes))
    task_ids = [labels[i:i + classes_per_task] for i in range(0, len(labels), classes_per_task)]
    return labels, task_ids

dataset_name = 'CIFAR10'
scenario = 'task'
n_tasks = 5
n_classes = 10
classes_per_task = 2
n_seeds = 100 #20
# Get label shuffles with their seed
seed_to_labels, seed_to_task_ids = {}, {}
# first add original task ids
labels = list(range(n_classes))
task_ids = [labels[i:i + classes_per_task] for i in range(0, len(labels), classes_per_task)]
seed_to_labels[0] = labels
seed_to_task_ids[0] = task_ids

for seed in range(1, n_seeds+1):
    labels, task_ids = get_shuffled_labels(n_classes, classes_per_task, seed=seed)
    seed_to_labels[seed] = labels
    seed_to_task_ids[seed] = task_ids
labels_to_seed = {str(v): k for k, v in seed_to_labels.items()}
task_ids_to_seed = {str(v): k for k, v in seed_to_task_ids.items()}

# Loop over tables
#root_dir = './experiments/jan17/%s_bfs_testing/transition_tables/original' %(dataset_name) #'./transition_tables_add_info/transition_tables_testing'
#dest_dir = './experiments/jan17/%s_bfs_testing/transition_tables/processed' %(dataset_name)
root_dir = './experiments/mar17/%s_bfs_M10_same_seed_in_cl_and_data_parts/transition_tables' %(dataset_name)#'./transition_tables_add_info/transition_tables_testing'
dest_dir = './experiments/mar17/%s_bfs_M10_same_seed_in_cl_and_data_parts/transition_tables/processed' %(dataset_name)
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

if scenario == 'task':
    for filename in os.listdir(root_dir):
        
        if filename.endswith('.pkl'): 
            print(os.path.join(root_dir, filename))
            tab = {}
            data = load_pickle(os.path.join(root_dir, filename))

            tab['data'] = data 

            parts = filename.split('.pkl')[0].split('-')
            tab['dataset_name'] = parts[0]
            tab['env_seed'] = int(parts[-1])

            task_ids = [ast.literal_eval(p) for p in parts[2:2+n_tasks]]
            dataset_seed = task_ids_to_seed[str(task_ids)]
            if dataset_seed > 0:
                tab['dataset_seed'] = dataset_seed
                tab['shuffled_labels'] = True 
            else:
                tab['dataset_seed'] = dataset_seed
                tab['shuffled_labels'] = False
            tab['task_ids'] = task_ids

            """
            new_filename = filename.split('.')[0]
            if dataset_seed >= 0:
                new_filename += '-DatasetSeed-%d.pkl' %(dataset_seed)
            else:
                new_filename += '.pkl'
            """
            new_filename = parts[0]
            new_filename += '_Seed_%d' %(dataset_seed)
            new_filename += '_%dSplit' %(n_tasks)
            tmp = [''.join([str(x) for x in tt]) for tt in task_ids]
            tmp = '_'.join(tmp)
            new_filename += '_Tasks_%s.pkl' %(tmp)

            print('new_filename: ', new_filename)

            print(task_ids)
            print(dataset_seed)
            #for k, v in data.items():
            #    print(k, len(v))
            print()
            # Save new table in destination dir
            save_pickle(tab, os.path.join(dest_dir, new_filename))
        else:
            continue

elif scenario == 'domain':
    for filename in os.listdir(root_dir):
        
        if filename.endswith('.pkl'): 
            print(os.path.join(root_dir, filename))
            tab = {}
            data = load_pickle(os.path.join(root_dir, filename))

            tab['data'] = data 

            parts = filename.split('.pkl')[0].split('-')
            print(parts)
            
            tab['dataset_name'] = parts[0]
            dataset_seed = int(parts[-1])
            tab['dataset_seed'] = dataset_seed
            tab['env_seed'] = int(parts[-3])
            if dataset_seed > 0:
                tab['shuffled_tasks'] = True 
            else:
                tab['shuffled_tasks'] = False
            n_tasks = int(parts[0].split('Task')[0])
            task_ids = [int(t) for t in parts[2:2+n_tasks]]
            tab['task_ids'] = task_ids

            print(task_ids)
            print(dataset_seed)
            #for k, v in data.items():
            #    print(k, len(v))
            print()
            # Save new table in destination dir
            save_pickle(tab, os.path.join(dest_dir, filename))

        else:
            continue
