
import os
import shutil
import numpy as np 

def get_observation_dim(args):
    # Get input dimension
    n_tasks = args.n_tasks
    obs_dim = n_tasks-1
    if args.state_add_delta:
        obs_dim += n_tasks-1 #obs_dim = 2*(n_tasks-1)

    if args.state_add_delta_max:
        obs_dim += n_tasks-1

    if args.state_add_forgetting:
        obs_dim += n_tasks-2#n_tasks-1
    
    if args.state_add_time:
        obs_dim += 1 # add the time indicator

    if args.state_add_bwt:
        obs_dim += 1 # add the bwt scalar
    return obs_dim


def get_filename_for_table(args, dataset_name, task_ids, seed):
    fname = '{}-5Split'.format(dataset_name)
    for i in range(len(task_ids)):
        fname += '-{}'.format(task_ids[i])
    fname += '-Seed-{}'.format(seed)
    if args.data.shuffle_labels:
        fname += '-DatasetSeed-{}'.format(args.data.seed)
    fname += '.pkl'
    return fname

def create_dir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname, exist_ok=True)

"""
def make_env(obs_dim, actions_space, dataset, seed, args):
    if args.use_table_env:
        env = EnvTable(obs_dim, action_space, dataset, seed, args)
    else:
        env = EnvFixedSeed(obs_dim, action_space, dataset, seed, args)
    return env
"""     
"""

def _envs(self, args):
    envs = []
    for idx, dataset_name in enumerate(args.train_envs):
        seed = args.train_seeds[idx] #args.seed + idx #+1
        #env = create_fixed_seed_env(args, dataset_name, seed, 
        #            use_universal_table=args.load_universal_table)
        env = create_env_with_universal_table(args, dataset_name, seed, shuffle_labels=False)
        envs.append(env)
    return envs

def _create_envs_with_shuffled_labels(self, args):
    envs = []
    created_datasets = set()
    for idx, dataset_name in enumerate(args.train_envs):
        seed = args.train_seeds[idx] #args.seed + idx 
        if dataset_name in created_datasets:
            env = create_fixed_seed_env(args, dataset_name, seed, 
                    use_universal_table=args.load_universal_table, shuffle_labels=True)
        else:
            env = create_fixed_seed_env(args, dataset_name, seed, 
                    use_universal_table=args.load_universal_table, shuffle_labels=False)
        envs.append(env)
        created_datasets.add(dataset_name)
    return envs
"""