
import numpy as np
import torch

from envs.env_fixed_seed import EnvFixedSeed
from envs.env_table import EnvTable  

"""
def make_envs(args, obs_dim, action_space, datasets, seed_start):
    envs = []
    created_datasets = set()
    for seed, dataset_name in enumerate(datasets, start=seed_start):
        # should dataset labels be shuffled
        shuffle_labels = False 
        if (dataset_name in created_datasets) and args.shuffle_labels:
            shuffle_labels = True 

        # Create env
        if args.use_table_env:
            env = EnvTable(obs_dim, action_space, dataset_name, seed, args, shuffle_labels)
        else:
            env = EnvFixedSeed(obs_dim, action_space, dataset_name, seed, args, shuffle_labels)
        envs.append(env)
    return envs
"""

class MultiEnvNew(object):
    
    def __init__(self, envs):
        self.envs = envs # list with envs

    def reset(self):
        obs = []
        infos = []
        for env in self.envs:
            ob, info = env.reset()
            obs.append(ob)
            infos.append(info)
        return obs, infos

    def step(self, t, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        #print('multienv step, actions: ', actions)
        for env, ac in zip(self.envs, actions):
            ob, rew, done, info = env.step(t, ac)
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

            #if done:
            #    env.reset()
        return obs, rewards, dones, infos

    def __len__(self):
        return len(self.envs)

class MultiEnv(object):
    
    def __init__(self, obs_dim, action_space, datasets, seed_start, args,):
                    #use_pytorch_wrapper=False):
        self.obs_dim = obs_dim 
        self.action_space = action_space
        self.datasets = datasets
        self.args = args 
        self.seed_start = seed_start
        #self.use_pytorch_wrapper = use_pytorch_wrapper
        self.setup_envs()

    def setup_envs(self):
        self.envs = []
        created_datasets = set()
        for seed, dataset_name in enumerate(self.datasets, start=self.seed_start):
            # should dataset labels be shuffled
            if seed == 1:
                shuffle_labels = False
            else:
                shuffle_labels = self.args.shuffle_labels 
            #if (dataset_name in created_datasets) and self.args.shuffle_labels:
            #    shuffle_labels = True 

            # Create env
            if self.args.use_table_env:
                env = EnvTable(self.obs_dim, self.action_space, dataset_name, seed, self.args, shuffle_labels)
            else:
                env = EnvFixedSeed(self.obs_dim, self.action_space, dataset_name, seed, self.args, shuffle_labels)
            
            #if self.use_pytorch_wrapper:
            #    env = EnvWrapperPyTorch(env, env.device) 
            self.envs.append(env)

    def reset(self):
        obs = []
        infos = []
        for env in self.envs:
            ob, info = env.reset()
            obs.append(ob)
            infos.append(info)
        return obs, infos

    def step(self, t, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        for env, ac in zip(self.envs, actions):
            ob, rew, done, info = env.step(t, ac)
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

            #if done:
            #    env.reset()
        return obs, rewards, dones, infos

    def __len__(self):
        return len(self.envs)

    def _create_envs(self, args):
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
                
class EnvWrapperPyTorch(object):
    def __init__(self, env, device):
        #"""Return only every `skip`-th frame"""
        #super(EnvWrapperPyTorch, self).__init__()
        self.env = env
        self.device = device
        self.n_tasks = self.env.n_tasks
        self.task_ids = self.env.task_ids
        self.seed = self.env.seed

    def reset(self):
        obs, info = self.env.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step(self, t, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape)>1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        if not isinstance(actions, int):
            actions = actions[0]
        obs, reward, done, info = self.env.step(t, actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(np.array([reward])).unsqueeze(dim=1).float()
        return obs, reward, done, info


class MultiEnvWrapperPyTorch(object):
    def __init__(self, multi_env, device):
        self.multi_env = multi_env
        self.device = device
        self.envs = multi_env.envs
        #self.action_space = multi_env.action_space

    def reset(self):
        obs, info = self.multi_env.reset()
        #print('obs in reset: ', obs)
        # return pytorch tensor with every env at dim 1 
        obs = [torch.from_numpy(ob).float().to(self.device) for ob in obs]
        #print('obs in reset2 : ', obs)
        obs = torch.stack(obs, dim=0)
        #obs = obs.unsqueeze(0)
        #print(obs)
        #obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step(self, t, actions):
        t = self._tensor_to_list(t)
        t = t[0]
        actions = self._tensor_to_list(actions)
        obs, reward, done, info = self.multi_env.step(t, actions)
        obs = [torch.from_numpy(ob).float().to(self.device) for ob in obs]
        obs = torch.stack(obs, dim=0)
        reward = self._reward_list_to_tensor(reward)
        #print('done in step: ', done)
        #reward = [torch.tensor([r]).float() for r in reward]
        #reward = torch.stack(reward, dim=0)
        return obs, reward, done, info

    def __len__(self):
        return len(self.multi_env)

    def _tensor_to_list(self, x):
        x = torch.flatten(x)
        return x.tolist()

    def _reward_list_to_tensor(self, rew):
        rew = torch.tensor(rew).float()
        return rew.unsqueeze(dim=-1)

