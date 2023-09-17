

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trainer.rs import ReplaySchedulingTrainer
from training.logger import Logger
from training.utils import load_pickle, save_pickle
from envs.utils import create_dir
#from dataloaders.core import CLDataset, ShuffledLabelsCLDataset
#from dataloaders.cl_dataset import ContinualDataset

class EnvBaseSimple(object):
    """ Training environment for CL classifier which RL algorithm can make calls to
        for receiving observations, rewards, etc. Similar to OpenAI gym environment.
    """
    def __init__(self, observation_dim, action_space, dataset, seed, args, shuffle_labels=False):
        """ seed is used for giving specific seed to init. models
        """
        self.args = args.copy()
        #self.dataset_name = dataset 
        self.seed = seed 
        self.device = args.device 
        self.verbose = args.verbose

        # Make dataset
        #print(self.dataset_name)
        #self.dataset = CLDataset(args, self.dataset_name, self.seed) 
        #self.dataset = self.make_dataset(args, self.dataset_name, self.seed, shuffle_labels)
        #self.dataset = ContinualDataset(args)
        self.shuffle_labels = shuffle_labels

        # Get shorthands from dataset
        #self.dataset_name = self.dataset.dataset # 
        #self.task_ids = self.dataset.task_ids
        self.n_tasks = self.args.n_tasks # use args because when debugging with smaller number of tasks
        self.classes_per_task = self.args.data.classes_per_task
        #self.input_size = self.dataset.input_size

        # Create checkpoint dir
        """
        self.env_name = self._get_env_name()
        self.checkpoint_dir = args.checkpoint_dir + '/' + self.env_name 
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        #create_dir(self.checkpoint_dir)
        """

        # Create trainer
        self.args.n_tasks = self.n_tasks
        self.args.classes_per_task = self.classes_per_task
        #self.args.input_size = self.input_size
        self.args.seed = self.seed
        #self.trainer = ReplaySchedulingTrainer(self.args)
        #self.trainer.checkpoint_dir = self.checkpoint_dir # set checkpoint dir for trainer

        # State and reward parameters
        self.state_add_delta = args.state_add_delta
        self.state_add_delta_max = args.state_add_delta_max
        self.state_add_forgetting = args.state_add_forgetting
        self.state_add_time = args.state_add_time
        self.state_add_bwt = self.args.state_add_bwt
        self.reward_type = args.reward_type
        self.reward_calc = args.reward_calc
        self.reward_penalty = args.reward_penalty
        self.forgetting_calc = args.forgetting_calc
        self.action_space = action_space

        self.ep_res = [] # used when computing states
        self.ep_info = []

        self.logger = Logger(log_dir=args.log_dir,
                        #monitoring=args.monitoring,
                        #monitoring_dir=os.path.join(args.log_dir, 'monitoring', self.env_name)
                        )

    #def make_dataset(self, args, dataset_name, seed, shuffle_labels):
    #    """
    #    if shuffle_labels:
    #        dataset = ShuffledLabelsCLDataset(args, dataset_name, seed)
    #    else:
    #        #dataset = CLDataset(args, dataset_name)
    #        dataset = ContinualDataset(args)
    #    """
    #    dataset = ContinualDataset(args)
    #    return dataset  
        

    def reset(self):
        raise NotImplementedError

    def step(self, action, t):
        raise NotImplementedError

    def _get_state(self, t):
        raise NotImplementedError

    def _get_reward(self, terminal_state=False):
        """ Compute reward
        """
        # Shorthands
        t = len(self.ep_res)
        res = self.ep_res[-1]['val'] 
        reward_type = self.reward_type
        if reward_type == 'sparse':
            reward = 0.0
            if terminal_state:
                if self.reward_calc == 'last':
                    accs = res['acc'][:t] 
                    reward = np.mean(accs)
                elif self.reward_calc == 'full':
                    accs = np.array([res['val']['acc'][:t] for res in self.ep_res])
                    reward = np.sum(accs) / np.sum((accs > 0.0))
                # Add penalty with forgetting metric
                current_task = len(self.ep_res)
                if self.reward_penalty in ['avg_forgetting', 'total_forgetting']:
                    forgetting = self._get_reward_penalty()
                    reward -= forgetting
                
        elif reward_type == 'dense': # NOTE: need to think about this one!
            if self.reward_calc == 'last':
                accs = res['acc'][:t] 
                reward = np.sum(accs) / np.sum(accs > 0).astype(np.float32)
            elif self.reward_calc == 'full':
                accs = np.array([res['val']['acc'][:t] for res in self.ep_res])
                reward = np.sum(accs) / np.sum((accs > 0.0))

            # Add penalty with forgetting metric
            current_task = len(self.ep_res)
            if self.reward_penalty in ['avg_forgetting', 'total_forgetting']:
                forgetting = self._get_reward_penalty()
                reward -= forgetting

        else:
            raise ValueError('Reward type %s is invalid.' %(reward_type))
        return reward

    def _get_reward_penalty(self):
        accs = np.array([res['val']['acc'] for res in self.ep_res])
        current_task = accs.shape[0]-1
        forgetting = self._compute_forgetting(accs, current_task)
        #print('total forgetting: ', forgetting)
        if self.reward_penalty == 'avg_forgetting' and (current_task > 0):
            forgetting = forgetting / current_task
            #print('avg forgetting: ', forgetting)
        return forgetting 

    def _compute_forgetting(self, acc, current_task):
        if current_task == 0:
            return 0.0
        total_forgetting = 0.0
        #print(acc)
        #print()
        for t in range(1, current_task):

            if self.forgetting_calc == 'last':
                max_acc = acc[current_task-1, t]
            elif self.forgetting_calc == 'bwt':
                max_acc = acc[current_task, current_task] # bwt
            elif self.forgetting_calc == 'max':
                max_acc = np.max(acc[:current_task, t])
            last_acc = acc[current_task, t]
            task_forgetting = max_acc - last_acc
            total_forgetting += task_forgetting
        return total_forgetting #/ (current_task)

    def _checkpoint_exists(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        return os.path.isfile(path)

    def _get_env_name(self):
        env_name = '{}-5Split'.format(self.dataset_name)
        if self.dataset_name == 'MNIST':
            for i in range(len(self.dataset.task_ids)):
                env_name += '-{}'.format(self.dataset.task_ids[i])
        env_name += '-Seed-{}'.format(self.seed)
        return env_name

    def print_info(self):
        print('-'*100)
        print('[Environment] {}Split-{} with task ids {}'.format(self.n_tasks, self.dataset_name, self.task_ids))
        print('[Seed] {}'.format(self.seed))
        print('-'*100)

    #######################################################
    # VISUALIZATION                       
    #######################################################
    def log_statistics(self, category, data, it, histogram=False):
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        # Add aggregated statistics
        self.logger.add(category, 'mean', np.mean(data), it)
        self.logger.add(category, 'std', np.std(data), it)
        # Add max/min
        self.logger.add(category, 'max', np.max(data), it)
        self.logger.add(category, 'min', np.min(data), it)
        # Add histogram option
        if histogram:
            self.logger.add_hist(data=data, class_name=category, it=it)

    def log_scalar(self, category, key, value, it):
        self.logger.add(category, key, value, it)

    def plot_aggregated_statistics(self, category):
        # Get stats from category
        assert category in self.logger.stats.keys(), "Key {:s} doesn't exist in logger stats.".format(category)
        means = np.array(self.logger.stats[category]['mean'])
        stds = np.array(self.logger.stats[category]['std'])
        maxs = np.array(self.logger.stats[category]['max'])
        mins = np.array(self.logger.stats[category]['min'])

        save_dir = self.logger.log_dir
        fix, ax = plt.subplots()
        x = np.arange(len(means))
        ax.plot(x, means, 'b-', label='mean')
        ax.fill_between(x, means+stds, means-stds, alpha=0.2)
        ax.plot(x, maxs, 'g-', label='max')
        ax.plot(x, mins, 'r-', label='min')
        ax.legend()
        ax.set_xlabel('batch updates')
        ax.set_ylabel(category)
        ax.set_title('Aggregated stats of {:s}'.format(category))
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/envseed_{:d}_{:s}.png'.format(self.seed, category))
        plt.close()

    def plot_scalar(self, category, key):
        # Get stats from category
        assert category in self.logger.stats.keys(), "Key {:s} doesn't exist in logger stats.".format(category)
        assert key in self.logger.stats[category].keys(), "Key {:s} doesn't exist in logger stats.".format(key)
        values = np.array(self.logger.stats[category][key])

        save_dir = self.logger.log_dir
        fix, ax = plt.subplots()
        x = np.arange(len(values))
        ax.plot(x, values, 'b-')
        ax.set_xlabel('iterations')
        ax.set_ylabel('{:s}/{:s}'.format(category, key))
        ax.set_title('Scalar value of {:s}/{:s}'.format(category, key))
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/envseed_{:d}_{:s}_{:s}.png'.format(self.seed, category, key))
        plt.close()

    def plot_action_trajectories(self, action_trajectories):
        trajectories = np.array(action_trajectories)
        n_actions = np.max(trajectories)
        n_lines = len(trajectories)
        c = np.arange(1, n_lines+1)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        save_dir = self.logger.log_dir
        fig, ax = plt.subplots()
        x = np.arange(1, len(trajectories[0])+1)
        
        for i, traj in enumerate(trajectories, start=1):
            ax.plot(x, traj, 'o-', c=cmap.to_rgba(i + 1))
        ax.set_xlabel('Time step')
        ax.set_ylabel('Action Index')
        ax.set_title('Action trajectories')
        ax.set_xticks(x)
        ax.set_yticks(np.arange(n_actions+1))
        fig.colorbar(cmap)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/envseed_{:d}_action_trajectories.png'.format(self.seed))
        plt.close()

    def plot_returns(self, key, value):
        assert key in self.logger.stats.keys(), "Key {:s} doesn't exist in logger stats.".format(key) 
        values = np.array(self.logger.stats[key][value])

        save_dir = self.logger.log_dir
        fig, ax = plt.subplots()
        x = np.arange(1, len(values)+1)
        ax.plot(x, values, label='{:s}/{:s}'.format(key, value))
        ax.legend()
        ax.set_xlabel('Time step')
        ax.set_ylabel(key)
        ax.set_title('{:s}/{:s}'.format(key, value))        
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/envseed_{:d}_{:s}_{:s}.png'.format(self.seed, key, value))
        plt.close()
     