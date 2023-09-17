
import pickle
import os
import torchvision
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, log_dir='./logs', img_dir=None,
                 monitoring=None, monitoring_dir=None):
        self.stats = dict()
        self.log_dir = log_dir
        self.img_dir = img_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #if not os.path.exists(img_dir):
        #    os.makedirs(img_dir)

        if not (monitoring is None or monitoring == 'none'):
            self.setup_monitoring(monitoring, monitoring_dir)
        else:
            self.monitoring = None
            self.monitoring_dir = None

    def setup_monitoring(self, monitoring, monitoring_dir=None):
        self.monitoring = monitoring
        self.monitoring_dir = monitoring_dir

        if monitoring == 'tensorboard':
            from torch.utils import tensorboard
            self.tb = tensorboard.SummaryWriter(monitoring_dir)
        else:
            raise NotImplementedError('Monitoring tool "%s" not supported!'
                                      % monitoring)

    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        #self.stats[category][k].append((it, v))
        self.stats[category][k].append(v)

        k_name = '%s/%s' % (category, k)
        if self.monitoring == 'tensorboard':
            self.tb.add_scalar(k_name, v, it)

    def add_imgs(self, imgs, class_name, it):
        outdir = os.path.join(self.img_dir, class_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, '%08d.png' % it)

        #imgs = imgs / 2 + 0.5
        imgs = torchvision.utils.make_grid(imgs)
        torchvision.utils.save_image(imgs, outfile, nrow=8)

        if self.monitoring == 'tensorboard':
            self.tb.add_image(class_name, imgs, it)

    def add_hist(self, data, class_name, it):
        if self.monitoring == 'tensorboard':
            self.tb.add_histogram(class_name, data, it)

    def add_fig(self, fig, class_name, it):
        """ Note fig is of type (matplotlib.pyplot.figure)   
        """
        if self.monitoring == 'tensorboard':
            self.tb.add_figure(class_name, fig, it)

    def get_last(self, category, k, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            #return self.stats[category][k][-1][1]
            return self.stats[category][k][-1]

    def save_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        if not os.path.exists(filename):
            print('Warning: file "%s" does not exist!' % filename)
            return

        try:
            with open(filename, 'rb') as f:
                self.stats = pickle.load(f)
        except EOFError:
            print('Warning: log file corrupted!')

    ### my functions
    def log_scalar(self, category, key, value, it):
        self.add(category, key, value, it)

    def plot_scalar(self, category, key):
        # Get stats from category
        assert category in self.stats.keys(), "Key {:s} doesn't exist in logger stats.".format(category)
        assert key in self.stats[category].keys(), "Key {:s} doesn't exist in logger stats.".format(key)
        values = np.array(self.stats[category][key])

        save_dir = self.log_dir
        fix, ax = plt.subplots()
        x = np.arange(len(values))
        ax.plot(x, values, 'b-')
        ax.set_xlabel('iterations')
        ax.set_ylabel('{:s}/{:s}'.format(category, key))
        ax.set_title('Scalar value of {:s}/{:s}'.format(category, key))
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/{:s}-{:s}.png'.format(category, key))
        plt.close()

class LoggerDQN(Logger):

    def __init__(self, log_dir='./logs', img_dir=None,
                monitoring=None, monitoring_dir=None):
        super().__init__(log_dir, img_dir, monitoring, monitoring_dir)

    def log_statistics(self, category, data, it, histogram=False):
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        # Add aggregated statistics
        self.add(category, 'mean', np.mean(data), it)
        self.add(category, 'std', np.std(data), it)
        # Add max/min
        self.add(category, 'max', np.max(data), it)
        self.add(category, 'min', np.min(data), it)
        # Add histogram option
        if histogram:
            self.add_hist(data=data, class_name=category, it=it)

    def plot_returns(self, key, value):
        assert key in self.stats.keys(), "Key {:s} doesn't exist in logger stats.".format(key) 
        values = np.array(self.stats[key][value])
        if 'running_mean/%s' %value in self.stats[key].values():
            run_means = np.array(self.stats[key]['running_mean/%s' %value])

        save_dir = self.log_dir
        fig, ax = plt.subplots()
        x = np.arange(1, len(values)+1)
        ax.plot(x, values, label='{:s}/{:s}'.format(key, value))
        if 'running_mean/%s' %value in self.stats[key].values():
            ax.plot(x, run_means, label='Running mean')
        ax.legend()
        ax.set_xlabel('Time step')
        ax.set_ylabel(key)
        ax.set_title('Val. Env. {:s} {:s}'.format(key, value))        
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/val_{:s}_{:s}.png'.format(key, value))
        plt.close()

    def plot_aggregated_statistics(self, category):
        # Get stats from category
        assert category in self.stats.keys(), "Key {:s} doesn't exist in logger stats.".format(category)
        means = np.array(self.stats[category]['mean'])
        stds = np.array(self.stats[category]['std'])
        maxs = np.array(self.stats[category]['max'])
        mins = np.array(self.stats[category]['min'])

        save_dir = self.log_dir
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
        plt.savefig(save_dir + '/{:s}.png'.format(category))
        plt.close()

    def plot_action_trajectories(self, action_trajectories):
        trajectories = np.array(action_trajectories)
        n_actions = np.max(trajectories)
        n_lines = len(trajectories)
        c = np.arange(1, n_lines+1)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        save_dir = self.log_dir
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
        plt.savefig(save_dir + '/action_trajectories.png')
        plt.close()

    def plot_gradient_statistics(self):
        # plot abs-max and mean square value of gradients
        assert 'grad_abs_max' in self.stats.keys(), "Key {:s} doesn't exist in logger stats.".format('grad_abs_max')
        grad_abs_max = self.stats['grad_abs_max']
        
        save_dir = self.log_dir
        fix, ax = plt.subplots()
        for name, abs_max in grad_abs_max.items():
            x = np.arange(len(abs_max))
            ax.plot(x, abs_max, label=name)
        ax.legend()
        ax.set_xlabel('batch updates')
        ax.set_ylabel('max(|grad|)')
        ax.set_title('Absolute maximum of gradients')
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/grad_abs_max.png')
        plt.close() 

        assert 'grad_mean_sq' in self.stats.keys(), "Key {:s} doesn't exist in logger stats.".format('grad_mean_sq')
        grad_mean_sq = self.stats['grad_mean_sq']
        fix, ax = plt.subplots()
        for name, mean_sq_value in grad_mean_sq.items():
            x = np.arange(len(mean_sq_value))
            ax.plot(x, mean_sq_value, label=name)
        ax.legend()
        ax.set_xlabel('batch updates')
        ax.set_ylabel('mean(grad**2)')
        ax.set_title('Mean square value of gradients')
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir + '/grad_mean_sqmean_sq_grad.png')
        plt.close()  