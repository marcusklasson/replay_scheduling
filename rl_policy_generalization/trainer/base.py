import os
import time
import numpy as np
import torch

from training.logger import Logger
from trainer.config import build_models, build_optimizers
from trainer.utils import get_data_loader
from trainer.networks.mlp import MLP
from training.utils import set_random_seed

class Trainer(object):

    def __init__(self, config):
        self.config = config.copy()
        #print(config)
        self.dataset = config['dataset']
        self.scenario = config['cl_scenario']
        self.classes_per_task = config['classes_per_task']

        self.out_dir = config['out_dir']
        self.checkpoint_dir = config['checkpoint_dir']
        self.log_dir = config['log_dir']
        self.device = config['device']
        self.num_workers = config['num_workers']
        self.pin_memory = config['pin_memory']
        self.print_every = config['cl']['print_every']

        #print('nworkers: ', self.num_workers)
        #print('pinmemory: ', self.pin_memory)

        self.n_tasks = config['n_tasks']
        self.seed = config['cl']['seed']
        self.verbose = config['verbose']

        self.logger = Logger(
            log_dir=os.path.join(self.log_dir),
            monitoring=config['monitoring'],
            monitoring_dir=os.path.join(self.log_dir, 'monitoring')
            )

        # Initialize model and optimizer
        #set_random_seed(self.seed)
        torch.manual_seed(self.seed)
        self.model = build_models(config)
        self.model = self.model.to(self.device)
        self.optimizer = self.prepare_optimizer()

        # Save initial checkpoint 
        self.path_initial_checkpoint = self.checkpoint_dir + '/cl_model_seed%d.pth.tar' %(self.seed)
        self.save_initial_checkpoint(path=self.path_initial_checkpoint)

        self.n_epochs = config['cl']['n_epochs']
        self.batch_size = config['cl']['batch_size']
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)
        
        print()

    def prepare_optimizer(self):
        optimizer = build_optimizers(net=self.model, config=self.config)
        return optimizer

    def train_single_task(self, task_id, train_dataset, valid_datasets=None): 
        raise NotImplementedError

    def train_batch(self, x, y, x_=None, y_=None, active_classes=None, task=1, rnt=0.5):
        raise NotImplementedError

    def get_replay_batch(self, task):
        raise NotImplementedError

    def eval_task(self, task_id, dataset, test_size=None):
        # Evaluate model on dataset.
        total_loss, total_tested, total_correct = 0, 0, 0
        batch_idx = 0

        self.model.eval()
        classes_per_task = self.classes_per_task
        scenario = self.scenario
        device = self.device
        # Get allowed classes and data loader
        allowed_classes = self._get_active_classes(task_id)
        data_loader = get_data_loader(dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=False)
        res = {}
        with torch.no_grad():
            #for batch_idx, (x, y, t) in enumerate(data_loader):
            for batch_idx, (x, y) in enumerate(data_loader):
                # -break on [test_size] (if "None", full dataset is used)
                if test_size:
                    if total_tested >= test_size:
                        break
                y = y-allowed_classes[0] if (scenario == "task") else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                         #--> transfer them to correct device
                #y_scores = self.model(x)[:, allowed_classes] # self.model(x) if (allowed_classes is None) else 
                y_logits = self.model(x)[:, allowed_classes] #y_logits = self.model(x, t)
                #print('y_logits: ', y_logits)
                #print('t: ', t)
                """
                if self.scenario == 'class': # class incremental learning
                    offset = (task_id+1) * self.classes_per_task 
                    y_logits[:, offset:].data.fill_(-10e10)  
                """ 
                _, y_pred = torch.max(y_logits, 1)
                # Get accuracy
                total_correct += (y_pred == y).sum().item()
                total_tested += x.size(0)
                # Get loss
                loss_t = self.criterion(y_logits, y) 
                total_loss += loss_t

        res['acc_t'] = total_correct / total_tested
        res['loss_t'] = total_loss.item() / (batch_idx + 1)
        self.model.train()
        return res

    
    def test(self, task_id, dataset, model):
        # Evaluate given model on dataset.
        total_loss, total_tested, total_correct = 0, 0, 0
        batch_idx = 0

        model.eval()
        device = self.device
        classes_per_task = self.classes_per_task
        scenario = self.scenario

        # Get task id for allowed classes and data loader
        allowed_classes = self._get_active_classes(task_id)
        data_loader = get_data_loader(dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=False)
        res = {}
        with torch.no_grad():
            #for batch_idx, (x, y, t) in enumerate(data_loader):
            for batch_idx, (x, y) in enumerate(data_loader):
                #x, y, t = x.to(device), y.to(device), t.to(device)
                y = y-allowed_classes[0] if (scenario == "task") else y 
                #print('y: ', y)
                x, y = x.to(device), y.to(device) 
                y_logits = model(x)[:, allowed_classes] #y_logits = model(x, t)
                """
                if self.scenario == 'class': # class incremental learning
                    offset = (task_id+1) * self.classes_per_task 
                    y_logits[:, offset:].data.fill_(-10e10)
                """   
                _, y_pred = torch.max(y_logits, 1)
                # Get accuracy
                total_correct += (y_pred == y).sum().item()
                total_tested += x.size(0)
                # Get loss
                loss_t = self.criterion(y_logits, y) 
                total_loss += loss_t

        res['acc_t'] = total_correct / total_tested
        res['loss_t'] = total_loss.item() / (batch_idx + 1)
        model.train()
        return res
    
    def _get_active_classes_up_to_task_id(self, task_id):
        # Get the active class up to the given task_id to fetch valid network outputs.
        classes_per_task = self.classes_per_task
        scenario = self.scenario

        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            if isinstance(classes_per_task, list): # classes_per_task is list for Omniglot due to different n_classes/task 
                active_classes = []
                seen_classes = 0
                for t in range(task_id):
                    n_classes = classes_per_task[t][1]
                    active_classes.append(list(range(seen_classes, seen_classes+n_classes)))
                    seen_classes += n_classes 
            else: 
                active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task_id)]
        elif scenario == "class": # NOTE: not implemented yet!!!
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            print('not implemented yet, must be based on current task id!!!')
            active_classes = list(range(classes_per_task * task_id))
        elif scenario == "domain": # only used for ['PermutedMNIST', 'RotatedMNIST']:
            active_classes = list(range(classes_per_task))
        return active_classes

    def _get_active_classes(self, task_id):
        # Get the active class for the given task_id to fetch valid network outputs.
        classes_per_task = self.classes_per_task
        scenario = self.scenario

        active_classes = None 
        if scenario == 'task':
            if isinstance(classes_per_task, list): # classes_per_task is list for Omniglot due to different n_classes/task 
                n_classes_task = classes_per_task[task_id-1][1] # have to use task_id-1 because omniglot task index starts at 0
                offset = classes_per_task[task_id-1][2]
                active_classes = list(range(offset, n_classes_task+offset))
            else:
                active_classes = list(range(classes_per_task*task_id, classes_per_task*(task_id+1)))
        elif scenario == 'class':
            print('not implemented yet, must be based on current task id!!!')
            active_classes = list(range(classes_per_task * (task_id+1)))
        elif scenario == "domain": # only used for ['PermutedMNIST', 'RotatedMNIST']:
            active_classes = list(range(classes_per_task))
        return active_classes

    def save_checkpoint(self, task_id, folder='./', file_name='model.pth.tar'):
        # Save checkiṕoint for model and optimizer in trainer to file.
        print("Saving model and optimizer for task {} at {}...\n".format(task_id, os.path.join(folder, file_name)))
        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'task_id': task_id,
                }
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(state, os.path.join(folder, file_name))

    def load_checkpoint(self, checkpoint_dir, file_path):
        # Load checkpoint for model and optimizer to trainer from file.
        # Build model and optimizer
        model = build_models(self.config)
        optimizer = build_optimizers(net=model, config=self.config)
        # Load model
        print("Loading checkpoint at {}...\n".format(os.path.join(checkpoint_dir, file_path)))
        checkpoint = torch.load(os.path.join(checkpoint_dir, file_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        # Load optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        task_id = checkpoint['task_id']
        # Set model and optimizer
        self.model = model
        self.optimizer = optimizer
    
    def load_model_from_file(self, file_name):
        # Load model from file and return it.
        model = build_models(self.config)
        model = model.to(self.device)
        fname = os.path.join(self.checkpoint_dir, file_name)
        #print('Loading checkpoint from file {}...'.format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def save_initial_checkpoint(self, path):
        model = self.model
        torch.save({'model_state_dict': model.state_dict(),
                    }, path)

    def load_initial_checkpoint(self):
        model = self.model
        checkpoint = torch.load(self.path_initial_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _get_env_name(self):
        env_name = '{}-5Split'.format(self.dataset_name)
        for i in range(len(self.dataset.task_ids)):
            env_name += '-{}'.format(self.dataset.task_ids[i])
        env_name += '-Seed-{}'.format(self.seed)
        return env_name