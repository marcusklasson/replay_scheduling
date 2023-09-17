
import os
import time
import numpy as np
import torch
#from torch.utils.data import DataLoader

from training.logger import Logger
from training.config import (
    build_models, build_optimizers, build_lr_scheduler
)
from trainer.utils import get_data_loader

class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.dataset = config['data']['name']
        self.batch_size = config['training']['batch_size']
        self.scenario = config['training']['scenario']
        self.classes_per_task = config['training']['classes_per_task']

        self.out_dir = config['training']['out_dir']
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.log_dir = config['training']['log_dir']
        self.device = config['session']['device']
        self.num_workers = config['training']['nworkers']
        self.pin_memory = config['training']['pin_memory']
        self.print_every = config['training']['print_every']

        self.model_file = config['training']['model_file']
        self.n_tasks = config['data']['n_tasks']
        self.seed = config['session']['seed']
        self.verbose = config['session']['verbose']

        self.logger = Logger(
            log_dir=os.path.join(self.out_dir, 'logs', 'seed%d' %(self.seed)),
            monitoring=config['training']['monitoring'],
            monitoring_dir=os.path.join(self.out_dir, 'monitoring', 'seed%d' %(self.seed))
            )

        # Initialize model and optimizer
        self.model = build_models(config)
        #print(self.model)
        self.model = self.model.to(self.device)
        self.optimizer = self.prepare_optimizer() #build_optimizers(net=self.model, config=config)

        # Save initial checkpoint 
        self.path_initial_checkpoint = self.checkpoint_dir + '/cl_model_seed%d.pt' %(self.seed)
        self.save_initial_checkpoint(path=self.path_initial_checkpoint)

        self.n_epochs = config['training']['n_epochs']
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)

        self.last_trained_task = 0 # to keep track of task, useful for getting allowed tasks in evaluation
        print()

        self.gen_pytorch = torch.Generator().manual_seed(self.seed)

        self.current_task = 0
        self.method = config['training']['method']

        if self.method in ['agem', 'er', 'rs', 'coreset', 'heuristic_scheduling', 'der', 'derpp', 'mer', 'hal']:
            self.use_episodic_memory = True
        else:
            self.use_episodic_memory = False 

        # Simple memory
        if self.use_episodic_memory:
            self.memories_per_class = config['replay']['examples_per_class']
            self.n_classes = config['data']['n_classes'] 
            #if self.scenario=='domain':
            # in domain-il setting
            # self.n_memories = self.n_classes * self.memories_per_class * self.n_tasks 
            #self.count_cls = torch.zeros(self.n_classes*self.n_tasks, dtype=torch.long)
            
            self.n_memories = self.classes_per_task * self.memories_per_class * self.n_tasks 
            self.count_cls = torch.zeros(self.n_classes, dtype=torch.long)
            #print('self.n_memories: ', self.n_memories)
            img_size = config['data']['img_size']
            in_channel = config['data']['in_channel']
            self.episodic_images = torch.FloatTensor(self.n_memories, in_channel, img_size, img_size)
            self.episodic_labels = -torch.ones(self.n_memories, dtype=torch.long) #torch.LongTensor(self.n_memories)
            
            #self.count_cls = torch.zeros(self.n_classes, dtype=torch.long)
            
            self.episodic_filled_counter = 0 # used for ring buffer
            self.examples_seen_so_far = 0 # used for reservoir sampling
            # Add tensors to gpu or cpu
            self.episodic_images = self.episodic_images.to(self.device)
            self.episodic_labels = self.episodic_labels.to(self.device)

    def prepare_optimizer(self):
        optimizer = build_optimizers(net=self.model, config=self.config)
        return optimizer

    def train_single_task(self, task_id, train_dataset, valid_datasets=None): 
        """ Train model on single task dataset.

            Args:
                task_id (int): Task identifier (splitMNIST: 1-5).
                train_dataset (torch.Dataset): Training dataset for current task.
                replay_dataset (dict with torch.Tensor): Replay data from previous tasks in each dictionary slot.
        """

        # For reproducibility in tree searches, setting the PyTorch seed here is important
        # to get the same batches in the dataloader. Set it based on task_id, so it will
        # be the same when we run the tree search in run_tree_search.py.
        # We do not set numpy.random.seed becuase this is used for shuffling replay
        # samples within a batch (necessary?) and also the node selection in MCTS. 
        #torch.manual_seed(2021+task_id)
        #np.random.seed(2021+task_id)
        self.model.train()

        # Shorthands
        scenario = self.scenario
        classes_per_task = self.classes_per_task
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        device = self.device
        logger = self.logger
        verbose = self.verbose

        # if Resnet and first task, train for 5 epochs
        if (self.config['model']['net'] == 'resnet18') and (task_id == 1):
            n_epochs = 5

        # Reset optimizer before every task in ER
        if self.config['training']['reset_optimizer']:
            print('resetting optimizer')
            self.optimizer = self.prepare_optimizer()

        active_classes = self._get_active_classes_up_to_task_id(task_id)
        data_loader = get_data_loader(train_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=True,
                                    rng=self.gen_pytorch)
        t0 = time.time()
        for epoch in range(n_epochs):
            loss_curr = 0.0
            loss_replay = 0.0
            acc, acc_replay = 0.0, 0.0
            for batch_idx, (x, y) in enumerate(data_loader):
                #-----------------Collect data------------------#
                ### Current Batch
                #--> ITL: adjust current y-targets to 'active range', e.g. [0, 1] if 2 classes/task 
                if isinstance(classes_per_task, list): # adjusting range is different for Omniglot though
                    class_offset = active_classes[-1][0] # get first index of current class tasks
                    y_curr = y-class_offset if (scenario == "task") else y 
                else:
                    y_curr = y-classes_per_task*(task_id-1) if (scenario == "task") else y  
                x_curr, y_curr = x.to(device), y_curr.to(device) #--> transfer them to correct device
                ### Get replay batch
                x_replay, y_replay = self.get_replay_batch(task=task_id) #None   #-> if no replay

                # Train the main model with this batch
                loss_dict = self.train_batch(x_curr, y_curr, 
                                            x_=x_replay, y_=y_replay,
                                            active_classes=active_classes, 
                                            task=task_id,)
                # Add batch results to metrics
                loss_curr += loss_dict['loss_current']
                loss_replay += loss_dict['loss_replay']
                acc += loss_dict['accuracy']
                acc_replay += loss_dict['accuracy_replay']
            # End of epoch
            loss_curr = loss_curr / (batch_idx + 1)
            loss_replay = loss_replay / (batch_idx + 1)
            acc = acc / (batch_idx + 1)
            acc_replay = acc_replay / (batch_idx + 1)

            # Add metrics to logger
            epochs_total = (n_epochs * (task_id-1)) + (epoch+1)
            logger.add('loss', 'train', loss_curr, it=epochs_total) 
            logger.add('accuracy', 'train', acc, it=epochs_total) 

            # Print stats
            loss_curr_last = logger.get_last('loss', 'train')
            acc_last = logger.get_last('accuracy', 'train')
            if ((epoch+1) % self.print_every) == 0:
                print('[epoch %3d/%3d] loss current = %.4f, acc = %.4f, loss replay = %.4f, acc replay = %.4f'
                        % (epoch+1, n_epochs, loss_curr, acc, loss_replay, acc_replay))
            
        ###################### END of training task ######################
        #print('training time: ', time.time() - t0)
        print('End training for task %d...' % task_id)
        self.last_trained_task = task_id
        self.logger.save_stats('stats.p')

    def train_batch(self, x, y, x_=None, y_=None, active_classes=None, task=1, rnt=0.5):
        raise NotImplementedError

    def get_replay_batch(self, task):
        raise NotImplementedError

    #def train_batch_new_memory(self, x, y, x_=None, y_=None, active_classes=None, task=1):
    #    raise NotImplementedError

    def evaluate_task(self, task_id, dataset, test_size=None):
        """ Evaluate model in trainer on dataset.
            Args:
                task_id (int): Task id number.
                dataset (torch.Dataset): Dataset to evlauate model on.
                test_size (int): Number of data points to evaluate (None means evaluate on all data points).
            Returns:
                res (dict): Computed metrics like loss and accuracy.
        """
        total_loss, total_tested, total_correct = 0, 0, 0
        batch_idx = 0

        self.model.eval()
        classes_per_task = self.classes_per_task
        scenario = self.scenario
        device = self.device
        # Get allowed classes and data loader
        allowed_classes = self._get_active_classes(task_id)
        #data_loader = DataLoader(dataset, batch_size=self.batch_size, 
        #                        num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
        data_loader = get_data_loader(dataset,
                                    batch_size=512,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=False)
        res = {}
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                # -break on [test_size] (if "None", full dataset is used)
                if test_size:
                    if total_tested >= test_size:
                        break
                y = y-allowed_classes[0] if (scenario == "task") else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
                y_scores = self.model(x)[:, allowed_classes] # self.model(x) if (allowed_classes is None) else 
                _, y_predicted = torch.max(y_scores, 1)
                # Get accuracy
                total_correct += (y_predicted == y).sum().item()
                total_tested += x.size(0)
                # Get loss
                loss_t = self.criterion(y_scores, y) #F.cross_entropy(y_predicted, y, reduction='mean')
                total_loss += loss_t

        res['acc_t'] = total_correct / total_tested
        res['loss_t'] = total_loss.item() / (batch_idx + 1)
        self.model.train()
        return res

    def test(self, task_id, dataset, model, batch_size=512):
        """ Evaluate given model on dataset.
            Args:
                task_id (int): Task id number.
                dataset (torch.Dataset): Dataset to evlauate model on.
                model (nn.Module): Model to test on dataset.
            Returns:
                res (dict): Computed metrics like loss and accuracy.
        """
        total_loss, total_tested, total_correct = 0, 0, 0
        batch_idx = 0

        model.eval()
        device = self.device
        classes_per_task = self.classes_per_task
        scenario = self.scenario

        # Get task id for allowed classes and data loader
        task_id = self.last_trained_task if (scenario == 'class') else task_id
        allowed_classes = self._get_active_classes(task_id)
        #print(allowed_classes)
        #data_loader = DataLoader(dataset, batch_size=self.batch_size, 
        #                        num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
        data_loader = get_data_loader(dataset,
                                    batch_size=batch_size,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=False)
        res = {}
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                y = y-allowed_classes[0] if (scenario == "task") else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
                y_scores = model(x)[:, allowed_classes] # if (allowed_classes is None) else 
                _, y_predicted = torch.max(y_scores, 1)

                # Get accuracy
                total_correct += (y_predicted == y).sum().item()
                total_tested += x.size(0)
                # Get loss
                loss_t = self.criterion(y_scores, y)
                total_loss += loss_t

        res['acc_t'] = total_correct / total_tested
        res['loss_t'] = total_loss.item() / (batch_idx + 1)
        model.train()
        return res

    def _get_active_classes_up_to_task_id(self, task_id):
        """ Get the active classes up to the given task_id.
            Args:
                task_id (int): Task identity, starting from 1
            Returns:
                active_classes: The active class outputs in the classification layer.
        """
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
                #print('active classes for current task: ', active_classes[-1])
            else: 
                active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task_id)]
        elif scenario == "class": # NOTE: not implemented yet!!!
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            active_classes = list(range(classes_per_task * task_id))
        elif scenario == "domain": # only used for ['PermutedMNIST', 'RotatedMNIST']:
            active_classes = list(range(classes_per_task))
        return active_classes

    def _get_active_classes(self, task_id):
        """ Get active classes for specific task id.
            Args:
                task_id (int): Task identity, starting at 1
            Returns:
                active_classes (list): Active classes in classification layer for given task id. 
        """
        classes_per_task = self.classes_per_task
        scenario = self.scenario

        active_classes = None 
        if scenario == 'task':
            if isinstance(classes_per_task, list): # classes_per_task is list for Omniglot due to different n_classes/task 
                n_classes_task = classes_per_task[task_id-1][1] # have to use task_id-1 because omniglot task index starts at 0
                offset = classes_per_task[task_id-1][2]
                active_classes = list(range(offset, n_classes_task+offset))
            else:
                active_classes = list(range(classes_per_task*(task_id-1), classes_per_task*(task_id)))
        elif scenario == 'class':
            active_classes = list(range(classes_per_task * task_id))
        elif scenario == "domain": # only used for ['PermutedMNIST', 'RotatedMNIST']:
            active_classes = list(range(classes_per_task))
        return active_classes

    def save_checkpoint(self, task_id, folder='./', file_name='model.pth.tar'):
        """ Save checkiṕoint for model and optimizer in trainer to file.
        """
        print("Saving model and optimizer for task {} at {}...\n".format(task_id, os.path.join(folder, file_name)))
        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'task_id': task_id,
                }
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(state, os.path.join(folder, file_name))

    def load_checkpoint(self, checkpoint_dir, file_path):
        """ Load checkpoint for model and optimizer to trainer from file.
        """
        # Build model and optimizer
        model = build_models(self.config)
        optimizer = build_optimizers(net=model, config=self.config)
        # Load model
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
        """ Load model from file and return it.
        """
        model = build_models(self.config)
        model = model.to(self.device)
        fname = os.path.join(self.checkpoint_dir, file_name)
        #print('Loading checkpoint from file {}...'.format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def save_initial_checkpoint(self, path):
        model = self.model
        #optimizer = self.optimizer
        torch.save({'model_state_dict': model.state_dict(),
                    #'optimizer_state_dict': optimizer.state_dict(),
                    }, path)

    def load_initial_checkpoint(self):
        model = self.model
        #optimizer = self.optimizer
        checkpoint = torch.load(self.path_initial_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model#, optimizer