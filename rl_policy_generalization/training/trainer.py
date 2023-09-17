

import os
import time
import numpy as np
import torch

from training.memory import ReplayMemory
from training.logger import Logger
from training.config import (
    build_models, build_optimizers, build_models_using_rng,
)

class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.checkpoint_dir = args.checkpoint
        self.scenario = args.cl_scenario
        self.num_tasks = args.n_tasks
        self.batch_size = args.cl.batch_size
        self.num_epochs = args.cl.n_epochs
        self.num_workers = args.num_workers
        self.device = args.device

        self.classes_per_task = args.classes_per_task 
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.logger = Logger(log_dir=args.log)
        self.print_every = args.cl.print_every

        self.replay_memory = ReplayMemory(args)
        self.replay_mode = args.replay_method
        self.memory_size = args.memory_size 

        self.model = build_models(self.args)
        self.model = self.model.to(self.device)
        self.optimizer = build_optimizers(net=self.model, args=self.args)
        
        #self.model = None 
        #self.optimizer = None 

        self.verbose = args.verbose

        print()

    #"""
    def build_model(self):
        self.model = build_models(self.args)
        self.model = self.model.to(self.device)
        self.optimizer = build_optimizers(net=self.model, args=self.args)
    """
    def build_model_using_rng(self, rng):
        self.model = build_models_using_rng(self.args, rng)
        self.model = self.model.to(self.device)
        self.optimizer = build_optimizers(net=self.model, args=self.args)
    """
    def train_single_task(self, task_id, train_loader, replay_dataset={}):
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
        #torch.manual_seed(self.args.seed)
        #np.random.seed(2021+task_id)
        self.model.train()

        # Shorthands
        scenario = self.scenario
        classes_per_task = self.classes_per_task
        n_epochs = self.num_epochs
        batch_size = self.batch_size
        replay_mode = self.replay_mode
        device = self.device
        logger = self.logger

        ## Get active classes and memory
        #active_classes = self._get_active_classes_up_to_task_id(task_id)

        # Get memory data from dict and put in tensors
        if len(replay_dataset) > 0 and (replay_mode not in ['none']):
            for i, (task, memory_task) in enumerate(replay_dataset.items()):
                mem_size = len(memory_task[1])
                #mem_indices = torch.randperm(mem_size) # shuffle the selected memory points within the task
                #mem_indices = torch.arange(mem_size) #if (self.verbose==2) else torch.randperm(mem_size) 
                x_temp = memory_task[0] #[mem_indices]
                y_temp = memory_task[1] #[mem_indices]
                t_temp = torch.ones(mem_size, dtype=torch.long)*task
                if i > 0:
                    x_mem = torch.cat([x_mem, x_temp], dim=0)
                    y_mem = torch.cat([y_mem, y_temp], dim=-1) # all labels are in range [0, 1] in Split MNIST
                    t_mem = torch.cat([t_mem, t_temp], dim=-1)
                else:
                    x_mem = x_temp
                    y_mem = y_temp 
                    t_mem = t_temp

        replay_data = None
        for epoch in range(n_epochs):
            loss_total = 0.0
            loss = 0.0
            loss_replay = 0.0
            acc = 0.0
            acc_replay = 0.0
            t0 = time.time()
            for batch_idx, (x, y, t) in enumerate(train_loader):
                
                #if batch_idx == 0:
                #    print(y[:10])
                #-----------------Collect data------------------#
                ### Current Batch
                """
                #--> ITL: adjust y-targets to 'active range'
                if isinstance(classes_per_task, list): # adjusting range is different for Omniglot though
                    class_offset = active_classes[-1][0] # get first index of current class tasks
                    y = y-class_offset if (scenario == "task") else y 
                """

                x, y, t = x.to(device), y.to(device), t.to(device) #--> transfer them to correct device

                ### Replay Batch
                if (replay_mode == 'none') or (len(replay_dataset) == 0):
                    x_ = y_ = t_ = None   #-> if no replay
                else:
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    if batch_size < len(y_mem): # If memory data is greater than batch size, randomly sample memory data every batch
                        indices = torch.randperm(len(y_mem))[:batch_size]
                        x_ = x_mem[indices]
                        y_ = y_mem[indices]
                        t_ = t_mem[indices]
                        #replay_data = {'x': x_.to(device), 'y': y_.to(device), 't': t_.to(device)}
                    
                    else:
                        x_ = x_mem
                        y_ = y_mem
                        t_ = t_mem
                    # Put all replay data in dict and put all tensors on right device
                    replay_data = {'x': x_.to(device), 'y': y_.to(device), 't': t_.to(device)}  

                # Train the main model with this batch
                loss_dict = self.train_batch(x, y, t, 
                                            replay_data=replay_data,
                                            task=task_id,
                                            rnt=0.5)#1/task_id)
                
                # Add batch results to metrics
                loss_total += loss_dict['loss_total']
                loss += loss_dict['loss_current']
                loss_replay += loss_dict['loss_replay']
                acc += loss_dict['accuracy']
                acc_replay += loss_dict['accuracy_replay']
            
            if self.verbose == 3:
                print('seconds for epoch: ', time.time()-t0)
            # End of epoch
            loss_total = loss_total / (batch_idx + 1)
            loss = loss / (batch_idx + 1)
            loss_replay = loss_replay / (batch_idx + 1)
            acc = acc / (batch_idx + 1)
            acc_replay = acc_replay / (batch_idx + 1)

            # Add metrics to logger
            epochs_total = (n_epochs * (task_id-1)) + (epoch+1)
            logger.add('loss', 'train', loss_total, it=epochs_total) 
            logger.add('acc', 'train', acc, it=epochs_total) 
            logger.add('loss_t', task_id, loss_total, it=epoch+1) 
            logger.add('acc_t', task_id, acc, it=epoch+1) 

            # Print stats
            if ((epoch+1) % self.print_every) == 0:
                print('[epoch %3d/%3d] loss total = %.4f, Current: acc = %.4f, loss = %.4f, Replay: acc = %.4f, loss = %.4f'
                        % (epoch+1, n_epochs, loss_total, acc, loss, acc_replay, loss_replay))
            
        ###################### END of training task ######################
        #print('End training for task %d...' % task_id)
        self.logger.save_stats('cl_stats.p')

    def train_batch(self, x, y, t, replay_data=None, task=1, rnt=0.5):
            """ Train model on single batch of current task samples
                and (optionally) replay samples for the Task Incremental
                Learning setting.
                Args:
                    x (torch.Tensor): Input data from current task. 
                    y (torch.LongTensor): Labels from current task.
                    t (torch.LongTensor): Task labels from current task.
                    replay_data (dict with torch.Tensors): Input data, labels, and task labels from replay tasks. 
                    task (int): Current task id starting from 1, e.g. splitMNIST: 1-5
                    rnt (float): Weight constant for current and replay losses (default = 0.5) 
                Returns:
                    loss_dict (dict): Dictionary with loss and accuracy metrics.
            """
            #toggle_grad(self.model, True)
            self.model.train()
            self.optimizer.zero_grad()
            # Shorthands
            classes_per_task = self.classes_per_task
            scenario = self.scenario

            ##--(1)-- REPLAYED DATA --##
            if replay_data is not None:
                xr, yr, tr = replay_data['x'], replay_data['y'], replay_data['t']
                if scenario == 'task':
                    # Get task-specific active indices
                    active_indices = torch.arange(classes_per_task, dtype=torch.long, device=self.device)
                    active_indices1 = active_indices.repeat(len(tr), 1)
                    active_indices2 = active_indices1 + (tr*classes_per_task).unsqueeze(1)
                    yr_hat = self.model(xr).gather(1, active_indices2)
                    
                elif scenario == 'class':
                    # adjust class labels so they match the active indices
                    active_indices = torch.arange(task*classes_per_task, dtype=torch.long, device=self.device)
                    yr = yr + (tr*classes_per_task)
                    yr_hat = self.model(xr)[:, active_indices]

                loss_replay = self.loss(yr_hat, yr)
            accuracy_replay = None if (replay_data is None) else (yr == yr_hat.max(1)[1]).sum().item() / xr.size(0)
            loss_replay = None if (replay_data is None) else loss_replay

            ##--(2)-- CURRENT DATA --##
            y_hat = self.model(x)
            # get active indices based on CL scenario and task labels
            if scenario == 'task':  
                active_indices = torch.arange(classes_per_task, dtype=torch.long, device=self.device)
                active_indices += (torch.unique(t)*classes_per_task) # shift active outputs to correct task range  
            elif scenario == 'class':
                active_indices = torch.arange(task*classes_per_task, dtype=torch.long, device=self.device)
                y = y + (t*classes_per_task) # shift class labels with task labels
            # Compute loss
            y_hat = y_hat[:, active_indices]
            loss_cur = self.loss(y_hat, y)
            accuracy = (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # Combine loss from current and replayed batch
            if replay_data is None:
                loss_total = loss_cur
            else:
                loss_total = loss_replay if (x is None) else rnt*loss_cur+(1-rnt)*loss_replay
            loss_total.backward()
            self.optimizer.step()

            # Return the dictionary with different training-loss split in categories
            return {
                'loss_total': loss_total.item(),
                'loss_current': loss_cur.item() if x is not None else 0,
                'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
                'accuracy': accuracy if accuracy is not None else 0.,
                'accuracy_replay': accuracy_replay if accuracy_replay is not None else 0.,
            }

    def eval_model(self, current_task_id, data_loader, test_size=None):
        """ Evaluate model in trainer on dataset.
            Args:
                current_task_id (int): Task id that model has been trained up to, ex. Split-MNIST range [1, 5]. 
                data_loader (torch.DataLoader): Dataloader for dataset to evlauate model on.
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

        res = {}
        with torch.no_grad():
            for batch_idx, (x, y, t) in enumerate(data_loader):
                x, y, t = x.to(device), y.to(device), t.to(device) #--> transfer them to correct device 
                # -break on [test_size] (if "None", full dataset is used)
                if test_size:
                    if total_tested >= test_size:
                        break
                # Adjust class range if class-il setting
                if scenario == 'task':
                    active_outputs = torch.arange(classes_per_task, dtype=torch.long, device=device) 
                    active_outputs += (torch.unique(t)*classes_per_task) # shift active outputs to correct task range
                elif scenario == 'class':
                    y = y + (t*classes_per_task) # shift true class labels
                    active_outputs = torch.arange(current_task_id*classes_per_task, dtype=torch.long, device=device)
                # Make predictions                    
                y_scores = self.model(x)[:, active_outputs] # self.model(x) if (allowed_classes is None) else 
                _, y_predicted = torch.max(y_scores, 1)
                # Get accuracy
                total_correct += (y_predicted == y).sum().item()
                total_tested += x.size(0)
                # Get loss
                loss_t = self.loss(y_scores, y) 
                total_loss += loss_t

        res['acc_t'] = total_correct / total_tested
        res['loss_t'] = total_loss.item() / (batch_idx + 1)
        self.model.train()
        return res

    def test(self, current_task_id, data_loader, model):
        """ Evaluate given model on dataset.
            Args:
                current_task_id (int): Task id that model has been trained up to.
                data_loader (torch.DataLoader): Dataloader for Dataset to evlauate model on.
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

        res = {}
        with torch.no_grad():
            for batch_idx, (x, y, t) in enumerate(data_loader):
                x, y, t = x.to(device), y.to(device), t.to(device) #--> transfer them to correct device
                # Adjust class range if class-il setting
                if scenario == 'task':
                    active_outputs = torch.arange(classes_per_task, dtype=torch.long, device=device) 
                    active_outputs += (torch.unique(t)*classes_per_task) # shift active outputs to correct task range
                elif scenario == 'class':
                    y = y + (t*classes_per_task) # shift true class labels
                    active_outputs = torch.arange(current_task_id*classes_per_task, dtype=torch.long, device=device)
                # Make predictions
                y_scores = model(x)[:, active_outputs] # if (allowed_classes is None) else 
                _, y_predicted = torch.max(y_scores, 1)
                # Get accuracy
                total_correct += (y_predicted == y).sum().item()
                total_tested += x.size(0)
                # Get loss
                loss_t = self.loss(y_scores, y)
                total_loss += loss_t

        res['acc_t'] = total_correct / total_tested
        res['loss_t'] = total_loss.item() / (batch_idx + 1)
        model.train()
        return res

    ### NOTE: This function could be needed when working with Omniglot, where nubmer of classes per task varies
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
        return active_classes

    ### NOTE: This function could be needed when working with Omniglot, where nubmer of classes per task varies
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
        return active_classes

    def get_replay_data(self, task_id):
        replay_dataset = self.replay_memory.get_memory_for_training(task_id) # get memory points from replay method
        return replay_dataset

    def get_replay_data_from_schedule(self, task_id, replay_schedule):
        replay_dataset = self.replay_memory.get_memory_for_training_with_partition(task_id, replay_schedule) # get memory points from replay schedule
        return replay_dataset

    def update_replay_memory(self, task_id, dataset):
        self.replay_memory.update_memory(task_id, dataset)
    
    def save_checkpoint(self, task_id, folder='./', file_name='model.pth.tar'):
        """ Save checkiṕoint for model and optimizer in trainer to file.
        """
        print("Saving model and optimizer for task {} at {}...\n".format(task_id, os.path.join(folder, file_name)))
        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'task_id': task_id,
                }
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        torch.save(state, os.path.join(folder, file_name))

    def load_checkpoint(self, checkpoint_dir, file_path):
        """ Load checkpoint for model and optimizer to trainer from file.
        """
        # Build model and optimizer
        model = build_models(self.args)
        optimizer = build_optimizers(net=model, args=self.args)
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
        model = build_models(self.args)
        model = model.to(self.device)
        fname = os.path.join(self.checkpoint_dir, file_name)
        #print('Loading checkpoint from file {}...'.format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
