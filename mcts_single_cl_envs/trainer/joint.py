
import time 
import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.base import Trainer
from trainer.summary import Summarizer
from trainer.utils import get_data_loader

class JointTrainer(Trainer):
    """ Trainer for Joint training setting. 
    """

    def __init__(self, config):
        super().__init__(config)
        
    def train_single_task(self, task_id, train_dataset, valid_datasets=None): 
        """ Train model on single task dataset.

            Args:
                task_id (int): Task identifier (splitMNIST: 1-5).
                train_dataset (torch.Dataset): Training dataset for current task.
                replay_dataset (dict with torch.Tensor): Replay data from previous tasks in each dictionary slot.
        """
        self.model.train()

        # Shorthands
        scenario = self.scenario
        classes_per_task = self.classes_per_task
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        device = self.device
        logger = self.logger

        # Reset optimizer before every task in ER
        self.optimizer = self.prepare_optimizer()

        active_classes = self._get_active_classes_up_to_task_id(task_id)
        #print('active classes: ', active_classes)
        data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

        t0 = time.time()
        #print('self.episodic_labels: ', self.episodic_labels)
        for epoch in range(n_epochs):
            loss_curr = 0.0
            loss_replay = 0.0
            acc, acc_replay = 0.0, 0.0
            for batch_idx, (x_, y_) in enumerate(data_loader):
                #-----------------Collect data------------------#
                x_, y_ = x_.to(device), y_.to(device) #--> transfer them to correct device

                # Train the main model with this batch
                loss_dict = self.train_batch(x=None, y=None, 
                                            x_=x_, y_=y_,
                                            active_classes=active_classes, 
                                            task=task_id)
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

    def train_batch(self, x, y, x_=None, y_=None, active_classes=None, task=1):
            """ Train model on single batch of current task samples
                and (optionally) replay samples for the Task Incremental
                Learning setting.
                Args:
                    x (torch.Tensor): Input data from current task. 
                    y (torch.LongTensor): Labels from current task.
                    x_ (dict with torch.Tensor): Input data from replay tasks. 
                    y_ (dict with torch.LongTensor): Labels from replay_tasks.
                    active_classes (list): Active classes for each task, (ex: [[0, 1], [2, 3], [4,5], ...])
                    task (int): Task id number starting from 1, e.g. splitMNIST: 1-5
                Returns:
                    loss_dict (dict): Dictionary with loss and accuracy metrics.
            """
            self.model.train()
            self.optimizer.zero_grad()
            # Shorthands
            classes_per_task = self.classes_per_task
            scenario = self.scenario

            if x_ is not None:
                # if Task-IL, find active output indices on model for replay batch
                task_ids = torch.floor(y_ / self.classes_per_task).long()
                if scenario == 'task':
                    active_indices = torch.arange(self.classes_per_task, dtype=torch.long, device=self.device)
                    active_indices = active_indices.repeat(len(task_ids), 1)
                    active_indices = active_indices + (task_ids*classes_per_task).unsqueeze(1)
                    #y_hat = y_hat_replay.gather(1, active_indices)
                    y_ = y_ - (task_ids*classes_per_task)

                #print('y_: ', y_)

                # Compute replay data
                y_logits_ = self.model(x_).gather(1, active_indices) if scenario == 'task' else self.model(x_)
                loss = self.criterion(y_logits_, y_) 
                accuracy = (y_ == y_logits_.max(1)[1]).sum().item() / x_.size(0)

            # Compute gradients for batch with current task data
            loss.backward()
            # Take optimization-step
            self.optimizer.step()

            # Calculate total replay loss
            loss_replay = None #if (x_ is None) else loss_replay #sum(loss_replay) / len(y_)
            acc_replay = None #if (x_ is None) else acc_replay #sum(acc_replay) / len(acc_replay)

            # Return the dictionary with different training-loss split in categories
            return {
                'loss_current': loss.item() if x is not None else 0,
                'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
                'accuracy': accuracy if accuracy is not None else 0.,
                'accuracy_replay': acc_replay if acc_replay is not None else 0.,
            }

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

        data_loader = get_data_loader(dataset,
                                    batch_size=100,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=False)
        res = {}
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(device), y.to(device) 
                # -break on [test_size] (if "None", full dataset is used)
                if test_size:
                    if total_tested >= test_size:
                        break

                task_ids = torch.floor(y / self.classes_per_task).long()
                if scenario == 'task':
                    active_indices = torch.arange(self.classes_per_task, dtype=torch.long, device=self.device)
                    active_indices = active_indices.repeat(len(task_ids), 1)
                    active_indices = active_indices + (task_ids*self.classes_per_task).unsqueeze(1)
                    #y_hat = y_hat_replay.gather(1, active_indices)
                    y = y - (task_ids*classes_per_task)
                  
                # Compute replay data
                y_logits = self.model(x).gather(1, active_indices) if scenario == 'task' else self.model(x)
                # Get loss
                loss_t = self.criterion(y_logits, y) 
                total_loss += loss_t
                # Get accuracy
                total_correct += (y == y_logits.max(1)[1]).sum().item()
                total_tested += x.size(0)

        res['acc_t'] = total_correct / total_tested
        res['loss_t'] = total_loss.item() / (batch_idx + 1)
        self.model.train()
        return res