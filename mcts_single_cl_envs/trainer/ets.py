
import time 
import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.base import Trainer
from trainer.summary import Summarizer

class EqualTaskScheduling(Trainer):
    """ Trainer for Equal Task Scheduling. 
    """

    def __init__(self, config):
        super().__init__(config)
        self.replay_selection = config['replay']['selection']
        self.summarizer = Summarizer.factory(type='uniform', rs=np.random.RandomState(self.seed))
        self.memory_limit = config['replay']['memory_limit'] # used in replay scheduling methods

    def get_memory_for_replay(self, task_id):
    """ Get memory samples from memory buffer stored in a dictionary.
        Works as an incremental memory that grabs all stored samples.
        Args:
            task_id (int): Current task id in range {0, 1, ..., n_tasks-1}
        Returns:
            memory (dict): Replay memory samples from tasks.
    """
    memory = {} 

    if self.replay_method == 'equal':
        n_samples_parts = self._divide_memory_size_into_equal_parts(task_id)
        #print('n_samples_parts: ', n_samples_parts)
        for t, n_samples_per_slot in enumerate(n_samples_parts): #partition.items():
            if n_samples_per_slot <= 0: # if any task_id should include no samples
                continue
            task_data = self._get_memory_points_from_buffer(t, n_samples_per_slot)
            memory[t] = task_data
    elif self.replay_method == 'concat':
        for t in range(task_id):
            memory[t] = [self.X_b[t], self.y_b[t]]
    return memory

    def divide_memory_size_into_equal_parts(self, div, shuffle=True):
        M = self.memory_size
        parts = [M // div + (1 if x < (M % div) else 0)  for x in range(div)]
        parts = np.array(parts)
        if shuffle:
            np.random.shuffle(parts)
        return parts
        
    def train_single_task(self, task_id, train_dataset): 
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

        # Reset optimizer before every task in ER
        self.optimizer = self.prepare_optimizer()

        active_classes = self._get_active_classes_up_to_task_id(task_id)
        #print('active classes: ', active_classes)
        data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)
        
        # sample data for replay 
        if task_id > 1:
            print('sample replay data')
            # NOTE: here we need to grab memory points to use for replay! 

        t0 = time.time()
        #print('self.episodic_labels: ', self.episodic_labels)
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

                ### Get Replay Batch
                if task_id == 1:
                    x_replay = y_replay = None # no replay
                else:
                    if self.memory_limit <= self.batch_size: 
                        mem_indices = torch.arange(self.memory_limit, dtype=torch.long, device=self.device)
                    else:
                        # Sample a random subset from episodic memory buffer
                        mem_indices = torch.randperm(self.memory_limit, dtype=torch.long, device=self.device)[:self.batch_size]
                    x_replay = None #self.episodic_images[mem_indices]
                    y_replay = None #self.episodic_labels[mem_indices]
                    #print(len(y_replay))

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
            
            # Update memory buffer and parameters 
            if self.use_episodic_memory:
                X = np.stack([img.numpy() for img, _ in train_dataset], axis=0)
                y = np.stack([label for _, label in train_dataset], axis=0)
                for y_ in np.unique(y):
                    er_x = X[y == y_] 
                    er_y = y[y == y_]
                    chosen_inds = self.summarizer.build_summary(er_x, er_y, self.memories_per_class, method='uniform')
                    #print('chosen_inds: ', chosen_inds)
                    with_in_task_offset = self.memories_per_class * y_
                    mem_index = list(range(with_in_task_offset, with_in_task_offset+self.memories_per_class))
                    #print('mem_index: ', mem_index)
                    self.episodic_images[mem_index] = torch.from_numpy(er_x[chosen_inds]).to(self.device)
                    self.episodic_labels[mem_index] = torch.from_numpy(er_y[chosen_inds]).to(self.device)
                 
                #print(self.episodic_labels)

                self.episodic_filled_counter += self.memories_per_class * len(np.unique(y)) 

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

                # Compute replay data
                y_logits_replay = self.model(x_).gather(1, active_indices) if scenario == 'task' else self.model(x_)
                loss_replay = self.criterion(y_logits_replay, y_) 
                acc_replay = (y_ == y_logits_replay.max(1)[1]).sum().item() / x_.size(0)

                # Compute current data
                #if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                
                y_logits = self.model(x)[:, class_entries]
                loss_curr = self.criterion(y_logits, y) 
                accuracy = (y == y_logits.max(1)[1]).sum().item() / x.size(0)

                # Compute loss for both current and replay data
                class_entries = torch.tensor(class_entries, device=self.device).repeat(len(y), 1)
                #print(class_entries)
                active_indices = torch.cat([class_entries, active_indices], dim=0)
                x = torch.cat([x, x_], dim=0)
                y = torch.cat([y, y_], dim=0)
                y_logits = self.model(x).gather(1, active_indices)
                loss = self.criterion(y_logits, y) 
                accuracy = (y == y_logits.max(1)[1]).sum().item() / x.size(0)

            else:
                # Run model on current task data
                y_hat = self.model(x)
                # -if needed, remove predictions for classes not in current task
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                    y_hat = y_hat[:, class_entries]
                # prediction loss
                loss = self.criterion(y_hat, y) 
                loss_curr = loss
                # Calculate training acc
                accuracy = (y == y_hat.max(1)[1]).sum().item() / x.size(0)
            
            # Compute gradients for batch with current task data
            loss.backward()
            # Take optimization-step
            self.optimizer.step()

            # Calculate total replay loss
            loss_replay = None if (x_ is None) else loss_replay #sum(loss_replay) / len(y_)
            acc_replay = None if (x_ is None) else acc_replay #sum(acc_replay) / len(acc_replay)

            # Return the dictionary with different training-loss split in categories
            return {
                'loss_current': loss_curr.item() if x is not None else 0,
                'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
                'accuracy': accuracy if accuracy is not None else 0.,
                'accuracy_replay': acc_replay if acc_replay is not None else 0.,
            }

