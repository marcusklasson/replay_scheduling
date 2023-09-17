
import os
import time 
import numpy as np
import torch

from training.config import build_models, build_optimizers
from trainer.rs import ReplaySchedulingTrainer
from trainer.rs_coreset_buffer import ReplaySchedulingTrainerCoreset
from trainer.summary import Summarizer
from trainer.utils import get_data_loader 

class HeuristicSchedulingTrainer(ReplaySchedulingTrainer):
    """ Trainer for Heuristic Scheduling. 
    """

    def __init__(self, config):
        super().__init__(config)
        self.val_threshold = config['replay']['val_threshold'] 
        self.heuristic_schedule = config['replay']['schedule'] # type of heuristic
        self.replay_schedule = []
        # NOTE: equal number of samples from each class! How to make this different easily?

    def get_memory_for_training_from_partition(self, task_id, partition):
        """ Get memory samples from memory buffer stored in a dictionary.
            Works as an incremental memory that grabs all stored samples.
            Args:
                task_id (int): Current task id in range {0, 1, ..., n_tasks-1}
            Returns:
                memory (dict): Replay memory samples from tasks.
        """

        if (sum(partition) == 0) and (self.scenario=='task'):
            x_replay, y_replay = None, None 
            return x_replay, y_replay 

        x_replay, y_replay = [], []
        n_samples_parts = self._divide_memory_size_into_parts_based_on_partition(task_id, partition)
        print('n_samples_parts: ', n_samples_parts)
        for t, n_samples_per_slot in enumerate(n_samples_parts): #partition.items():
            if n_samples_per_slot <= 0: # if any task_id should include no samples
                continue
            x_replay_t, y_replay_t = self.select_memory_data_from_task(task=t, n_samples=n_samples_per_slot)
            x_replay.append(x_replay_t) 
            y_replay.append(y_replay_t) 
        return torch.cat(x_replay, dim=0), torch.cat(y_replay, dim=0)

    def train_single_task(self, task_id, train_dataset): 
        """ Train model on single task dataset.

            Args:
                task_id (int): Task identifier (splitMNIST: 1-5).
                train_dataset (torch.Dataset): Training dataset for current task.
                partition (dict): Proportion of samples to grab from each task in each dictionary slot.
        """
        self.model.train()

        # Shorthands
        scenario = self.scenario
        classes_per_task = self.classes_per_task
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        device = self.device
        logger = self.logger
        #n_replays = 0
        self.current_task = task_id

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
                                    rng=torch.Generator().manual_seed(self.seed+task_id))#self.gen_pytorch)
        memory_replay_shuffler = np.random.RandomState(self.seed+task_id)
        # Get replay data from partition
        if task_id > 1 and self.replay_enabled and len(self.replay_schedule)>0:
            partition = self.replay_schedule[self.n_replays]
            print('schedule: ', partition)
            x_replay_from_partition, y_replay_from_partition = self.get_memory_for_training_from_partition(task_id, partition)
            if (x_replay_from_partition is not None) and (self.verbose > 0):
                print('in trainer, len(selected y_replay): ', len(y_replay_from_partition))
                print('in trainer, selected y_replay: ', y_replay_from_partition)
                print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
                print()
            self.n_replays += 1

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
                if (task_id == 1) or (x_replay_from_partition is None):
                    x_replay = y_replay = None   #-> if no replay
                else:
                    x_replay, y_replay = self.get_replay_batch(task_id, 
                                                            x_replay_from_partition, 
                                                            y_replay_from_partition, 
                                                            shuffler=memory_replay_shuffler)
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

                #break
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
        # Update memory buffer and parameters 
        if self.use_episodic_memory:
            self.update_episodic_memory(train_dataset)
            self.episodic_filled_counter += self.memories_per_class * self.classes_per_task
        
        print('End training for task %d...' % task_id)
        self.last_trained_task = task_id
        self.logger.save_stats('stats.p')


    def select_next_replay_schedule(self, task_id, val_accs):
        props = [0.0]*(self.n_tasks-1)
        replay_tasks = []
        task_id = task_id - 1
        current_val_accs = val_accs[task_id, :]
        
        print('Select tasks for replay: ')
        if self.heuristic_schedule in ['heuristic_global_drop', 'heuristic1']:
            # Add replay task based on if current acc is below threshold based on top task acc
            max_val_accs = np.max(val_accs, axis=0)[:task_id+1]
            for t, acc in enumerate(current_val_accs[:task_id+1]):
                threshold = max_val_accs[t] * self.val_threshold
                print('Task {} performance: {:.4f} < {:.4f}*{} = {}'.format(t+1, acc, max_val_accs[t], self.val_threshold, acc <= threshold))
                if acc <= threshold:
                    replay_tasks.append(t)
        elif self.heuristic_schedule in ['heuristic_local_drop', 'heuristic2']:
            # Add replay task based on if current acc is below threshold based on previous task acc
            prev_val_accs = val_accs[task_id-1, :] if task_id>0 else np.zeros(self.n_tasks)
            for t, acc in enumerate(current_val_accs[:task_id+1]):
                threshold = prev_val_accs[t] * self.val_threshold
                print('Task {} performance: {:.4f} < {:.4f}*{} = {}'.format(t+1, acc, prev_val_accs[t], self.val_threshold, acc <= threshold))
                if acc <= threshold:
                    replay_tasks.append(t)
        elif self.heuristic_schedule in ['heuristic_accuracy_threshold', 'heuristic3']:
            # Add replay task based on if current acc is below threshold 
            for t, acc in enumerate(current_val_accs[:task_id+1]):
                #print(t, acc)
                threshold = self.val_threshold
                print('Task {} performance: {:.4f} < {:.4f} = {}'.format(t+1, acc, self.val_threshold, acc <= threshold))
                if acc <= threshold:
                    replay_tasks.append(t)
        else:
            raise ValueError('Heuristic schedue %s does not exist...' %(self.heuristic_schedule))

        # calculate proportion
        if len(replay_tasks) > 0:
            for t in replay_tasks:
                props[t] = 1/len(replay_tasks)
            assert np.isclose(np.sum(props), 1.0), 'proportion values = {} should sum to {}'.format(props, 1)
        # add task proportion to replay schedule
        self.replay_schedule.append(props)

        """
        props = [0.0]*(self.n_tasks-1)
        replay_tasks = []
        max_val_accs = np.max(val_accs, axis=0)[:task_id]
        current_val_accs = val_accs[task_id-1]
        print('Select tasks for replay: ')
        for t, acc in enumerate(current_val_accs[:task_id]):
            threshold = max_val_accs[t] * self.val_threshold
            print('Task {} performance: {:.4f} < {:.4f}*{:.2f} = {}'.format(t+1, acc, max_val_accs[t], self.val_threshold, acc < threshold))
            if acc < threshold:
                replay_tasks.append(t)
        # calculate proportion
        if len(replay_tasks) > 0:
            for t in replay_tasks:
                props[t] = 1/len(replay_tasks)
            assert np.isclose(np.sum(props), 1.0), 'proprotion values = {} should sum to {}'.format(props, 1)
        # add task proportion to replay schedule
        self.replay_schedule.append(props)
        """

class HeuristicSchedulingTrainerCoreset(ReplaySchedulingTrainerCoreset):
    """ Trainer for Heuristic Scheduling. 
    """

    def __init__(self, config):
        super().__init__(config)
        self.val_threshold = config['replay']['val_threshold'] 
        self.heuristic_schedule = config['replay']['schedule'] # type of heuristic
        self.replay_schedule = []

    def train_single_task(self, task_id, train_dataset): 
        """ Train model on single task dataset.

            Args:
                task_id (int): Task identifier (splitMNIST: 1-5).
                train_dataset (torch.Dataset): Training dataset for current task.
                partition (dict): Proportion of samples to grab from each task in each dictionary slot.
        """
        self.model.train()
        # Shorthands
        scenario = self.scenario
        classes_per_task = self.classes_per_task
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        device = self.device
        logger = self.logger
        #n_replays = 0
        self.current_task = task_id

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
                                    rng=torch.Generator().manual_seed(self.seed+task_id))#self.gen_pytorch)
        memory_replay_shuffler = np.random.RandomState(self.seed+task_id)
        # Get replay data from partition
        if task_id > 1 and self.replay_enabled and self.replay_schedule is not None:
            partition = self.replay_schedule[self.n_replays]
            if np.sum(partition) > 0.0:
                x_replay_from_partition, y_replay_from_partition = self.get_memory_for_training_from_partition(task_id, partition)
                if self.verbose > 0:
                    print('in trainer, selected y_replay: ', y_replay_from_partition)
                    print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
                    print()
            else:
                x_replay_from_partition = y_replay_from_partition = None

            self.n_replays += 1
        #labels_in_buffer = [y for (X, y) in self.buffer]
        #print('labels_in_buffer: ', labels_in_buffer)
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
                if (task_id == 1) or (x_replay_from_partition is None):
                    x_replay = y_replay = None   #-> if no replay
                else:
                    x_replay, y_replay = self.get_replay_batch(task_id, 
                                                            x_replay_from_partition, 
                                                            y_replay_from_partition, 
                                                            shuffler=memory_replay_shuffler)
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
        # Update memory buffer and parameters 
        self.update_coreset(train_dataset, task_id-1)
        
        print('End training for task %d...' % task_id)
        self.last_trained_task = task_id
        self.logger.save_stats('stats.p')

    def select_next_replay_schedule(self, task_id, val_accs):
        props = [0.0]*(self.n_tasks-1)
        replay_tasks = []
        #max_val_accs = np.max(val_accs, axis=0)[:task_id]
        #print(max_val_accs)
        task_id = task_id - 1
        current_val_accs = val_accs[task_id, :]
        
        print('Select tasks for replay: ')
        if self.heuristic_schedule in ['heuristic_global_drop', 'heuristic1']:
            # Add replay task based on if current acc is below threshold based on top task acc
            max_val_accs = np.max(val_accs, axis=0)[:task_id+1]
            for t, acc in enumerate(current_val_accs[:task_id+1]):
                threshold = max_val_accs[t] * self.val_threshold
                print('Task {} performance: {:.4f} < {:.4f}*{} = {}'.format(t+1, acc, max_val_accs[t], self.val_threshold, acc <= threshold))
                if acc <= threshold:
                    replay_tasks.append(t)
        elif self.heuristic_schedule in ['heuristic_local_drop', 'heuristic2']:
            # Add replay task based on if current acc is below threshold based on previous task acc
            prev_val_accs = val_accs[task_id-1, :] if task_id>0 else np.zeros(self.n_tasks)
            for t, acc in enumerate(current_val_accs[:task_id+1]):
                threshold = prev_val_accs[t] * self.val_threshold
                print('Task {} performance: {:.4f} < {:.4f}*{} = {}'.format(t+1, acc, prev_val_accs[t], self.val_threshold, acc <= threshold))
                if acc <= threshold:
                    replay_tasks.append(t)
        elif self.heuristic_schedule in ['heuristic_accuracy_threshold', 'heuristic3']:
            # Add replay task based on if current acc is below threshold 
            for t, acc in enumerate(current_val_accs[:task_id+1]):
                #print(t, acc)
                threshold = self.val_threshold
                print('Task {} performance: {:.4f} < {:.4f} = {}'.format(t+1, acc, self.val_threshold, acc <= threshold))
                if acc <= threshold:
                    replay_tasks.append(t)
        else:
            raise ValueError('Heuristic schedue %s does not exist...' %(self.heuristic_schedule))

        # calculate proportion
        if len(replay_tasks) > 0:
            for t in replay_tasks:
                props[t] = 1/len(replay_tasks)
            assert np.isclose(np.sum(props), 1.0), 'proportion values = {} should sum to {}'.format(props, 1)
        # add task proportion to replay schedule
        self.replay_schedule.append(props)
        """
        for t, acc in enumerate(current_val_accs[:task_id]):
            threshold = max_val_accs[t] * self.val_threshold
            print('Task {} performance: {:.4f} < {:.4f}*{:.2f} = {}'.format(t+1, acc, max_val_accs[t], self.val_threshold, acc < threshold))
            if acc < threshold:
                replay_tasks.append(t)
        # calculate proportion
        if len(replay_tasks) > 0:
            for t in replay_tasks:
                props[t] = 1/len(replay_tasks)
            assert np.isclose(np.sum(props), 1.0), 'proprotion values = {} should sum to {}'.format(props, 1)
        # add task proportion to replay schedule
        self.replay_schedule.append(props)
        """