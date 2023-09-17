
import os
import time 
import numpy as np
import torch
from torch.nn import functional as F

from trainer.rs import ReplaySchedulingTrainer
from trainer.summary import Summarizer
from trainer.utils import get_data_loader 

class ReplaySchedulingTrainerMER(ReplaySchedulingTrainer):
    """ Trainer for Replay Scheduling for Meta ER.
    """

    def __init__(self, config):
        super().__init__(config)
        self.beta = config['replay']['beta'] # default: 1.0
        self.gamma = config['replay']['gamma'] # default: 1.0

        if 'val_threshold' in config['replay'].keys():
            self.replay_schedule = [] # create list for replay schedule when using heuristic
            self.val_threshold = config['replay']['val_threshold']

    def draw_batches(self, x, y, x_, y_, shuffler):
        batches = []
        for i in range(1):
            if x_ is not None:
                if len(x_) <= self.batch_size:
                    mem_indices = np.arange(len(x_)) 
                else:
                    mem_indices = shuffler.choice(len(x_), size=self.batch_size, replace=False) 
                x_buf, y_buf = x_[mem_indices], y_[mem_indices]
                # shift the labels for the replay samples
                task_ids = torch.floor(y_buf / self.classes_per_task).long()
                if self.scenario == 'task':
                    active_indices = torch.arange(self.classes_per_task, dtype=torch.long, device=self.device)
                    active_indices = active_indices.repeat(len(task_ids), 1)
                    active_indices = active_indices + (task_ids*self.classes_per_task).unsqueeze(1)
                    y_buf = y_buf - (task_ids*self.classes_per_task)
                inputs = torch.cat((x_buf, x))
                labels = torch.cat((y_buf, y))

                batches.append((inputs, labels))
            else:
                #print(y.size())
                batches.append((x, y))
        return batches

    def train_batch(self, x, y, x_=None, y_=None, active_classes=None, task=1):

        memory_replay_shuffler = np.random.RandomState(self.seed+task) # shuffler for grabbing memory samples
        self.model.train()
        self.optimizer.zero_grad()
        # Shorthands
        classes_per_task = self.classes_per_task
        scenario = self.scenario

        batches = self.draw_batches(x, y, x_, y_, shuffler=memory_replay_shuffler)
        theta_A0 = self.model.get_params().data.clone() 

        for i in range(1):
            theta_Wi0 = self.model.get_params().data.clone()

            batch_inputs, batch_labels = batches[i]
            class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
            # get replay samples for computing task identities, had to do this here as well
            if x_ is not None:
                r_bsize = len(x_) if len(x_) <= self.batch_size else self.batch_size 
                r_images, r_labels = batch_inputs[:r_bsize], batch_labels[:r_bsize]
                
                task_ids = torch.floor(r_labels / classes_per_task).long()
                if scenario == 'task':
                    active_indices = torch.arange(classes_per_task, dtype=torch.long, device=self.device)
                    active_indices = active_indices.repeat(len(task_ids), 1)
                    active_indices = active_indices + (task_ids*classes_per_task).unsqueeze(1)
                    r_labels = r_labels - (task_ids*classes_per_task)
                
                if scenario == 'task':
                    class_entries = torch.tensor(class_entries, device=self.device).repeat(len(y), 1)
                    active_indices = torch.cat([active_indices, class_entries], dim=0)
                output = self.model(batch_inputs).gather(1, active_indices) if scenario == 'task' else self.model(batch_inputs)

            else:
                output = self.model(batch_inputs)[:, class_entries] if scenario == 'task' else self.model(batch_inputs)

            # Compute loss for both current and replay data
            loss = self.criterion(output, batch_labels) 
            accuracy = (batch_labels == output.max(1)[1]).sum().item() / batch_inputs.size(0)
            # Compute gradients and take step
            loss.backward()
            self.optimizer.step()

            # within batch reptile meta-update
            new_params = theta_Wi0 + self.beta * (self.model.get_params() - theta_Wi0)
            self.model.set_params(new_params)
        # across batch reptile meta-update
        new_new_params = theta_A0 + self.gamma * (self.model.get_params() - theta_A0)
        self.model.set_params(new_new_params)

        # Calculate total replay loss
        loss_replay = None #if (x_ is None) else loss_replay 
        acc_replay = None# if (x_ is None) else acc_replay 
        loss_curr = loss 
        # Return the dictionary with different training-loss split in categories
        return {
            'loss_current': loss_curr.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'accuracy': accuracy if accuracy is not None else 0.,
            'accuracy_replay': acc_replay if acc_replay is not None else 0.,
        }

    def train_single_task(self, task_id, train_dataset): 
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
            if sum(partition) > 0.0:
                x_replay_from_partition, y_replay_from_partition = self.get_memory_for_training_from_partition(task_id, partition)
                #assert len(y_replay_from_partition) == self.memory_limit or len(y_replay_from_partition) == self.episodic_filled_counter
                if self.verbose > 0:
                    print('in trainer, len(selected y_replay): ', len(y_replay_from_partition))
                    print('in trainer, selected y_replay: ', y_replay_from_partition)
                    print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
                    print()
            else:
                x_replay_from_partition, y_replay_from_partition = None, None
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
                if (task_id == 1) or (self.replay_enabled==False) or (x_replay_from_partition is None):
                    x_replay = y_replay = None   #-> if no replay
                else:
                    x_replay, y_replay = x_replay_from_partition, y_replay_from_partition

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

    def select_memory_data_from_task(self, task, n_samples):
        """ task = {0, ..., T-1}
        """
        classes_per_task = self.classes_per_task
        offset = task * self.memories_per_class * classes_per_task
        #mem_index = list(range(offset, offset+(self.memories_per_class*classes_per_task)))
        #print(mem_index)
        ## NOTE: some list manipulation to balance the classes during sample selection
        mem_index = np.arange(offset, offset+(self.memories_per_class*classes_per_task))
        tmp = np.reshape(mem_index, (classes_per_task, self.memories_per_class))
        mem_index = list((tmp.T).flatten())
        #print('mem_index: ', mem_index)

        r_images, r_labels = [], []

        class_samples = {}
        # NOTE: using np.ceil and then exact_num_points = num_points fills up the memory as equally as it can
        # NOTE: using np.floor and then num_points_per_class*classes_per_task always returns equal number/class but don't fill up memory
        n_samples_per_class = int(np.ceil(n_samples/classes_per_task)) #int(np.ceil(num_points/classes_per_task))

        n_selected_samples = 0
        for idx in range(len(mem_index)):
            #print('mem_index[idx]: ', mem_index[idx])
            data = self.episodic_images[mem_index[idx], :]
            label = self.episodic_labels[mem_index[idx]]
            cid = label.item() if isinstance(label, torch.Tensor) else label
            if cid in class_samples:
                if len(class_samples[cid]) < n_samples_per_class: # we want even number of classes here
                    class_samples[cid].append(data)
                    n_selected_samples += 1
                    #print('picked mem_index: ', mem_index[idx])
            else:
                class_samples[cid] = [data]
                n_selected_samples += 1
                #print('picked mem_index: ', mem_index[idx])
            if n_selected_samples >= n_samples:
                break
        for cid in class_samples.keys(): #range(num_classes):
            r_images.append(torch.stack(class_samples[cid], dim=0))
            r_labels.append(torch.ones(len(class_samples[cid]), dtype=torch.long, device=self.device)*cid)
        #print('r_labels: ', r_labels)
        return torch.cat(r_images, dim=0), torch.cat(r_labels, dim=0)

    def select_next_replay_schedule(self, task_id, val_accs):
        # NOTE: for heuristic scheduling!
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