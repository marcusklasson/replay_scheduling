


import os
import time 
import numpy as np
import torch
from torch.nn import functional as F

from trainer.rs import ReplaySchedulingTrainer
from trainer.summary import Summarizer
from trainer.utils import get_data_loader 

class ReplaySchedulingTrainerDER(ReplaySchedulingTrainer):
    """ Trainer for Replay Scheduling. 
    """

    def __init__(self, config):
        super().__init__(config)
        self.episodic_logits = -1e10*torch.ones((self.n_memories, self.n_classes), device=self.device) #torch.FloatTensor(self.n_memories, self.n_classes)
        self.alpha = config['replay']['alpha']
        if 'val_threshold' in config['replay'].keys():
            self.replay_schedule = [] # create list for replay schedule when using heuristic
            self.val_threshold = config['replay']['val_threshold']

    def get_memory_for_training_from_partition(self, task_id, partition):
        if self.episodic_filled_counter <= self.memory_limit:
            return (self.episodic_images[:self.episodic_filled_counter], self.episodic_labels[:self.episodic_filled_counter], 
                    self.episodic_logits[:self.episodic_filled_counter])

        x_replay, y_replay, logits_replay = [], [], []
        n_samples_parts = self._divide_memory_size_into_parts_based_on_partition(task_id, partition)
        #print(n_samples_parts)
        for t, n_samples_per_slot in enumerate(n_samples_parts): #partition.items():
            if n_samples_per_slot <= 0: # if any task_id should include no samples
                continue
            x_replay_t, y_replay_t, logits_replay_t = self.select_memory_data_from_task(task=t, n_samples=n_samples_per_slot)
            #print('mean of replay data: {:.5f}'.format(torch.mean(x_replay_t).item()))
            x_replay.append(x_replay_t) 
            y_replay.append(y_replay_t) 
            logits_replay.append(logits_replay_t)
        return torch.cat(x_replay, dim=0), torch.cat(y_replay, dim=0), torch.cat(logits_replay, dim=0),

    def select_memory_data_from_task(self, task, n_samples):
        """ task = {0, ..., T-1}
        """
        classes_per_task = self.classes_per_task
        offset = task * self.memories_per_class * classes_per_task
        #mem_index = list(range(offset, offset+(self.memories_per_class*classes_per_task)))
        
        ## NOTE: some list manipulation to balance the classes during sample selection
        mem_index = np.arange(offset, offset+(self.memories_per_class*classes_per_task))
        tmp = np.reshape(mem_index, (classes_per_task, self.memories_per_class))
        mem_index = list((tmp.T).flatten())
        
        r_images, r_labels, r_logits = [], [], []

        class_samples = {}
        # NOTE: using np.ceil and then exact_num_points = num_points fills up the memory as equally as it can
        # NOTE: using np.floor and then num_points_per_class*classes_per_task always returns equal number/class but don't fill up memory
        n_samples_per_class = int(np.ceil(n_samples/classes_per_task)) #int(np.ceil(num_points/classes_per_task))
        #print('n_samples_per_class: ', n_samples_per_class)
        #exact_n_samples = num_points #num_points_per_class*classes_per_task

        n_selected_samples = 0
        for idx in range(len(mem_index)):
            #print('mem_index[idx]: ', mem_index[idx])
            data = self.episodic_images[mem_index[idx], :]
            label = self.episodic_labels[mem_index[idx]]
            logits = self.episodic_logits[mem_index[idx], :]
            cid = label.item() if isinstance(label, torch.Tensor) else label
            if cid in class_samples:
                if len(class_samples[cid]) < n_samples_per_class: # we want even number of classes here
                    class_samples[cid].append((data, logits))
                    n_selected_samples += 1
                    #print('picked mem_index: ', mem_index[idx])
            else:
                class_samples[cid] = [(data, logits)]
                n_selected_samples += 1
                #print('picked mem_index: ', mem_index[idx])
            if n_selected_samples >= n_samples:
                break
        for cid in class_samples.keys(): #range(num_classes):
            imgs = [x[0] for x in class_samples[cid]]
            logs = [x[1] for x in class_samples[cid]]

            r_images.append(torch.stack(imgs, dim=0))
            r_logits.append(torch.stack(logs, dim=0))
            r_labels.append(torch.ones(len(class_samples[cid]), dtype=torch.long, device=self.device)*cid)
        #print('r_labels: ', r_labels)
        return torch.cat(r_images, dim=0), torch.cat(r_labels, dim=0), torch.cat(r_logits, dim=0)

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
        print('replay schedule! - ', self.replay_schedule)
        print(self.replay_enabled)
        if task_id > 1:
            print(self.replay_schedule[self.n_replays])

        if task_id > 1 and (self.replay_enabled) and (self.replay_schedule is not None):
            partition = self.replay_schedule[self.n_replays]
            if sum(partition) > 0.0: # check if any task has bee nselected to be replayed
                replay_data_from_partition = self.get_memory_for_training_from_partition(task_id, partition)
                x_replay_from_partition = replay_data_from_partition[0]
                y_replay_from_partition = replay_data_from_partition[1] 
                logits_replay_from_partition = replay_data_from_partition[2]
                if self.verbose > 0:
                    print('in trainer, len(selected y_replay): ', len(y_replay_from_partition))
                    print('in trainer, selected y_replay: ', y_replay_from_partition)
                    #print('in trainer, logits_replay: ', logits_replay_from_partition)
                    print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
                    print()
            else: 
                x_replay_from_partition, y_replay_from_partition, logits_replay_from_partition = None, None, None
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
                    x_replay = y_replay = logits_replay = None   #-> if no replay
                else:
                    x_replay, y_replay, logits_replay = self.get_replay_batch(task_id, 
                                                            x_replay_from_partition, 
                                                            y_replay_from_partition, 
                                                            logits_replay_from_partition,
                                                            shuffler=memory_replay_shuffler)
                # Train the main model with this batch
                loss_dict = self.train_batch(x_curr, y_curr, 
                                            x_=x_replay, y_=y_replay, logits_=logits_replay,
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

    def train_batch(self, x, y, x_=None, y_=None, logits_=None, active_classes=None, task=1):
        self.model.train()
        self.optimizer.zero_grad()
        # Shorthands
        classes_per_task = self.classes_per_task
        scenario = self.scenario

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

        if x_ is not None:
            # if Task-IL, find active output indices on model for replay batch
            task_ids = torch.floor(y_ / self.classes_per_task).long()
            if scenario == 'task':
                active_indices = torch.arange(self.classes_per_task, dtype=torch.long, device=self.device)
                active_indices = active_indices.repeat(len(task_ids), 1)
                active_indices = active_indices + (task_ids*classes_per_task).unsqueeze(1)
                y_ = y_ - (task_ids*classes_per_task)
                logits_ = logits_.gather(1, active_indices) # get the correct indices accorind to task

            # Compute replay data
            logits_replay = self.model(x_).gather(1, active_indices) if scenario == 'task' else self.model(x_)
            loss_replay = self.alpha * F.mse_loss(logits_replay, logits_) 
            loss += loss_replay # add to current loss
            acc_replay = (y_ == logits_replay.max(1)[1]).sum().item() / x_.size(0)
            
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

    def get_replay_batch(self, task, x_replay_from_partition, y_replay_from_partition, logits_replay_from_partition, shuffler):
        if len(x_replay_from_partition) <= self.batch_size:
            mem_indices = np.arange(len(x_replay_from_partition)) 
        else:
            mem_indices = shuffler.choice(len(x_replay_from_partition), size=self.batch_size, replace=False) 
        x_replay = x_replay_from_partition[mem_indices]
        y_replay = y_replay_from_partition[mem_indices]
        logits_replay = logits_replay_from_partition[mem_indices]
        return x_replay, y_replay, logits_replay

    def update_episodic_memory(self, dataset):
        X = np.stack([img.numpy() for img, _ in dataset], axis=0)
        y = np.stack([label for _, label in dataset], axis=0)
        for y_ in np.unique(y):
            er_x = X[y == y_] 
            er_y = y[y == y_]
            chosen_inds = self.summarizer.build_summary(er_x, er_y, self.memories_per_class, method=self.sample_selection,
                                                        model=self.model, device=self.device)
            #print(chosen_inds)
            if self.scenario == 'domain':
                with_in_task_offset = self.memories_per_class * y_ + self.episodic_filled_counter
            else:
                with_in_task_offset = self.memories_per_class * y_
            mem_index = list(range(with_in_task_offset, with_in_task_offset+self.memories_per_class))
            #print('mem_index: ', mem_index)
            self.episodic_images[mem_index] = torch.from_numpy(er_x[chosen_inds]).to(self.device)
            self.episodic_labels[mem_index] = torch.from_numpy(er_y[chosen_inds]).to(self.device)
            # Compute logits from selected images and store them
            with torch.no_grad():
                er_logits = self.model(torch.from_numpy(er_x[chosen_inds]).to(self.device))
            self.episodic_logits[mem_index] = er_logits 

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


class ReplaySchedulingTrainerDERPP(ReplaySchedulingTrainerDER):

    def __init__(self, config):
        super().__init__(config)
        self.beta = config['replay']['beta']
    
    def train_batch(self, x, y, x_=None, y_=None, logits_=None, active_classes=None, task=1):
        self.model.train()
        self.optimizer.zero_grad()
        # Shorthands
        classes_per_task = self.classes_per_task
        scenario = self.scenario

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

        if x_ is not None:
            # if Task-IL, find active output indices on model for replay batch
            task_ids = torch.floor(y_ / self.classes_per_task).long()
            if scenario == 'task':
                active_indices = torch.arange(self.classes_per_task, dtype=torch.long, device=self.device)
                active_indices = active_indices.repeat(len(task_ids), 1)
                active_indices = active_indices + (task_ids*classes_per_task).unsqueeze(1)
                y_ = y_ - (task_ids*classes_per_task)
                logits_ = logits_.gather(1, active_indices) # get the correct indices accorind to task

            # Compute replay data
            logits_replay = self.model(x_).gather(1, active_indices) if scenario == 'task' else self.model(x_)
            loss_replay1 = self.alpha * F.mse_loss(logits_replay, logits_) 
            loss += loss_replay1 # add to current loss

            
            loss_replay2 = self.beta * self.criterion(logits_replay, y_)
            loss += loss_replay2 # add to current loss
            acc_replay = (y_ == logits_replay.max(1)[1]).sum().item() / x_.size(0)

            loss_replay = loss_replay1 + loss_replay2 # sum of regularizers
            
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