import os
import time 
import numpy as np
import torch

import tracemalloc

from trainer.summary import Summarizer
from trainer.utils import get_data_loader 
from trainer.base import Trainer

class ReplaySchedulingTrainer(Trainer):
    # Trainer for Replay Scheduling with memory similar to Coresets

    def __init__(self, config):
        super().__init__(config)
        # coreset buffer
        self.buffer = []
        self.buffer_size = config['replay']['memory_size'] # self.memories_per_class * self.classes_per_task * self.n_tasks 
        self.memory_limit = config['replay']['memory_size']
        self.sample_selection = config['replay']['sample_selection']
        self.summarizer = Summarizer.factory(type=self.sample_selection, rs=np.random.RandomState(self.seed))
        self.load_checkpoints = config['load_checkpoints']

        # episodic memory
        self.memories_per_class = config['replay']['examples_per_class']
        self.n_classes = self.classes_per_task * self.n_tasks
        self.n_memories = self.n_classes * self.memories_per_class
        self.count_cls = torch.zeros(self.n_classes, dtype=torch.long)
        img_size = config['data']['img_size']
        in_channel = config['data']['in_channel']
        self.episodic_images = torch.FloatTensor(self.n_memories, in_channel, img_size, img_size)
        self.episodic_labels = -torch.ones(self.n_memories, dtype=torch.long) #torch.LongTensor(self.n_memories)
        self.episodic_filled_counter = 0 # used for ring buffer
        self.examples_seen_so_far = 0 # used for reservoir sampling
        # Add tensors to gpu or cpu
        self.episodic_images = self.episodic_images.to(self.device)
        self.episodic_labels = self.episodic_labels.to(self.device)

    def get_memory_for_training_from_partition(self, task_id, partition):
        # Get replay memory from the history given the task proportions (partition)
        x_replay, y_replay, t_replay = [], [], []
        n_samples_parts = self._divide_memory_size_into_parts_based_on_partition(task_id, partition)
        for t, n_samples_per_slot in enumerate(n_samples_parts): 
            if n_samples_per_slot <= 0: # if any task_id should include no samples
                continue
            x_replay_t, y_replay_t, t_replay_t = self.select_memory_data_from_task(task=t, n_samples=n_samples_per_slot)
            x_replay.append(x_replay_t) 
            y_replay.append(y_replay_t) 
            t_replay.append(t_replay_t)
        return torch.cat(x_replay, dim=0), torch.cat(y_replay, dim=0), torch.cat(t_replay, dim=0)
    
    def select_memory_data_from_task(self, task, n_samples):
        # select n_samples of replay samples from given task 
        (X, y) = self.buffer[task] # get data from coreset buffer
        mem_index = list(range(len(y)))
        r_images, r_labels, r_tasks = [], [], []
        class_samples = {}
        n_selected_samples = 0

        for idx in range(len(mem_index)):
            data = X[mem_index[idx], :]
            label = y[mem_index[idx]]
            cid = label.item() if isinstance(label, torch.Tensor) else label
            if cid in class_samples:
                class_samples[cid].append(data)
                n_selected_samples += 1
            else:
                class_samples[cid] = [data]
                n_selected_samples += 1
            if n_selected_samples >= n_samples:
                break
        for cid in class_samples.keys(): 
            r_images.append(torch.stack(class_samples[cid], dim=0))
            r_labels.append(torch.ones(len(class_samples[cid]), dtype=torch.long, device=self.device)*cid)
            r_tasks.append(torch.ones(len(class_samples[cid]), dtype=torch.long, device=self.device)*task)
        return torch.cat(r_images, dim=0), torch.cat(r_labels, dim=0), torch.cat(r_tasks, dim=0)

    def _divide_memory_size_into_parts_based_on_partition(self, task_id, partition):
        # get the number of samples per task given the task proportion (partition)
        M = self.memory_limit
        x = [i*M for i in partition]
        n_samples_per_slot = [int(i) for i in x]
        samples_left = M - sum(n_samples_per_slot)
        if samples_left > 0:
            frac_part, _ = np.modf(x)
            # Add samples to slot with largest fraction part
            # If fraction parts are equal, add sample to oldest task first until no samples left to give
            indices = (-frac_part).argsort() # get indices of the largest fraction number
            indices = indices[:task_id-1]
            for idx in indices:
                n_samples_per_slot[idx] += 1
                samples_left -= 1
                if samples_left == 0:
                    break
        return n_samples_per_slot

    def update_coreset(self, dataset, t):
        # update coreset-like memory buffer
        size_per_task = self.memory_limit
        # Get new data
        X = np.stack([img.numpy() for img, _ in dataset], axis=0)
        y = np.stack([label for _, label in dataset], axis=0)
        chosen_inds = self.summarizer.build_summary(X, y, size_per_task, method=self.sample_selection,
                                                        model=self.model, device=self.device)

        #print('chosen_inds: ', chosen_inds)
        X, y = X[chosen_inds], y[chosen_inds]
        assert (X.shape[0] == size_per_task)
        self.buffer.append((torch.from_numpy(X).to(self.device), torch.from_numpy(y).to(self.device)) )

    """
    ### EPISODIC MEMORY 
    def select_memory_data_from_task(self, task, n_samples):
        # select n_samples of replay samples from given task 
        classes_per_task = self.classes_per_task
        offset = task * self.memories_per_class * classes_per_task
        #mem_index = list(range(offset, offset+(self.memories_per_class*classes_per_task)))
        ## NOTE: some list manipulation to balance the classes during sample selection
        mem_index = np.arange(offset, offset+(self.memories_per_class*classes_per_task))
        tmp = np.reshape(mem_index, (classes_per_task, self.memories_per_class))
        mem_index = list((tmp.T).flatten())
        r_images, r_labels, r_tasks = [], [], []
        class_samples = {}
        n_selected_samples = 0
        n_samples_per_class = int(np.ceil(n_samples/classes_per_task))

        for idx in range(len(mem_index)):
            data = self.episodic_images[mem_index[idx], :]
            label = self.episodic_labels[mem_index[idx]]
            cid = label.item() if isinstance(label, torch.Tensor) else label
            if cid in class_samples:
                if len(class_samples[cid]) < n_samples_per_class: # we want even number of classes here
                    class_samples[cid].append(data)
                    n_selected_samples += 1
            else:
                class_samples[cid] = [data]
                n_selected_samples += 1
            if n_selected_samples >= n_samples:
                break
        for cid in class_samples.keys(): 
            r_images.append(torch.stack(class_samples[cid], dim=0))
            r_labels.append(torch.ones(len(class_samples[cid]), dtype=torch.long, device=self.device)*cid)
            r_tasks.append(torch.ones(len(class_samples[cid]), dtype=torch.long, device=self.device)*task)
        return torch.cat(r_images, dim=0), torch.cat(r_labels, dim=0), torch.cat(r_tasks, dim=0)

    def _divide_memory_size_into_parts_based_on_partition(self, task_id, partition):

        M = self.memory_limit
        mem_samples_per_task = self.classes_per_task * self.memories_per_class
        x = [i*M for i in partition]
        n_samples_per_slot = [int(i) for i in x]
        for i in range(len(n_samples_per_slot)):
            if n_samples_per_slot[i] > mem_samples_per_task:
                n_samples_per_slot[i] = mem_samples_per_task
        samples_left = M - sum(n_samples_per_slot)
        if samples_left > 0:
            #print(x)
            frac_part, _ = np.modf(x[:task_id]) # only need to take fraction of samples in history
            # Add samples to slot with largest fraction part
            # If fraction parts are equal, add sample to oldest task first until no samples left to give
            indices = (-frac_part).argsort() # get indices of the largest fraction number
            indices = indices[:task_id]
            idx = 0
            while samples_left != 0:
                if n_samples_per_slot[indices[idx]] < mem_samples_per_task:
                    n_samples_per_slot[indices[idx]] += 1
                    samples_left -= 1
                if idx+1 < len(indices):
                    idx += 1
                else:
                    idx = 0
        return n_samples_per_slot
    
    def update_coreset(self, dataset, t):
        X = np.stack([img.numpy() for img, _, _ in dataset], axis=0)
        y = np.stack([label for _, label, _ in dataset], axis=0)
        for y_ in np.unique(y):
            er_x = X[y == y_] 
            er_y = y[y == y_]
            chosen_inds = self.summarizer.build_summary(er_x, er_y, self.memories_per_class, method=self.sample_selection,
                                                        model=self.model, device=self.device)
            if self.scenario == 'domain':
                with_in_task_offset = self.memories_per_class * y_ + self.episodic_filled_counter
            else:
                with_in_task_offset = self.memories_per_class * y_
            mem_index = list(range(with_in_task_offset, with_in_task_offset+self.memories_per_class))
            #print('mem_index: ', mem_index)
            self.episodic_images[mem_index] = torch.from_numpy(er_x[chosen_inds]).to(self.device)
            self.episodic_labels[mem_index] = torch.from_numpy(er_y[chosen_inds]).to(self.device)
        self.episodic_filled_counter += self.memories_per_class * self.classes_per_task
    """

    ### TRAINING
    def train_single_task(self, task_id, train_dataset, task_proportions=None): 
        # Train model on single task with train_dataset
        self.model.train()
        # Shorthands
        scenario = self.scenario
        classes_per_task = self.classes_per_task
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        device = self.device
        logger = self.logger
        self.current_task = task_id

        #active_classes = self._get_active_classes_up_to_task_id(task_id)
        data_loader = get_data_loader(train_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=True,
                                    rng=torch.Generator().manual_seed(self.seed+task_id))
        memory_replay_shuffler = np.random.RandomState(self.seed+task_id)
        # Get replay data from partition
        if task_id > 0 and (task_proportions is not None) and (sum(task_proportions) > 0.0):
            #partition = self.replay_schedule[self.n_replays]
            x_replay_from_partition, y_replay_from_partition, t_replay_from_partition = self.get_memory_for_training_from_partition(task_id, task_proportions)
            if self.verbose > 0:
                print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
                print('in trainer, selected y_replay: ', y_replay_from_partition)
                print('in trainer, t_replay: ', t_replay_from_partition)
                print()
        else:
            x_replay_from_partition, y_replay_from_partition, t_replay_from_partition = None, None, None

        for epoch in range(n_epochs):
            loss, loss_replay = 0.0, 0.0
            acc, acc_replay = 0.0, 0.0
            #for batch_idx, (x, y, t) in enumerate(data_loader):
            for batch_idx, (x, y) in enumerate(data_loader):
                #-----------------Collect data------------------#
                #x, y, t = x.to(device), y.to(device), t.to(device)
                x, y = x.to(device), y.to(device) 
                ### Get Replay Batch
                if task_id > 0 and x_replay_from_partition is not None:
                    x_replay, y_replay, t_replay = self.get_replay_batch(task_id, 
                                                            x_replay_from_partition, 
                                                            y_replay_from_partition, 
                                                            t_replay_from_partition,
                                                            shuffler=memory_replay_shuffler)
                else:
                    x_replay = y_replay = t_replay = None

                # Train the main model with this batch
                #print('in trainer, task_id: ', task_id)
                loss_dict = self.train_batch(x, y, x_=x_replay, y_=y_replay, task_id=task_id)
                #loss_dict = self.train_batch(x, y, t, 
                #                            x_=x_replay, y_=y_replay, t_=t_replay)
                #print('model.training: ', self.model.training)
                # Add batch results to metrics
                loss += loss_dict['loss']
                loss_replay += loss_dict['loss_replay']
                acc += loss_dict['acc']
                acc_replay += loss_dict['acc_replay']

            # End of epoch
            loss = loss / (batch_idx + 1)
            loss_replay = loss_replay / (batch_idx + 1)
            acc = acc / (batch_idx + 1)
            acc_replay = acc_replay / (batch_idx + 1)

            # Add metrics to logger
            epochs_total = (n_epochs * (task_id-1)) + (epoch+1)
            logger.add('loss', 'train', loss, it=epochs_total) 
            logger.add('accuracy', 'train', acc, it=epochs_total) 

            # Print stats
            loss_curr_last = logger.get_last('loss', 'train')
            acc_last = logger.get_last('accuracy', 'train')
            if ((epoch+1) % self.print_every) == 0:
                #print('[epoch %3d/%3d] loss current = %.4f, acc = %.4f' % (epoch+1, n_epochs, loss, acc))
                print('[epoch %3d/%3d] loss = %.4f, acc = %.4f, loss replay = %.4f, acc replay = %.4f'
                        % (epoch+1, n_epochs, loss, acc, loss_replay, acc_replay))
            
        ###################### END of training task ######################
        # Update memory buffer and parameters 
        self.update_coreset(train_dataset, task_id)
        
        print('End training for task %d...' % (task_id+1))
        self.last_trained_task = task_id
        self.logger.save_stats('stats.p')

    #def train_batch(self, x, y, t=None, x_=None, y_=None, t_=None, task_id=0):
    def train_batch(self, x, y, x_=None, y_=None, task_id=0):
        # Train model on single batch of current task samples 
        # and (optionally) replay samples
        self.model.train()
        self.optimizer.zero_grad()
        # Shorthands
        classes_per_task = self.classes_per_task
        scenario = self.scenario
        acc_replay = loss_replay = 0.0
        acc_curr = loss_curr = 0.0

        if x_ is None:
            if scenario == 'task':
                active_outputs = list(np.arange(task_id*classes_per_task, (task_id+1)*classes_per_task))
                y = y - (task_id*classes_per_task) # shift labels in task-il setting
            elif scenario == 'class':
                active_outputs = list(np.arange(0, (task_id+1)*classes_per_task))
            # compute loss and acc
            y_logits = self.model(x)[:, active_outputs] if (scenario in ['task', 'class']) else self.model(x)

            loss_curr = self.criterion(y_logits, y) 
            acc_curr = (y == y_logits.max(1)[1]).sum().item() / x.size(0)
            acc_replay, loss_replay = 0.0, 0.0
            loss = loss_curr 
            accuracy = acc_curr 
        else: 
            # get active outputs for replay data
            n_curr = len(y)
            x_all = torch.cat([x, x_], dim=0)
            y_all = torch.cat([y, y_], dim=0)
            task_ids = torch.floor(y_all / classes_per_task).long()
            if scenario == 'task':
                active_outputs = torch.arange(classes_per_task, dtype=torch.long, device=self.device)
                active_outputs = active_outputs.repeat(len(task_ids), 1)
                active_outputs = active_outputs + (task_ids*classes_per_task).unsqueeze(1)
                y_all = y_all - (task_ids*classes_per_task)
                #print('y_all: ', y_all)
            elif scenario == 'class':
                active_outputs = torch.arange(task_id*classes_per_task, dtype=torch.long, device=self.device)
                active_outputs = active_outputs.repeat(len(task_ids), 1)
            # compute loss and acc
            y_logits = self.model(x_all).gather(1, active_outputs) if (scenario in ['task', 'class']) else self.model(x_all)
            #print('active_outputs: ', active_outputs)
            #print('y_logits.size(): ', y_logits.size())
            #print()
            loss = self.criterion(y_logits, y_all) 
            accuracy = (y_all == y_logits.max(1)[1]).sum().item() / x_all.size(0)
            # compute separate current/replay loss and acc
            loss_curr = self.criterion(y_logits[:n_curr], y_all[:n_curr]) 
            #print('y_logits[:n_curr].size(): ', y_logits[:n_curr].size())
            #print('y_all[:n_curr]: ', y_all[:n_curr])
            acc_curr = (y_all[:n_curr] == y_logits[:n_curr].max(1)[1]).sum().item() / x.size(0)
            loss_replay = self.criterion(y_logits[n_curr:], y_all[n_curr:]) 
            acc_replay = (y_all[n_curr:] == y_logits[n_curr:].max(1)[1]).sum().item() / x_.size(0)

        """
        # compute acc and loss on replay
        #with torch.no_grad():
        if (x_ is not None):
            y_logits_ = self.model(x_, t_)
            loss_replay = self.criterion(y_logits_, y_) 
            acc_replay = (y_ == y_logits_.max(1)[1]).sum().item() / x_.size(0)
        else:
            acc_replay = loss_replay = 0.0

        # compute acc and loss on current task data
        y_logits = self.model(x, t)

        #print('zeroth data point: ', y_logits[0], t[0])
        #print('last data point: ', y_logits[-1], t[-1])
        loss_curr = self.criterion(y_logits, y) 
        acc_curr = (y == y_logits.max(1)[1]).sum().item() / x.size(0)

        if (x_ is not None):
            x = torch.cat([x, x_], dim=0)
            y = torch.cat([y, y_], dim=0)
            t = torch.cat([t, t_], dim=0) 

        y_logits = self.model(x, t)
        if self.scenario == 'class': # class incremental learning
            offset = (t[0].item() + 1) * self.classes_per_task 
            y_logits[:, offset:].data.fill_(-10e10)  
        loss = self.criterion(y_logits, y) 
        accuracy = (y == y_logits.max(1)[1]).sum().item() / x.size(0)
        """

        # Compute gradients for batch with current task data
        loss.backward()
        # Take optimization-step
        self.optimizer.step()
        return {'loss': loss, 'acc': accuracy,
                'loss_curr': loss_curr, 'acc_curr': acc_curr, 
                'loss_replay': loss_replay, 'acc_replay': acc_replay}

    def get_replay_batch(self, task, x_replay_from_partition, y_replay_from_partition, t_replay_from_partition, shuffler):
        # get mini-batch with replay sampels during training
        if len(x_replay_from_partition) <= self.batch_size:
            mem_indices = np.arange(len(x_replay_from_partition)) 
        else:
            mem_indices = shuffler.choice(len(x_replay_from_partition), size=self.batch_size, replace=False) 
        x_replay = x_replay_from_partition[mem_indices]
        y_replay = y_replay_from_partition[mem_indices]
        t_replay = t_replay_from_partition[mem_indices]
        return x_replay, y_replay, t_replay

    def reset(self,):
        # Initialize model and optimizer
        self.model = self.load_initial_checkpoint() 
        self.optimizer = self.prepare_optimizer()
        # reset replay buffer
        self.buffer = []
        print('in trainer reset(), self.seed', self.seed)
        self.summarizer = Summarizer.factory(type=self.sample_selection, rs=np.random.RandomState(self.seed))
        
        # reset epsiodic memory
        self.episodic_images.fill_(0.0)
        self.episodic_labels.fill_(-1)
        self.count_cls.fill_(0)
        self.episodic_filled_counter = 0 # used for ring buffer

        # Reset random number generators
        self.current_task = 0
        self.n_replays = 0

    def train_model_at_task(self, t, train_dataset, actions, task_proportion=None):

        # Train on task t
        if t <= len(actions):
            #print('actions: ', actions)
            path_exists, filename = self.model_checkpoint_exists(t, actions)  
            #print('filename: ', filename)
            path_exists = path_exists if self.load_checkpoints else False
        else:
            # if all actions could not be stored becuase they are too many
            filename = 'model_rollout_id_{}_task_{}.pth.tar'.format(rollout_id, t+1)
            path_exists = False
        if path_exists:
            #print('loading checkpoint {}'.format(checkpoint_dir + '/' + filename))
            self.load_checkpoint(checkpoint_dir=self.checkpoint_dir, file_path=filename)
            self.update_coreset(train_dataset, t) # update memory since doesn't go in to train_single_task()
            #self.trainer.episodic_filled_counter += self.trainer.memories_per_class * self.trainer.classes_per_task
        else:
            self.train_single_task(t, train_dataset, task_proportion)
            # Save checkpoint
            if self.config.session.save_checkpoints:
                self.save_checkpoint(task_id=t, folder=self.checkpoint_dir, file_name=filename)
        #print('filename: ', filename)
        #self.trainer.model = self.trainer.load_model_from_file(file_name=filename) # uses checkpoint_dir inside function

    def model_checkpoint_exists(self, task, actions=None):
        # Check if model checkpoint exists at task.
        if task == 0:
            model_path = 'model_0.pth.tar'
        else:    
            assert task == len(actions) # check that number of seen actions are the same as the task id
            indexing = '0-' + '-'.join([str(a) for a in actions])
            model_path = 'model_{}.pth.tar'.format(indexing)
        path = os.path.join(self.checkpoint_dir, model_path)
        return os.path.exists(path), model_path



    def train_single_task_in_bfs(self, task_id, train_dataset, task_proportions=None): 
        # Train model on single task with train_dataset
        self.model.train()
        # Shorthands
        scenario = self.scenario
        classes_per_task = self.classes_per_task
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        device = self.device
        logger = self.logger
        self.current_task = task_id

        #tracemalloc.start()
        #active_classes = self._get_active_classes_up_to_task_id(task_id)
        data_loader = get_data_loader(train_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=True,
                                    rng=torch.Generator().manual_seed(self.seed+task_id))
        memory_replay_shuffler = np.random.RandomState(self.seed+task_id)
        # Get replay data from partition
        if task_id > 0 and task_proportions is not None:
            x_replay_from_partition, y_replay_from_partition, t_replay_from_partition = self.get_memory_for_training_from_partition(task_id, 
                                                                                                                                    task_proportions)
            if self.verbose > 0:
                print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
                print('in trainer, selected y_replay: ', y_replay_from_partition)
                print('in trainer, t_replay: ', t_replay_from_partition)
                print()
        for epoch in range(n_epochs):
            loss, loss_replay = 0.0, 0.0
            acc, acc_replay = 0.0, 0.0
            #for batch_idx, (x, y, t) in enumerate(data_loader):
            for batch_idx, (x, y) in enumerate(data_loader):
                #-----------------Collect data------------------#
                #x, y, t = x.to(device), y.to(device), t.to(device)
                x, y = x.to(device), y.to(device)
                ### Get Replay Batch
                if task_id > 0:
                    x_replay, y_replay, t_replay = self.get_replay_batch(task_id, 
                                                            x_replay_from_partition, 
                                                            y_replay_from_partition, 
                                                            t_replay_from_partition,
                                                            shuffler=memory_replay_shuffler)
                else:
                    x_replay = y_replay = t_replay = None
                # Train the main model with this batch
                loss_dict = self.train_batch(x, y, x_=x_replay, y_=y_replay, task_id=task_id)
                #loss_dict = self.train_batch(x, y, t, 
                #                            x_=x_replay, y_=y_replay, t_=t_replay)
                #current, peak =  tracemalloc.get_traced_memory()
                #print("2 RAM current: {:0.2f}, peak: {:0.2f}".format(current, peak))
                # Add batch results to metrics
                loss += loss_dict['loss']
                loss_replay += loss_dict['loss_replay']
                acc += loss_dict['acc']
                acc_replay += loss_dict['acc_replay']
            # End of epoch
            loss = loss / (batch_idx + 1)
            loss_replay = loss_replay / (batch_idx + 1)
            acc = acc / (batch_idx + 1)
            acc_replay = acc_replay / (batch_idx + 1)
            # Add metrics to logger
            epochs_total = (n_epochs * (task_id-1)) + (epoch+1)
            logger.add('loss', 'train', loss, it=epochs_total) 
            logger.add('accuracy', 'train', acc, it=epochs_total) 
            # Print stats
            loss_curr_last = logger.get_last('loss', 'train')
            acc_last = logger.get_last('accuracy', 'train')
            if ((epoch+1) % self.print_every) == 0:
                print('[epoch %3d/%3d] loss = %.4f, acc = %.4f, loss replay = %.4f, acc replay = %.4f'
                        % (epoch+1, n_epochs, loss, acc, loss_replay, acc_replay))
        ###################### END of training task ######################
        ## Update memory buffer and parameters 
        if len(self.buffer) < task_id+1:
            self.update_coreset(train_dataset, task_id)
        
        #tracemalloc.stop()
        #print('End training for task %d...' % (task_id+1))
        self.last_trained_task = task_id
        self.logger.save_stats('stats.p')