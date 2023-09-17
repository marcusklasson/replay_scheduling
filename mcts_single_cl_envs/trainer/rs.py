
import os
import time 
import numpy as np
import torch

from training.config import build_models, build_optimizers
from trainer.base import Trainer
from trainer.summary import Summarizer
from trainer.utils import get_data_loader 

class ReplaySchedulingTrainer(Trainer):
    """ Trainer for Replay Scheduling. 
    """

    def __init__(self, config):
        super().__init__(config)
        #self.replay_selection = config['replay']['selection']
        self.sample_selection = config['replay']['sample_selection']
        self.summarizer = Summarizer.factory(type=self.sample_selection, rs=np.random.RandomState(self.seed))
        self.memory_rs = np.random.RandomState(self.seed)
        self.memory_limit = config['replay']['memory_limit'] # cap on number of samples that can be stored in replay memory

        #if self.memory_limit > self.n_memories: # use full episodic memory for replay
        #    raise ValueError('Memory limit {} cannot be greater than episodic memory size {}'.format(self.memory_limit, self.n_memories))
            #self.memory_limit = self.n_memories
        self.replay_schedule = None # has to set before training
        self.n_replays = 0 # for indexing the replay schedule
        self.load_checkpoints = config['session']['load_checkpoints']
        #self.pre_select_inds = config['replay']['pre_select_inds']
        self.replay_enabled = True # used in single task replay experiment
        self.action_space_type = config['search']['action_space'] 

        # cap on number of samples that can be stored in replay memory if action space is task sampling type
        self.memory_limit_per_task = config['replay']['memory_limit_per_task'] if (self.action_space_type=='task_sampling') else -1 
        print(self.model)

        # Create history here based on memory_limit
        self.episodic_filled_counter = 0 # used for ring buffer
        self.examples_seen_so_far = 0 # used for reservoir sampling
        self.memories_per_class = config['replay']['examples_per_class'] #self.memory_limit // self.classes_per_task #config['replay']['examples_per_class']
        #assert config['replay']['memory_limit']/self.classes_per_task >= self.memories_per_class, 'set example_per_class in config properly!'
        self.n_classes = config['data']['n_classes'] 
        self.n_memories = self.classes_per_task * self.memories_per_class * self.n_tasks 
        self.count_cls = torch.zeros(self.n_classes, dtype=torch.long)
        #print('self.n_memories: ', self.n_memories)
        img_size = config['data']['img_size']
        in_channel = config['data']['in_channel']
        self.episodic_images = torch.FloatTensor(self.n_memories, in_channel, img_size, img_size)
        self.episodic_labels = -torch.ones(self.n_memories, dtype=torch.long) #torch.LongTensor(self.n_memories)
        # Add tensors to gpu or cpu
        self.episodic_images = self.episodic_images.to(self.device)
        self.episodic_labels = self.episodic_labels.to(self.device)

    def get_memory_for_training_from_partition(self, task_id, partition):
        """ Get memory samples from memory buffer stored in a dictionary.
            Works as an incremental memory that grabs all stored samples.
            Args:
                task_id (int): Current task id in range {0, 1, ..., n_tasks-1}
            Returns:
                memory (dict): Replay memory samples from tasks.
        """

        if self.episodic_filled_counter <= self.memory_limit:
            return self.episodic_images[:self.episodic_filled_counter], self.episodic_labels[:self.episodic_filled_counter]

        x_replay, y_replay = [], []
        n_samples_parts = self._divide_memory_size_into_parts_based_on_partition(task_id, partition)
        #print('n_samples_parts: ', n_samples_parts)
        #print('self.episodic_labels: ', self.episodic_labels)
        for t, n_samples_per_slot in enumerate(n_samples_parts): #partition.items():
            if n_samples_per_slot <= 0: # if any task_id should include no samples
                continue
            x_replay_t, y_replay_t = self.select_memory_data_from_task(task=t, n_samples=n_samples_per_slot)
            #print('mean of replay data: {:.5f}'.format(torch.mean(x_replay_t).item()))
            x_replay.append(x_replay_t) 
            y_replay.append(y_replay_t) 
        #print('y_replay: ', y_replay)
        return torch.cat(x_replay, dim=0), torch.cat(y_replay, dim=0)

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
        #print('mem_index: ', mem_index)

        # don't shuffle data
        #mem_ind_shuffler = np.random.RandomState(self.seed+self.current_task+(task+1))
        #mem_ind_shuffler.shuffle(mem_index)

        r_images, r_labels = [], []

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

    def _divide_memory_size_into_parts_based_on_partition(self, task_id, partition):
        #n_slots = sum([x > 0 for x in partition.values()])
        if self.action_space_type == 'task_sampling':
            M = self.memory_limit_per_task
        else:
            M = self.memory_limit
            mem_samples_per_task = self.classes_per_task * self.memories_per_class
        #print(partition)
        #x = [i*M for i in partition] #[i*M for i in list(partition.values())]
        n_samples_per_slot = [0] * len(partition)
        M_rest = M
        if (self.scenario == 'class') and (task_id > 1):
            # assure each slot receives at least "classes/task" number of samples
            # task_id = {1, ..., T}
            for t in range(task_id-1):
                n_samples_per_slot[t] += self.classes_per_task
                M_rest -= self.classes_per_task # compute new M
        #print('task_id: ', task_id)
        x = [i*M_rest for i in partition] #[i*M for i in partition]
        #print('partition: ', partition)
        #print('M_rest: ', M_rest)
        #print('x: ', x)
        n_samples_per_slot = [int(n_samples_per_slot[i] + k) for i,k in enumerate(x)]
        #n_samples_per_slot = [int(i) for i in x]
        #print('n_samples_per_slot1: ', n_samples_per_slot1)
        #print('n_samples_per_slot: ', n_samples_per_slot)
        for i in range(len(n_samples_per_slot)):
            if n_samples_per_slot[i] > mem_samples_per_task:
                n_samples_per_slot[i] = mem_samples_per_task
        samples_left = M - sum(n_samples_per_slot)

        #print('samples_left: ', samples_left)
        #print('n_samples_per_slot: ', n_samples_per_slot)
        if samples_left > 0 and (sum(partition) > 0.0):
            #print(x)
            frac_part, _ = np.modf(x[:task_id-1]) # only need to take fraction of samples in history
            # Add samples to slot with largest fraction part
            # If fraction parts are equal, add sample to oldest task first until no samples left to give
            #print(frac_part)
            indices = (-frac_part).argsort() # get indices of the largest fraction number
            #print('indices: ', indices[:task_id])
            #print('task_id:', task_id)
            indices = indices[:task_id-1]
            #print('task: ', task_id)
            #print('indices: ', indices)
            idx = 0
            while samples_left != 0:
                if n_samples_per_slot[indices[idx]] < mem_samples_per_task:
                    n_samples_per_slot[indices[idx]] += 1
                    samples_left -= 1
                if idx+1 < len(indices):
                    idx += 1
                else:
                    idx = 0
                #print(samples_left)
            """
            for idx in indices:
                if n_samples_per_slot[idx] >= mem_samples_per_task:
                    continue
                n_samples_per_slot[idx] += 1
                samples_left -= 1
                if samples_left == 0:
                    break
            """
        #print('after, n_samples_per_slot: ', n_samples_per_slot)
        return n_samples_per_slot

    def train_single_task(self, task_id, train_dataset): 
        """ Train model on single task dataset.

            Args:
                task_id (int): Task identifier (splitMNIST: 1-5).
                train_dataset (torch.Dataset): Training dataset for current task.
                partition (dict): Proportion of samples to grab from each task in each dictionary slot.
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
        if task_id > 1 and self.replay_enabled and (self.replay_schedule is not None) and (np.sum(self.replay_schedule[self.n_replays]) > 0.0):
            partition = self.replay_schedule[self.n_replays]
            x_replay_from_partition, y_replay_from_partition = self.get_memory_for_training_from_partition(task_id, partition)
            #assert len(y_replay_from_partition) == self.memory_limit or len(y_replay_from_partition) == self.episodic_filled_counter
            if self.verbose > 0:
                print('in trainer, len(selected y_replay): ', len(y_replay_from_partition))
                print('in trainer, selected y_replay: ', y_replay_from_partition)
                print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
                print()
            self.n_replays += 1
        else:
            x_replay_from_partition, y_replay_from_partition = None, None
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

    def get_replay_batch(self, task, x_replay_from_partition, y_replay_from_partition, shuffler):
        if len(x_replay_from_partition) <= self.batch_size:
            mem_indices = np.arange(len(x_replay_from_partition)) 
        else:
            mem_indices = shuffler.choice(len(x_replay_from_partition), size=self.batch_size, replace=False) 
        x_replay = x_replay_from_partition[mem_indices]
        y_replay = y_replay_from_partition[mem_indices]
        return x_replay, y_replay

    def update_coreset(self, dataset, t):
        size_per_task = self.buffer_size // (t+1)
        #size_per_class = size_per_task // self.classes_per_task
        # shrink buffer per task
        for j in range(t):
            (X, y) = self.buffer[j]
            X, y = X[:size_per_task], y[:size_per_task]
            self.buffer[j] = (X, y) 
        # Get new data
        X = np.stack([img.numpy() for img, _ in dataset], axis=0)
        y = np.stack([label for _, label in dataset], axis=0)
        chosen_inds = self.summarizer.build_summary(X, y, size_per_task, method=self.sample_selection,
                                                        model=self.model, device=self.device)
        X, y = X[chosen_inds], y[chosen_inds]
        assert (X.shape[0] == size_per_task)
        self.buffer.append((torch.from_numpy(X).to(self.device), torch.from_numpy(y).to(self.device)) )
        
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
                x_all = torch.cat([x, x_], dim=0)
                y_all = torch.cat([y, y_], dim=0)
                if scenario == 'task':
                    class_entries = torch.tensor(class_entries, device=self.device).repeat(len(y), 1)
                    #print('class_entries.size(): ', class_entries.size())
                    active_indices = torch.cat([class_entries, active_indices], dim=0)
                    #print('active_indices.size(): ', active_indices.size())
                    y_logits = self.model(x_all).gather(1, active_indices)
                else:
                    y_logits = self.model(x_all)[:, class_entries]

                loss = self.criterion(y_logits, y_all) 
                accuracy = (y_all == y_logits.max(1)[1]).sum().item() / x_all.size(0)

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

    def set_replay_schedule(self, rs):
        assert isinstance(rs, list)
        print('in trainer, setting replay schedule: ')
        print(np.stack(rs, axis=0))
        print()
        self.replay_schedule = rs.copy()

    def run_with_replay_schedule(self, datasets, replay_schedule, actions, rollout_id=None):
        self.set_replay_schedule(replay_schedule) # set replay schedule before training
        #print(actions)
        checkpoint_dir = self.checkpoint_dir

        n_tasks = self.n_tasks #config['data']['n_tasks']
        acc = np.zeros([n_tasks, n_tasks], dtype=np.float32)
        loss = np.zeros([n_tasks, n_tasks], dtype=np.float32)
        val_acc = np.zeros([n_tasks, n_tasks], dtype=np.float32)
        val_loss = np.zeros([n_tasks, n_tasks], dtype=np.float32)
        t0 = time.time() # for measuring elapsed time

        model_paths = []

        #for t, train_dataset in enumerate(datasets['train']):
        for t in range(self.n_tasks):
            train_dataset = datasets['train'][t]
            print('Training on dataset from task %d...' %(t+1))
            #print('Number of training examples: ', len(train_dataset))
            print()

            # Train on task t
            if t <= len(actions):
                path_exists, filename = self.model_checkpoint_exists(t+1, actions[:t])  
                path_exists = path_exists if self.load_checkpoints else False
            else:
                # if all actions could not be stored becuase they are too many
                filename = 'model_rollout_id_{}_task_{}.pth.tar'.format(rollout_id, t+1)
                path_exists = False

            if path_exists:
                #print('loading checkpoint {}'.format(checkpoint_dir + '/' + filename))
                self.load_checkpoint(checkpoint_dir=checkpoint_dir, file_path=filename)
                self.update_episodic_memory(train_dataset) # update memory since doesn't go in to train_single_task()
                self.episodic_filled_counter += self.memories_per_class * self.classes_per_task
                if (t+1) > 1:
                    self.n_replays += 1
            else:
                self.train_single_task(t+1, train_dataset)
                # Save checkpoint
                self.save_checkpoint(task_id=t+1, folder=checkpoint_dir, file_name=filename)
            # Testing model on all seen tasks
            test_model = self.load_model_from_file(file_name=filename) # uses checkpoint_dir inside function
            for u in range(t+1):
                #test_task = u+1 if scenario=='task' else t+1 # select correct head if Task-IL, otherwise pick all heads up to current task
                val_res = self.test(u+1, datasets['valid'][u], model=test_model)
                test_res = self.test(u+1, datasets['test'][u], model=test_model)
                print('>>> Test on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(u+1,
                                                                                    test_res['loss_t'],
                                                                                    100*test_res['acc_t']))
                acc[t, u], loss[t, u] = test_res['acc_t'], test_res['loss_t']
                val_acc[t, u], val_loss[t, u] = val_res['acc_t'], val_res['loss_t']
            # Update stuff
            model_paths.append(filename)
        # Get reward
        reward = np.mean(val_acc[-1, :]) # reward = avg acc over all tasks after learning task T
        res = {'reward': reward, # computed using validation set
                'acc': acc,
                'loss': loss,
                'rs': self.replay_schedule,
                'model_paths': model_paths,
                'val_acc': val_acc,
                'val_loss': val_loss,
                }
        return res

    def reset(self,):
        # Initialize model and optimizer
        self.model = self.load_initial_checkpoint() 
        self.optimizer = self.prepare_optimizer()

        # reset replay buffer
        self.episodic_images.fill_(0.0)
        #print(self.episodic_images)
        self.episodic_labels.fill_(-1)
        #print(self.episodic_labels) 
        self.count_cls.fill_(0)
        self.episodic_filled_counter = 0 # used for ring buffer
        self.examples_seen_so_far = 0 # used for reservoir sampling
        self.summarizer = Summarizer.factory(type=self.sample_selection, rs=np.random.RandomState(self.seed))

        # Reset random number generators
        self.memory_rs = np.random.RandomState(self.seed)
        self.gen_pytorch = torch.Generator().manual_seed(self.seed)
        self.current_task = 0
        self.n_replays = 0

    def model_checkpoint_exists(self, task, actions=None):
        """ Check if model checkpoint exists.
            Args:

                checkpoint_dir (str): Folder with saved model checkpoints.
            Return:
                (bool): 

        """
        if task == 1:
            model_path = 'model_0.pth.tar'
        else:    
            assert task-1 == len(actions) # check that number of seen actions are the same as the task id
            indexing = '0-' + '-'.join([str(a) for a in actions])
            model_path = 'model_{}.pth.tar'.format(indexing)
        path = os.path.join(self.checkpoint_dir, model_path)
        return os.path.exists(path), model_path

    
    def pre_select_memory_inds(self, datasets):
        chosen_inds_all = []
        for dataset in datasets:
            X = np.stack([img.numpy() for img, _ in dataset], axis=0)
            y = np.stack([label for _, label in dataset], axis=0)
            for y_ in np.unique(y):
                er_x = X[y == y_] 
                er_y = y[y == y_]
                chosen_inds = self.summarizer.build_summary(er_x, er_y, self.memories_per_class, method='uniform')
                chosen_inds_all.append(chosen_inds)
        self.chosen_inds = chosen_inds_all