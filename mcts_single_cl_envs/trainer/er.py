
import time 
import numpy as np
import torch

from trainer.base import Trainer
from trainer.utils import update_reservoir, update_fifo_buffer, get_data_loader
from trainer.summary import Summarizer

class ER(Trainer):
    """ Trainer for Experience Replay. 
    """

    def __init__(self, config):
        super().__init__(config)
        #self.replay_selection = config['replay']['selection']
        self.sample_selection = config['replay']['sample_selection']
        if self.sample_selection not in ['ring_buffer', 'reservoir']:
            self.summarizer = Summarizer.factory(type=self.sample_selection, rs=np.random.RandomState(self.seed))

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
        #print('active classes: ', active_classes)
        data_loader = get_data_loader(train_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=True,
                                    rng=torch.Generator().manual_seed(self.seed+task_id))#self.gen_pytorch)
        t0 = time.time()
        print('self.episodic_filled_counter: ', self.episodic_filled_counter)
        print('self.episodic_labels: ', self.episodic_labels)
        print('mean(self.episodic_images): ', torch.mean(self.episodic_images[:self.episodic_filled_counter]))
        for epoch in range(n_epochs):
            loss_curr = 0.0
            loss_replay = 0.0
            acc, acc_replay = 0.0, 0.0
            for batch_idx, (x, y) in enumerate(data_loader):
                
                # evaluate on all tasks before training on batch
                if (verbose > 0) and (valid_datasets is not None):
                    if (batch_idx < 100 and batch_idx % 20 == 0) or (batch_idx % 100 == 0):
                        accuracy_list = []
                        for t in range(self.n_tasks):
                            res = self.evaluate_task(t+1, valid_datasets[t], 500)
                            accuracy_list.append(round(res['acc_t'], 4))
                        print('batch idx {}, accuracy_list ==> {}'.format(batch_idx, accuracy_list))
                
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
                x_replay, y_replay = self.get_replay_batch(task=task_id)
                #if task_id>1:
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

                # Update episodic memory during training
                if self.sample_selection == 'ring_buffer':
                    # Put the batch in the ring buffer
                    update_fifo_buffer(current_images=x,
                        current_labels=y,
                        episodic_images=self.episodic_images,
                        episodic_labels=self.episodic_labels,
                        count_cls=self.count_cls,
                        memories_per_class=self.memories_per_class,
                        episodic_filled_counter=self.episodic_filled_counter,
                        cl_scenario=scenario)
                    #print()
                elif self.sample_selection == 'reservoir':
                    for er_x, er_y in zip(x, y):
                        update_reservoir(current_image=er_x, 
                                        current_label=er_y, 
                                        episodic_images=self.episodic_images, 
                                        episodic_labels=self.episodic_labels, 
                                        M=self.n_memories,
                                        N=self.examples_seen_so_far)
                        self.examples_seen_so_far += 1
                    if (batch_idx % 1000 == 0) and batch_idx > 0:
                        #print('in ER, task {}, mem_filledso_far: {}'.format(task_id, mem_filled_so_far))
                        print('examples_seen_so_far: {}'.format(self.examples_seen_so_far))
                        print('episodic_labels: ', self.episodic_labels)
                        print()
                        print('er_minibatch: ', y_replay)
                        #print('size of minibatch: ', y_replay.size(0))
                        print()
                #else:
                #    raise ValueError('Selection method {} for replay samples does not exist'.format(self.sample_selection))
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
        # Update memory parameters 
        if self.use_episodic_memory:
            if self.sample_selection not in ['ring_buffer', 'reservoir']:
                self.update_episodic_memory(train_dataset)
            self.episodic_filled_counter += self.memories_per_class * self.classes_per_task

        print('End training for task %d...' % task_id)
        self.last_trained_task = task_id
        self.logger.save_stats('stats.p')

    def get_replay_batch(self, task):

        if self.sample_selection == 'reservoir':
            mem_filled_so_far = self.examples_seen_so_far if (self.examples_seen_so_far < self.n_memories) else self.n_memories
            if mem_filled_so_far < self.batch_size:
                mem_indices = torch.arange(mem_filled_so_far, dtype=torch.long, device=self.device)
            else:
                # Sample a random subset from episodic memory buffer
                mem_indices = torch.randperm(mem_filled_so_far, dtype=torch.long, device=self.device)[:self.batch_size]
            x_replay = self.episodic_images[mem_indices] if mem_filled_so_far>0 else None
            y_replay = self.episodic_labels[mem_indices] if mem_filled_so_far>0 else None

        #if self.replay_selection == 'ring_buffer':
        else:
            #mem_filled = torch.ne(self.episodic_labels[0], -1).item()
            if task == 1:
                x_replay = y_replay = None   #-> if no replay
            else:
                if self.episodic_filled_counter <= self.batch_size:
                    mem_indices = torch.arange(self.episodic_filled_counter, dtype=torch.long, device=self.device)
                else:
                    # Sample a random subset from episodic memory buffer
                    mem_indices = torch.randperm(self.episodic_filled_counter, dtype=torch.long, device=self.device)[:self.batch_size]
                x_replay = self.episodic_images[mem_indices] #if mem_filled else None
                y_replay = self.episodic_labels[mem_indices] #if mem_filled else None
        return x_replay, y_replay

    def update_episodic_memory(self, dataset):
        X = np.stack([img.numpy() for img, _ in dataset], axis=0)
        y = np.stack([label for _, label in dataset], axis=0)
        for y_ in np.unique(y): # class-wise selection
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
                #print('x.size(): ', x.size())
                #print('y.size(): ', y.size())
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