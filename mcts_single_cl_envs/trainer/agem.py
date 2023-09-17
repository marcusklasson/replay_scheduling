
import time 

import torch
from torch.utils.data import DataLoader

from trainer.base import Trainer
from trainer.utils import flatten_grads, assign_grads, get_data_loader, update_fifo_buffer

class AGEM(Trainer):

    def __init__(self, config):
        super().__init__(config)

    def _is_violating_direction_constraint(self, grad_ref, grad_batch):
        """
        Check if gradient direction have angle less than 90 degrees against reference gradient.
        The gradient vectors have opposite directions when the dot product is negative.
        The gradient vectors are orthogonal if dot product is zero. 
        Args:
            grad_ref (torch.Tensor): Reference gradient (i.e., grads on episodic memory) 
            grad_batch (torch.Tensor): Batch gradient 
        Returns:
            (bool): 
        """
        return torch.dot(grad_ref, grad_batch) < 0

    def _project_grad_vector(self, grad_ref, grad_batch):
        """
        projects the proposed gradient from batch to the closest gradient (in squared L2 norm)
        to gradient that keeps the angle within the bound dot(grad_ref, grad_batch) >= 0.
        Eq. 11 in A-GEM paper, Chaudhry etal. (https://arxiv.org/abs/1812.00420)
        Args:
            grad_ref (torch.Tensor): Reference gradient (i.e., grads on episodic memory) 
            grad_batch (torch.Tensor): Batch gradient 
        Returns:
            (torch.Tensor): Projected gradient
        """
        dotp = torch.dot(grad_batch, grad_ref)
        ref_mag = torch.dot(grad_ref, grad_ref)
        return grad_batch - ((dotp / ref_mag) * grad_ref)

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

        valid_datasets=None

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

        #print(self.model)

        active_classes = self._get_active_classes_up_to_task_id(task_id)
        #print('active classes: ', active_classes)
        data_loader = get_data_loader(train_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=True,
                                    rng=torch.Generator().manual_seed(self.seed+task_id))
        t0 = time.time()
        print('self.episodic_labels: ', self.episodic_labels)
        print('self.episodic_filled_counter: ', self.episodic_filled_counter)
        for epoch in range(n_epochs):
            loss_curr = 0.0
            loss_replay = 0.0
            acc, acc_replay = 0.0, 0.0
            for batch_idx, (x, y) in enumerate(data_loader):
                #print(torch.unique(y))
                # evaluate on all tasks before training on batch
                if (verbose > 0) and (valid_datasets is not None):
                    if (batch_idx < 100 and batch_idx % 20 == 0) or (batch_idx % 100 == 0):
                        accuracy_list = []
                        for t in range(self.n_tasks):
                            res = self.evaluate_task(t+1, valid_datasets[t], test_size=500)
                            accuracy_list.append(round(res['acc_t'], 4))
                        print('batch idx {}, accuracy_list ==> {}'.format(batch_idx, accuracy_list))
                #print('y: ', y)
                #-----------------Collect data------------------#
                ### Current Batch
                #--> ITL: adjust current y-targets to 'active range', e.g. [0, 1] if 2 classes/task 
                if isinstance(classes_per_task, list): # adjusting range is different for Omniglot though
                    class_offset = active_classes[-1][0] # get first index of current class tasks
                    y_curr = y-class_offset if (scenario == "task") else y 
                else:
                    y_curr = y-classes_per_task*(task_id-1) if (scenario == "task") else y  
                x_curr, y_curr = x.to(device), y_curr.to(device) #--> transfer them to correct device
                
                ### Replay Batch
                x_replay, y_replay = self.get_replay_batch(task=task_id)
                #print('y_replay: ', y_replay)
                #print('x_replay mean: ', torch.mean(x_replay))

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

                # Use ring buffer strategy to update memory for A-GEM
                update_fifo_buffer(current_images=x,
                                    current_labels=y,
                                    episodic_images=self.episodic_images,
                                    episodic_labels=self.episodic_labels,
                                    count_cls=self.count_cls,
                                    memories_per_class=self.memories_per_class,
                                    episodic_filled_counter=self.episodic_filled_counter,
                                    cl_scenario=scenario)
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
            self.episodic_filled_counter += self.memories_per_class * self.classes_per_task

        print('End training for task %d...' % task_id)
        self.last_trained_task = task_id
        self.logger.save_stats('stats.p')

    def get_replay_batch(self, task):
        if task == 1:
            x_replay = y_replay = None   #-> if no replay
        else:
            if self.episodic_filled_counter <= self.batch_size:
                mem_indices = torch.arange(self.episodic_filled_counter, dtype=torch.long, device=self.device)
            else:
                # Sample a random subset from episodic memory buffer
                mem_indices = torch.randperm(self.episodic_filled_counter, dtype=torch.long, device=self.device)[:self.batch_size]
            x_replay = self.episodic_images[mem_indices]
            y_replay = self.episodic_labels[mem_indices]
        return x_replay, y_replay

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

            # Run model on current task data
            y_hat = self.model(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]
            # prediction loss
            loss_curr = self.criterion(y_hat, y) 
            # Calculate training acc
            accuracy = (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # Compute gradients for batch with current task data
            loss_curr.backward()
            if task > 1:
                grad_batch = flatten_grads(self.model)

            # Run model on replay data
            if x_ is not None:
                # In the Task-IL scenario, [y_] is a list and [x_] needs to be evaluated on each of them
                # (in case of 'exact' or 'exemplar' replay, [x_] is also a list!
                n_replays = len(y_) 

                # Prepare lists to store losses for each replay
                loss_replay = [] #[None]*n_replays
                acc_replay = []

                y_hat_all = self.model(x_)
                task_ids = torch.floor(y_ / self.classes_per_task).long()
                #print(task_ids)
                if scenario == 'task':
                    active_indices = torch.arange(self.classes_per_task, dtype=torch.long, device=self.device)
                    active_indices1 = active_indices.repeat(len(task_ids), 1)
                    active_indices2 = active_indices1 + (task_ids*classes_per_task).unsqueeze(1)
                    y_hat = y_hat_all.gather(1, active_indices2)
                    y_ = y_ - (task_ids*classes_per_task)
                elif scenario == 'class' or scenario == 'domain':
                    y_hat = y_hat_all[:, active_classes]

                # Compute loss and accuracy
                loss_replay = self.criterion(y_hat, y_)
                acc_replay = (y_ == y_hat.max(1)[1]).sum().item() / x_.size(0)

            # Calculate total replay loss
            loss_replay = None if (x_ is None) else loss_replay #sum(loss_replay) / len(y_)
            acc_replay = None if (x_ is None) else acc_replay #sum(acc_replay) / len(acc_replay)

            # calculate and store averaged gradient of replayed data
            if x_ is not None:
                # Perform backward pass to calculate gradient of replayed batch (if not yet done)
                loss_replay.backward()
                # Reorganize the gradient of the replayed batch as a single vector
                grad_ref = flatten_grads(self.model)
                # Check violating direction constraint
                if self._is_violating_direction_constraint(grad_ref, grad_batch):
                    #print('violated')
                    grad_batch = self._project_grad_vector(grad_ref, grad_batch)
                # Reset gradients (with A-GEM, gradients of replayed batch should only be used as inequality constraint)
                self.optimizer.zero_grad()
                # Assign gradients to model
                self.model = assign_grads(self.model, grad_batch)
            # Take optimization-step
            self.optimizer.step()

            # Return the dictionary with different training-loss split in categories
            return {
                'loss_current': loss_curr.item() if x is not None else 0,
                'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
                'accuracy': accuracy if accuracy is not None else 0.,
                'accuracy_replay': acc_replay if acc_replay is not None else 0.,
            }