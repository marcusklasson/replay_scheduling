
import os
import time 
import numpy as np
import torch

from trainer.rs import ReplaySchedulingTrainer
from trainer.utils import get_data_loader, pre_select_memory_inds
from trainer.utils import update_reservoir, update_fifo_buffer
from trainer.utils import flatten_grads, assign_grads

class ReplaySchedulingTrainerExtension(ReplaySchedulingTrainer):
    """ Trainer for Replay Scheduling. 
    """

    def __init__(self, config):
        super().__init__(config)
        self.training_extension = config['training']['extension']

        if self.training_extension in ['er', 'agem']:
            self.replay_selection = 'ring_buffer' 
        else:
            raise ValueError('Extension {} to Replay Scheduling is not implemented.'.format(self.training_extension))

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
        if task_id > 1 and self.replay_enabled and self.replay_schedule is not None:
            partition = self.replay_schedule[self.n_replays]
            x_replay_from_partition, y_replay_from_partition = self.get_memory_for_training_from_partition(partition)
            print('in trainer, selected y_replay: ', y_replay_from_partition)
            print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
            print()
            self.n_replays += 1
        t0 = time.time()
        print('self.episodic_labels: ', self.episodic_labels)
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
                if (task_id == 1) or (self.replay_enabled==False):
                    x_replay = y_replay = None   #-> if no replay
                else:
                    x_replay, y_replay = self.get_replay_batch(task_id, 
                                                            x_replay_from_partition, 
                                                            y_replay_from_partition, 
                                                            shuffler=memory_replay_shuffler)
                # Train the main model with this batch
                if self.training_extension == 'agem':
                    loss_dict = self.train_batch_with_agem(x_curr, y_curr, 
                                                        x_=x_replay, y_=y_replay,
                                                        active_classes=active_classes, 
                                                        task=task_id,)
                else:
                    loss_dict = self.train_batch(x_curr, y_curr, 
                                                x_=x_replay, y_=y_replay,
                                                active_classes=active_classes, 
                                                task=task_id,)
                # Add batch results to metrics
                loss_curr += loss_dict['loss_current']
                loss_replay += loss_dict['loss_replay']
                acc += loss_dict['accuracy']
                acc_replay += loss_dict['accuracy_replay']
                # Update episodic memory
                self.update_memory_during_training(x, y)

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
            #self.update_episodic_memory(train_dataset)
            self.episodic_filled_counter += self.memories_per_class * self.classes_per_task
        
        print('End training for task %d...' % task_id)
        self.last_trained_task = task_id
        self.logger.save_stats('stats.p')

    def update_memory_during_training(self, x, y):
        # Update episodic memory during training
        if self.replay_selection == 'ring_buffer':
            # Put the batch in the ring buffer
            update_fifo_buffer(current_images=x,
                current_labels=y,
                episodic_images=self.episodic_images,
                episodic_labels=self.episodic_labels,
                count_cls=self.count_cls,
                memories_per_class=self.memories_per_class,
                episodic_filled_counter=self.episodic_filled_counter,
                cl_scenario=self.scenario)
        else:
            raise ValueError('Selection method {} for replay samples does not exist'.format(self.replay_selection))


    def train_batch_with_agem(self, x, y, x_=None, y_=None, active_classes=None, task=1):
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