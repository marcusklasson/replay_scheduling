
import os
import time 
import numpy as np
import torch

#from training.config import build_models, build_optimizers
#from trainer.base import Trainer
from trainer.summary import Summarizer
from trainer.utils import get_data_loader 
from trainer.rs import ReplaySchedulingTrainer

class ReplaySchedulingTrainerCoreset(ReplaySchedulingTrainer):
    """ Trainer for Replay Scheduling. 
    """

    def __init__(self, config):
        super().__init__(config)

        # coreset buffer
        self.buffer = []
        self.buffer_size = self.memories_per_class * self.classes_per_task * self.n_tasks 

        """
        # update during training (online) or after training task
        self.update_episodic_memory_online = False
        if self.replay_selection in ['ring_buffer', 'reservoir']:
            self.update_episodic_memory_online = True
        """

    def get_memory_for_training_from_partition(self, task_id, partition):
        """ Get memory samples from memory buffer stored in a dictionary.
            Works as an incremental memory that grabs all stored samples.
            Args:
                task_id (int): Current task id in range {0, 1, ..., n_tasks-1}
            Returns:
                memory (dict): Replay memory samples from tasks.
        """
        x_replay, y_replay = [], []
        n_samples_parts = self._divide_memory_size_into_parts_based_on_partition(task_id, partition)
        #print(n_samples_parts)
        for t, n_samples_per_slot in enumerate(n_samples_parts): #partition.items():
            if n_samples_per_slot <= 0: # if any task_id should include no samples
                continue
            x_replay_t, y_replay_t = self.select_memory_data_from_task(task=t, n_samples=n_samples_per_slot)
            #print('mean of replay data: {:.5f}'.format(torch.mean(x_replay_t).item()))
            x_replay.append(x_replay_t) 
            y_replay.append(y_replay_t) 
        return torch.cat(x_replay, dim=0), torch.cat(y_replay, dim=0)

    def select_memory_data_from_task(self, task, n_samples):
        """ task = {0, ..., T-1}
        """
        (X, y) = self.buffer[task] # get data from coreset buffer
        #mem_ind_shuffler = np.random.RandomState(self.seed+self.current_task+(task+1))
        mem_index = list(range(len(y)))
        #mem_ind_shuffler.shuffle(mem_index)

        r_images, r_labels = [], []
        class_samples = {}
        n_selected_samples = 0

        for idx in range(len(mem_index)):
            data = X[mem_index[idx], :]
            label = y[mem_index[idx]]
            cid = label.item() if isinstance(label, torch.Tensor) else label
            if cid in class_samples:
                #if len(class_samples[cid]) < n_samples_per_class:
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
        #print('task {}, r_labels: {}'.format(task, r_labels))
        return torch.cat(r_images, dim=0), torch.cat(r_labels, dim=0)

    def _divide_memory_size_into_parts_based_on_partition(self, task_id, partition):
        #n_slots = sum([x > 0 for x in partition.values()])
        if self.action_space_type == 'task_sampling':
            M = self.memory_limit_per_task
        else:
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
            x_replay_from_partition, y_replay_from_partition = self.get_memory_for_training_from_partition(task_id, partition)
            if self.verbose > 0:
                print('in trainer, selected y_replay: ', y_replay_from_partition)
                print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
                print()
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
                if (task_id == 1) or (self.replay_enabled==False):
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
            self.update_coreset(train_dataset, task_id-1)
        
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
        """
        size_per_task = self.buffer_size // (t+1)
        #size_per_class = size_per_task // self.classes_per_task
        # shrink buffer per task
        for j in range(t):
            (X, y) = self.buffer[j]
            X, y = X[:size_per_task], y[:size_per_task]
            self.buffer[j] = (X, y) 
        """
        size_per_task = self.memory_limit
        # Get new data
        X = np.stack([img.numpy() for img, _ in dataset], axis=0)
        y = np.stack([label for _, label in dataset], axis=0)
        chosen_inds = self.summarizer.build_summary(X, y, size_per_task, method=self.sample_selection,
                                                        model=self.model, device=self.device)
        X, y = X[chosen_inds], y[chosen_inds]
        assert (X.shape[0] == size_per_task)
        self.buffer.append((torch.from_numpy(X).to(self.device), torch.from_numpy(y).to(self.device)) )


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
                self.update_coreset(train_dataset, t) # update memory since doesn't go in to train_single_task()
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
        self.buffer = []
        self.summarizer = Summarizer.factory(type=self.sample_selection, rs=np.random.RandomState(self.seed))

        # Reset random number generators
        self.memory_rs = np.random.RandomState(self.seed)
        self.gen_pytorch = torch.Generator().manual_seed(self.seed)
        self.current_task = 0
        self.n_replays = 0
