
import os
import sys
import time 
import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim

from trainer.rs import ReplaySchedulingTrainer
from trainer.summary import Summarizer
from trainer.utils import get_data_loader 
from training.config import build_models

class ReplaySchedulingTrainerHAL(ReplaySchedulingTrainer):
    """ Trainer for Replay Scheduling with Hindsight Anchor Learning. 
    """

    def __init__(self, config):
        super().__init__(config)
        self.hal_lambda = config['replay']['hal_lambda'] # defuault: 0.1
        self.beta = config['replay']['beta'] # default: 0.7
        self.gamma = config['replay']['gamma'] #default: 0.5
        self.hal_lr = 0.1  #default: 0.1
        self.anchor_optimization_steps = 100 
        self.finetuning_epochs = 1
        self.spare_model = build_models(config) 
        self.spare_model.to(self.device)
        self.spare_opt = optim.SGD(self.spare_model.parameters(), lr=self.hal_lr)

        if 'val_threshold' in config['replay'].keys():
            self.replay_schedule = [] # create list for replay schedule when using heuristic
            self.val_threshold = config['replay']['val_threshold']

    def get_anchors(self, task_id, buffer):
        theta_t = self.model.get_params().detach().clone()
        self.spare_model.set_params(theta_t)

        memory_replay_shuffler = np.random.RandomState(self.seed+task_id+1) # shuffler for grabbing memory samples

        # fine tune on memory buffer
        inputs, labels = buffer
        for _ in range(self.finetuning_epochs):
            if len(inputs) <= self.batch_size:
                mem_indices = np.arange(len(inputs)) 
            else:
                mem_indices = memory_replay_shuffler.choice(len(inputs), size=self.batch_size, replace=False) 
            self.spare_opt.zero_grad()
            # Get task labels
            task_ids = torch.floor(labels / self.classes_per_task).long()
            #print(task_ids)
            if self.scenario == 'task':
                active_indices = torch.arange(self.classes_per_task, dtype=torch.long, device=self.device)
                active_indices = active_indices.repeat(len(task_ids), 1)
                active_indices = active_indices + (task_ids*self.classes_per_task).unsqueeze(1)
                labels = labels - (task_ids*self.classes_per_task)
            # Compute loss and take gradient step
            out = self.spare_model(inputs).gather(1, active_indices) if self.scenario == 'task' else self.spare_model(inputs)
            loss = self.criterion(out, labels)
            loss.backward()
            self.spare_opt.step()
        #print('replay loss: ', loss.item())
        theta_m = self.spare_model.get_params().detach().clone()

        #classes_for_this_task = np.unique(dataset.train_loader.dataset.targets)
        nct = self.classes_per_task
        classes_for_this_task = list(range(nct*(task_id-1), nct*task_id)) if self.scenario=='task' else list(range(self.n_classes))
        #print('in get_anchor, classes_for_this_task: ', classes_for_this_task)
        active_indices = torch.tensor(classes_for_this_task, device=self.device)
        #print(active_indices)
        #print('mean embedding vector, phi: ', self.phi.mean().item())
        for a_class in classes_for_this_task:
            e_t = torch.rand(self.input_shape, requires_grad=True, device=self.device)
            e_t_opt = optim.SGD([e_t], lr=self.hal_lr) 
            print(file=sys.stderr)
            for i in range(self.anchor_optimization_steps):
                e_t_opt.zero_grad()
                cum_loss = 0

                self.spare_opt.zero_grad()
                self.spare_model.set_params(theta_m.detach().clone())
                out = self.spare_model(e_t.unsqueeze(0))[0, active_indices] if self.scenario=='task' else self.spare_model(e_t.unsqueeze(0))[0]
                #out = out[:, classes_for_this_task] if self.scenario=='task' else out
                a_label = a_class*torch.ones((1), dtype=torch.long, device=self.device)
                # have to shift label if task-il
                a_label = a_label - ((task_id-1)*nct) if self.scenario == 'task' else a_label 
                #print('a_label: ', a_label)
                loss = -torch.sum(self.criterion(out.unsqueeze(0), a_label))
                loss.backward()
                cum_loss += loss.item()
                #print('finetuned weights, loss:', loss.item())

                self.spare_opt.zero_grad()
                self.spare_model.set_params(theta_t.detach().clone())
                out = self.spare_model(e_t.unsqueeze(0))[0, active_indices] if self.scenario=='task' else self.spare_model(e_t.unsqueeze(0))[0]
                #out = out[:, classes_for_this_task] if self.scenario=='task' else out
                loss = torch.sum(self.criterion(out.unsqueeze(0), a_label))
                loss.backward()
                cum_loss += loss.item()
                #print('old weights, loss:', loss.item())

                self.spare_opt.zero_grad()
                loss = torch.sum(self.gamma * (self.spare_model.features(e_t.unsqueeze(0)) - self.phi) ** 2)
                assert not self.phi.requires_grad
                loss.backward()
                cum_loss += loss.item()
                #print('features, mse loss:', loss.item())
                #

                e_t_opt.step()
                #print('anchor mean: ', e_t.mean().item())
                #print()
            e_t = e_t.detach()
            e_t.requires_grad = False
            self.anchors = torch.cat((self.anchors, e_t.unsqueeze(0)))
            del e_t
            #print('Total anchors:', len(self.anchors), file=sys.stderr)
            #print('cum loss: ', cum_loss)
        self.spare_model.zero_grad()

    def train_batch(self, x, y, x_=None, y_=None, active_classes=None, task=1):
        real_batch_size = x.shape[0]
        if not hasattr(self, 'input_shape'):
            self.input_shape = x.shape[1:]
        if not hasattr(self, 'anchors'):
            self.anchors = torch.zeros(tuple([0] + list(self.input_shape))).to(self.device)
        if not hasattr(self, 'phi'):
            print('Building phi', file=sys.stderr)
            with torch.no_grad():
                self.phi = torch.zeros_like(self.model.features(x[0].unsqueeze(0)), requires_grad=False)
            assert not self.phi.requires_grad

        self.model.train()
        self.optimizer.zero_grad()
        # Shorthands
        classes_per_task = self.classes_per_task
        scenario = self.scenario
        class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes

        old_weights = self.model.get_params().detach().clone()

        # Compute loss for current batch and replay batch
        if x_ is not None:
            # if Task-IL, find active output indices on model for replay batch
            task_ids = torch.floor(y_ / self.classes_per_task).long()
            if scenario == 'task':
                active_indices = torch.arange(self.classes_per_task, dtype=torch.long, device=self.device)
                active_indices = active_indices.repeat(len(task_ids), 1)
                active_indices = active_indices + (task_ids*classes_per_task).unsqueeze(1)
                #y_hat = y_hat_replay.gather(1, active_indices)
                y_ = y_ - (task_ids*classes_per_task)
            # Compute loss for both current and replay data
            x_all = torch.cat([x, x_], dim=0)
            y_all = torch.cat([y, y_], dim=0)
            if scenario == 'task':
                class_entries = torch.tensor(class_entries, device=self.device).repeat(len(y), 1)
                active_indices = torch.cat([class_entries, active_indices], dim=0)
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
                y_hat = y_hat[:, class_entries]
            # prediction loss
            loss = self.criterion(y_hat, y) 
            # Calculate training acc
            accuracy = (y == y_hat.max(1)[1]).sum().item() / x.size(0)

        # Compute gradients for batch with current task data
        loss.backward()
        # Take optimization-step
        self.optimizer.step()
        loss_curr = loss

        first_loss = 0
        #assert len(self.anchors) == self.classes_per_task * (task-1) # NOTE: disabled this assert for heuristic
        
        #print('in train_batch, anchor_classes: ', anchor_classes)
        if len(self.anchors) > 0:
            anchor_classes = list(range(self.classes_per_task * (task-1)))
            y_ = torch.tensor(anchor_classes, dtype=torch.long, device=self.device)
            task_ids = torch.floor(y_ / self.classes_per_task).long()
            if scenario == 'task':
                active_indices = torch.arange(self.classes_per_task, dtype=torch.long, device=self.device)
                active_indices = active_indices.repeat(len(task_ids), 1)
                active_indices = active_indices + (task_ids*classes_per_task).unsqueeze(1)
            first_loss = loss.item()
            with torch.no_grad():
                pred_anchors = self.model(self.anchors)[:, active_indices] if self.scenario =='task' else  self.model(self.anchors)

            self.model.set_params(old_weights)
            pred_anchors -= self.model(self.anchors)[:, active_indices] if self.scenario =='task' else  self.model(self.anchors) 
            
            loss = self.hal_lambda * (pred_anchors ** 2).mean()
            loss.backward()
            self.optimizer.step()
            loss_replay = loss
        with torch.no_grad():
            self.phi = self.beta * self.phi + (1 - self.beta) * self.model.features(x).mean(0) # use current batch only   

        # Calculate total replay loss
        loss_replay = None if (x_ is None) else loss_replay #sum(loss_replay) / len(y_)
        acc_replay = None #if (x_ is None) else acc_replay #sum(acc_replay) / len(acc_replay)

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
            if sum(partition) > 0.0: # check if any task has been selected to be replayed
                x_replay_from_partition, y_replay_from_partition = self.get_memory_for_training_from_partition(task_id, partition)
                #assert len(y_replay_from_partition) == self.memory_limit or len(y_replay_from_partition) == self.episodic_filled_counter
                if self.verbose > 0:
                    print('in trainer, len(selected y_replay): ', len(y_replay_from_partition))
                    print('in trainer, selected y_replay: ', y_replay_from_partition)
                    print('in trainer, mean selected x_replay: ', torch.mean(x_replay_from_partition))
                    print()
                
                # Get anchors from the replay data before learning current task!
                self.get_anchors(task_id-1, (x_replay_from_partition, y_replay_from_partition)) # take previous task_id
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
        
        #print('self.phi.mean(): ', self.phi.mean().item())
        print('End training for task %d...' % task_id)
        self.last_trained_task = task_id
        self.logger.save_stats('stats.p')

    def compute_phi(self, task_id, dataset):

        data_loader = get_data_loader(dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=True,
                                    rng=torch.Generator().manual_seed(self.seed+task_id))
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(self.device)
            if not hasattr(self, 'input_shape'):
                self.input_shape = x.shape[1:]
            # specific resets for trainer extension
            if not hasattr(self, 'anchors'):
                self.anchors = torch.zeros(tuple([0] + list(self.input_shape))).to(self.device)
            if not hasattr(self, 'phi'):
                print('Building phi', file=sys.stderr)
                with torch.no_grad():
                    self.phi = torch.zeros_like(self.model.features(x[0].unsqueeze(0)), requires_grad=False)
                assert not self.phi.requires_grad

            with torch.no_grad():
                self.phi = self.beta * self.phi + (1 - self.beta) * self.model.features(x).mean(0) # use current batch only 

    def run_with_replay_schedule(self, datasets, replay_schedule, actions, rollout_id=None):
        if hasattr(self, 'anchors'):
            del self.anchors 
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
                self.compute_phi(t+1, train_dataset)
                #print('self.phi.mean(): ', self.phi.mean().item())
                if (t+1) > 1:
                    partition = self.replay_schedule[self.n_replays]
                    x_replay_from_partition, y_replay_from_partition = self.get_memory_for_training_from_partition(t+1, partition)
                    self.get_anchors(t, (x_replay_from_partition, y_replay_from_partition)) # take previous task_id
                    self.n_replays += 1
            else:
                self.train_single_task(t+1, train_dataset)
                # Save checkpoint
                self.save_checkpoint(task_id=t+1, folder=checkpoint_dir, file_name=filename)
            #print('len(self.anchors): ', len(self.anchors))
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