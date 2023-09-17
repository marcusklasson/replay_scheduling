
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

class ReplayMemory(object):

    def __init__(self, args): 

        self.replay_method = args.replay_method #config['replay']['method']
        self.memory_size = args.memory_size
        self.classes_per_task = 2 # for split MNIST 
        self.device = args.device # if we need to extract features
        self.datasets = [] # keeping all datasets that can be used for replay

        self.X_b = []
        self.y_b = []

        self.n_seen_datasets = 0

        #self.selection_method = args.selection_method # options: [first, random]
        #self.random_selection = True if (self.selection_method == 'random') else False
        #self.balance_classes = True 
        self.verbose = args.verbose

        if self.replay_method == 'single_task':
            self.task_for_replay = config['replay']['task_for_replay']
            self.use_memory_at_task = config['replay']['use_memory_at_task']

    def _add_dataset(self, dataset):
        self.datasets.append(dataset)
        self.n_seen_datasets += 1

    def reset_memory(self):
        self.X_b = []
        self.y_b = []
        self.n_seen_datasets = 0
        self.datasets = []

    def update_memory(self, task_id, dataset):
        """ Update memory with M samples from dataset with data points from current task.
            Args:
                task_id (int): Task identity
                dataset (torch.Dataset): Training dataset we grab subset of replay points from.
        """
        self._add_dataset(dataset) # store dataset just in case
        task_data = self._get_memory_points_from_dataset(task_id, dataset, num_points=self.memory_size)
        self.X_b.append(task_data[0])
        self.y_b.append(task_data[1])

    def update_memory_with_given_samples_per_class(self, task_id, dataset, samples_per_class=1):
        """ Update memory with X samples per class from dataset with data points from current task.
            Args:
                task_id (int): Task identity
                dataset (torch.Dataset): Training dataset we grab subset of replay points from.
        """
        self._add_dataset(dataset) # store dataset just in case
        task_data = self._get_memory_points_from_dataset(task_id, dataset, num_points=self.memory_size,
                                                        samples_per_class=samples_per_class)
        self.X_b.append(task_data[0])
        self.y_b.append(task_data[1])


    def get_memory_for_training(self, task_id):
        """ Get memory samples from memory buffer stored in a dictionary!
            Args:
                task_id (int): Current task id in range {0, 1, ..., n_tasks-1}
            Returns:
                memory (dict): Replay memory samples from wanted tasks.
        """
        replay_method = self.replay_method
        memory = {} # NOTE: using dict instead of list in this function!
        if replay_method == 'equal':
            n_samples_parts = self._divide_memory_size_into_equal_parts(task_id)
            for t, n_samples_per_slot in enumerate(n_samples_parts): #partition.items():
                if n_samples_per_slot <= 0: # if any task_id should include no samples
                    continue
                task_data = self._get_memory_points_from_buffer(t, n_samples_per_slot)
                memory[t] = task_data

        elif replay_method == 'single_task':
            if (self.task_for_replay < (task_id+1)) and (self.use_memory_at_task == (task_id+1)):
                X_task = self.X_b[self.task_for_replay-1]
                y_task = self.y_b[self.task_for_replay-1]
                memory[self.task_for_replay-1] = [X_task, y_task]
        else:
            raise ValueError('Replay method %s is invalid for this function.' %(replay_method))
        return memory

    def get_memory_for_training_with_partition(self, task_id, partition):
        """ Get memory samples from memory buffer stored in a dictionary!
            Args:
                task_id (int): Current task id in range {0, 1, ..., n_tasks-1}
                partition (dict): Percentages of samples per task t opstore in memory {1: p(1), 2: p(2), ..., T-1: p(T-1)}
            Returns:
                memory (dict): Replay memory samples from wanted tasks.
        """
        memory = {} # NOTE: using dict instead of list in this function!
        n_samples_parts = self._divide_memory_size_into_parts_based_on_partition(partition)
        for t, n_samples_per_slot in enumerate(n_samples_parts): #partition.items():
            if n_samples_per_slot <= 0: # if any task_id should include no samples
                continue
            task_data = self._get_memory_points_from_buffer(t, n_samples_per_slot)
            memory[t] = task_data
        return memory

    def _get_memory_points_from_buffer(self, task_id, num_points):
        """ Grab data points from existing memory buffer.
            Args:
                task_id (int): Task identity, starting from 0
                n_samples (int): Number of samples to grab from task
            Return:
                list with task_data
        """

        r_points = []
        r_labels = []

        X_b = self.X_b[task_id]
        y_b = self.y_b[task_id]

        classes_per_task = self.classes_per_task
        if isinstance(classes_per_task, list):
            classes_per_task = classes_per_task[task_id][1]

        """
        if self.verbose == 2:
            #print('verbose: ', self.verbose)
            indices = torch.arange(y_b.size(0))
        else:
            indices = torch.randperm(y_b.size(0)) # grab random samples from dataset in experiment mode
        """
        indices = torch.arange(y_b.size(0))

        class_samples = {}
        # NOTE: using np.ceil and then exact_num_points = num_points fills up the memory as equally as it can
        # NOTE: using np.floor and then num_points_per_class*classes_per_task always returns equal number/class but don't fill up memory
        num_points_per_class = int(np.ceil(num_points/classes_per_task)) #int(np.ceil(num_points/classes_per_task))
        exact_num_points = num_points #num_points_per_class*classes_per_task
        #print('Number of points total: ', num_points)
        #print('number of points/class: ', num_points_per_class)

        select_points_num = 0
        for idx in range(len(indices)):
            data = X_b[indices[idx], :]
            label = y_b[indices[idx]]
            cid = label.item() if isinstance(label, torch.Tensor) else label
            if cid in class_samples:
                if len(class_samples[cid]) < num_points_per_class:
                    class_samples[cid].append(data)
                    select_points_num += 1
            else:
                class_samples[cid] = [data]
                select_points_num += 1
            if select_points_num >= exact_num_points:
                break
        for cid in class_samples.keys(): #range(num_classes):
            r_points.append(torch.stack(class_samples[cid], dim=0))
            r_labels.append(torch.ones(len(class_samples[cid]), dtype=torch.long)*cid)
        return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)] 
        """
        if self.balance_classes:
            class_samples = {}
            # NOTE: using np.ceil and then exact_num_points = num_points fills up the memory as equally as it can
            # NOTE: using np.floor and then num_points_per_class*classes_per_task always returns equal number/class but don't fill up memory
            num_points_per_class = int(np.ceil(num_points/classes_per_task)) #int(np.ceil(num_points/classes_per_task))
            exact_num_points = num_points #num_points_per_class*classes_per_task
            #print('Number of points total: ', num_points)
            #print('number of points/class: ', num_points_per_class)

            select_points_num = 0
            for idx in range(len(indices)):
                data = X_b[indices[idx], :]
                label = y_b[indices[idx]]
                cid = label.item() if isinstance(label, torch.Tensor) else label
                if cid in class_samples:
                    if len(class_samples[cid]) < num_points_per_class:
                        class_samples[cid].append(data)
                        select_points_num += 1
                else:
                    class_samples[cid] = [data]
                    select_points_num += 1
                if select_points_num >= exact_num_points:
                    break
            for cid in class_samples.keys(): #range(num_classes):
                r_points.append(torch.stack(class_samples[cid], dim=0))
                r_labels.append(torch.ones(len(class_samples[cid]), dtype=torch.long)*cid)
            return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)] 
        else:
            exact_num_points = num_points
            select_points_num = 0
            for idx in range(len(indices)):
                data = X_b[indices[idx], :].unsqueeze(0)
                label = y_b[indices[idx]]
                r_points.append(data)
                r_labels.append(label)
                select_points_num += 1
                if select_points_num >= exact_num_points:
                    break
            return [torch.cat(r_points, dim=0), torch.LongTensor(r_labels)] 
        """

    def _get_memory_points_from_dataset(self, task_id, dataset, num_points, samples_per_class=0):
        """ Grab M number of data points from training dataset.
            Args:
                task_id (int): Task identity, starting from 0
                dataset (torch.Dataset): Training dataset.
                num_points (int): Number of memory points to fetch.
            Return:
                list with task_data
        """

        r_points = []
        r_labels = []

        classes_per_task = self.classes_per_task
        if isinstance(classes_per_task, list):
            classes_per_task = classes_per_task[task_id][1]   

        if samples_per_class > 0:
            num_points_per_class = samples_per_class
            exact_num_points = (num_points_per_class*classes_per_task)
        else:  
            num_points_per_class = int(np.ceil(num_points/classes_per_task))
            exact_num_points = num_points
        select_points_num = 0

        """
        #print('verbose: ', self.verbose)
        if self.verbose == 2:
            indices = torch.arange(len(dataset))
        else:
            indices = torch.randperm(len(dataset)) # grab random samples from dataset in experiment mode
        """
        indices = torch.randperm(len(dataset)) # grab random samples from dataset in experiment mode

        class_samples = {}
        for idx in range(len(indices)):
            data, label, task = dataset[indices[idx]]
            cid = label.item() if isinstance(label, torch.Tensor) else label
            if cid in class_samples:
                if len(class_samples[cid]) < num_points_per_class:
                    class_samples[cid].append(data)
                    select_points_num += 1
            else:
                class_samples[cid] = [data]
                select_points_num += 1
            if select_points_num >= exact_num_points:
                break
        for cid in class_samples.keys(): #range(num_classes):
            r_points.append(torch.stack(class_samples[cid], dim=0))
            r_labels.append(torch.ones(len(class_samples[cid]), dtype=torch.long)*cid)
        #print('n_examples: ', torch.cat(r_points, dim=0).size())
        return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)] 
        """
        if self.balance_classes:
            class_samples = {}
            for idx in range(len(indices)):
                data, label, task = dataset[indices[idx]]
                cid = label.item() if isinstance(label, torch.Tensor) else label
                if cid in class_samples:
                    if len(class_samples[cid]) < num_points_per_class:
                        class_samples[cid].append(data)
                        select_points_num += 1
                else:
                    class_samples[cid] = [data]
                    select_points_num += 1
                if select_points_num >= exact_num_points:
                    break
            for cid in class_samples.keys(): #range(num_classes):
                r_points.append(torch.stack(class_samples[cid], dim=0))
                r_labels.append(torch.ones(len(class_samples[cid]), dtype=torch.long)*cid)
            #print('n_examples: ', torch.cat(r_points, dim=0).size())
            return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)] 
        else:
            for idx in range(len(indices)):
                data, label, task = dataset[indices[idx]]
                r_points.append(data.unsqueeze(0))
                r_labels.append(label)
                select_points_num += 1
                if select_points_num >= exact_num_points:
                    break
            return [torch.cat(r_points, dim=0), torch.LongTensor(r_labels)] 
        """

    def get_memory_for_training_from_dataset(self, task_id):
        """ Get memory samples from the full datasets.
            Args:
                task_id (int): Current task id in range {0, 1, ..., n_tasks-1}
            Returns:
                memory (list): Replay memory samples from wanted tasks.
        """
        replay_method = self.replay_method
        memory = []
        if replay_method == 'equal':
            n_samples_parts = self._divide_memory_size_into_equal_parts(task_id)
            print(n_samples_parts)
            for t, n_samples_per_slot in enumerate(n_samples_parts): #partition.items():
                dataset = self.datasets[t]
                task_data = self._get_memory_points_from_dataset(task_id, dataset, n_samples_per_slot)
                memory.append(task_data)

        elif replay_method == 'single_task':
            #print(task_id)
            #print(len(self.datasets))
            if (self.task_for_replay < (task_id+1)) and (self.use_memory_at_task == (task_id+1)):
                dataset = self.datasets[self.task_for_replay-1]
                task_data = self._get_memory_points_from_dataset(task_id, dataset, self.memory_size)
                memory.append(task_data)

        elif replay_method == 'rs':
            partition = self.mem_partition[task_id]
            n_samples_parts = self._divide_memory_size_into_parts_based_on_partition(partition)
            for t, n_samples_per_slot in enumerate(n_samples_parts): #partition.items():
                if n_samples_per_slot <= 0: # if any task_id should include no samples
                    continue
                dataset = self.datasets[t]
                task_data = self._get_memory_points_from_dataset(task_id, dataset, n_samples_per_slot)
                memory.append(task_data)
        return memory

    def set_partition(self, partition):
        self.mem_partition = partition

    def _divide_memory_size_into_equal_parts(self, div, shuffle=True):
        M = self.memory_size
        parts = [M // div + (1 if x < (M % div) else 0)  for x in range(div)]
        parts = np.array(parts)
        if shuffle:
            np.random.shuffle(parts)
        return parts

    def _divide_memory_size_into_parts_based_on_partition(self, partition):
        #n_slots = sum([x > 0 for x in partition.values()])
        M = self.memory_size
        x = [i*M for i in list(partition.values())]
        n_samples_per_slot = [int(i) for i in x]
        samples_left = M - sum(n_samples_per_slot)

        if samples_left > 0:
            frac_part, _ = np.modf(x)
            """ TO-DO: Random sample an index if fractions are the same??
            all_fracs_equal = np.all(frac_part == frac_part[0])
            if all_fracs_equal:
                indices = np.random.permutation(len(frac_part))
            else:
                indices = (-frac_part).argsort() # get indices of the largest fraction number
            """
            # Add samples to slot with largest fraction part
            # If fraction parts are equal, add sample to oldest task first until no samples left to give
            indices = (-frac_part).argsort() # get indices of the largest fraction number
            for idx in indices:
                n_samples_per_slot[idx] += 1
                samples_left -= 1
                if samples_left == 0:
                    break
        return n_samples_per_slot

    def print_memory_samples(self):
        print('Printing memory slots and labels')
        for t, y_ in enumerate(self.y_b):
            print(t, len(y_), torch.sort(y_)[0])
        print() 

    def update_memory_with_selection_method(self, task_id, dataset, model):
        """
        Samples from a dataset based on a icarl - mean of features
        Args:
            dataset             Dataset to sample from
            features            Features - activation before the last layer
            task                Labels with in a task
            samples_count       Number of samples to return
        Return:
            images              Important images
            labels              Important labels
        """
        # Shorthands
        device = self.device
        classes_per_task = self.classes_per_task
        selection_method = self.selection_method 

        n_train = len(dataset)
        batch_size = 256
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # Get features H for all classes in task
        H = []
        Y = []
        model.eval()
        with torch.no_grad():
            count = 0
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
                h = model.features(x) if (selection_method == 'mean_of_features') else x # coreset is in input space
                H.append(h.detach())
                Y.append(y.detach())
        H = torch.cat(H, dim=0)
        Y = torch.cat(Y, dim=0)
        
        assert H.size(0) == n_train
        assert Y.size(0) == n_train

        task_classes = list(range(classes_per_task*(task_id), classes_per_task*(task_id+1)))        
        count = 0
        list_of_selected = []
        # For each label in the task extract the important samples
        for label in task_classes:
            
            class_idxs = torch.squeeze(torch.nonzero((Y == label)), dim=-1)
            n_class_samples = class_idxs.size(0)
            actual_samples_count = min(self.memory_size//classes_per_task, n_class_samples)

            if selection_method == 'mean_of_features':
                """ Mean-of-Features selection """
                H_mean = torch.mean(H[class_idxs, :], dim=0)
                sample_sum = torch.zeros(H_mean.size()).to(device) #np.zeros(mean_feature.shape)
                if (actual_samples_count != 0):
                    # Extract the important indices
                    for i in range(actual_samples_count):
                        sample_mean = (H[class_idxs, :] + torch.tile(sample_sum, [n_class_samples, 1])) / float(i+1)
                        norm_distance = torch.norm((torch.tile(H_mean, [n_class_samples, 1]) - sample_mean), p=2, dim=1)
                        idx_selected = class_idxs[torch.argmin(norm_distance)] # selected in H 
                        #print(idx_selected)
                        if idx_selected in list_of_selected:
                            raise ValueError("Exemplars should not be repeated.")
                        list_of_selected.append(idx_selected.item())
                        sample_sum = sample_sum + H[idx_selected, :]
                        H[idx_selected, :] = H[idx_selected, :] + 10000 # to avoid selecting this feature again

            elif selection_method == 'k_center_coreset':
                """ K-center coreset selection """
                dists = torch.ones(n_class_samples)*1e6 #np.full(x_train.shape[0], np.inf)
                dists = dists.to(device)
                current_id = 0
                dists = self._update_distance(dists, H[class_idxs, :], current_id)
                idxs = [current_id]
                for i in range(1, actual_samples_count):
                    current_id = torch.argmax(dists)
                    current_id = current_id.item()
                    dists = self._update_distance(dists, H[class_idxs, :], current_id)
                    if current_id in idxs:
                        raise ValueError("Exemplars should not be repeated.")
                    idxs.append(current_id)
                # Add indices from K-center algorithm
                for i in idxs:
                    list_of_selected.append(class_idxs[i].item())
            else:
                raise ValueError('Selection method {} not valid.'.format(strategy))
        # Get the selected images and labels from dataset
        selected_data = [dataset[i] for i in list_of_selected]
        images = torch.stack([d[0] for d in selected_data], dim=0)
        labels = torch.LongTensor([d[1] for d in selected_data]) #torch.cat([d[1] for d in selected_data], dim=0)
        # Update list with replay buffers per task
        self.X_b.append(images)
        self.y_b.append(labels)

        #print(list_of_selected)

    def _update_distance(self, dists, x_train, current_id):
        x_train = x_train.view(x_train.size(0), -1)
        current_dist = torch.norm((x_train - torch.tile(x_train[current_id,:], [x_train.size(0), 1])), p=2, dim=1)
        dists = torch.minimum(current_dist, dists)
        return dists

    def reset_memory_from_task(self, task_id):
        if (task_id < len(self.X_b)) and (task_id < len(self.y_b)):
            print('curr buffer size: ', len(self.X_b) )
            n_tasks_to_reset = len(self.y_b) - task_id
            for _ in range(n_tasks_to_reset):
                self.X_b.pop(-1)
                self.y_b.pop(-1)
            print('new buffer size: ', len(self.X_b) )
