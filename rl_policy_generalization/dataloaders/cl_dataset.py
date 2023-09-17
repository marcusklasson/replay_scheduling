
import os
import random 
import copy
import pickle
import numpy as np
import torch, torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset, Subset
import torchvision.transforms.functional as TF


class ContinualDataset(object):

    def __init__(self, config):
        """ Get training and test datasets for continual learning experiments. 
            Args:
                config (dict): Configuration file for experiment.
            Returns:
                train_datasets (list): Training datasets
                test_datasets (list): Test datasets
                classes_per_task (int): Number of classes per task in dataset.
        """
        dataset = config['data']['name']
        data_dir = config['data']['data_dir']
        n_tasks = config['data']['n_tasks']
        classes_per_task = config['data']['classes_per_task']
        scenario = config['cl_scenario']
        img_size = config['data']['img_size']
        n_classes = config['data']['n_classes'] 
        shuffle_labels = config['data']['shuffle_labels']
        pc_valid = config['data']['pc_valid'] # percentage of training data to validation set
        n_samples = config['data']['n_samples'] # number of training samples

        seed = config['seed'] # used for train/val split
        dataset_seed = config['data']['seed'] # used for label shuffling

        # Transforms
        if dataset in ['notMNIST']:
            data_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),])
        else:
            data_transforms = transforms.ToTensor()
        train_transforms = data_transforms
        test_transforms = data_transforms

        # Get fixed parameters for datasets
        if n_tasks is None: 
            if dataset in ['MNIST', 'CIFAR10', 'FashionMNIST', 'notMNIST', ]:
                n_tasks = 5
                classes_per_task = 2
                total_num_classes = 10
            elif dataset in ['CIFAR100', 'miniImagenet',]:
                n_tasks = 20
                classes_per_task = 5
                total_num_classes = 100

        target_transform = None
        # Get original datasets
        train_dataset = get_dataset(dataset, data_dir, train=True, transforms=train_transforms, target_transforms=target_transform)
        test_dataset = get_dataset(dataset, data_dir, train=False, transforms=test_transforms, target_transforms=target_transform)

        # Get subdatasets for continual learning
        if dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'notMNIST', 'miniImagenet']:
            # check for number of tasks
            if (n_tasks > n_classes):
                raise ValueError("Experiment %s cannot have more than %d tasks!" %(dataset, total_num_classes))
            # generate labels-per-task
            if shuffle_labels:
                rs = np.random.RandomState(dataset_seed)
                labels = list(rs.permutation(n_classes))
                class_mapping = dict(zip(labels, list(np.arange(n_classes))))
                print('class_mapping: ', class_mapping)
                labels_per_task = [labels[i:i + classes_per_task] for i in range(0, len(labels), classes_per_task)]
                self.class_mapping = class_mapping
            else:
                labels_per_task = [
                    list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(n_tasks)
                ]
            #random.shuffle(labels_per_task)
            print('Labels for each task: ', labels_per_task)
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for task_id, labels in enumerate(labels_per_task):
                # get target transforms
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y - x) if scenario=='domain' else None
                if shuffle_labels:
                    target_transform = transforms.Lambda(lambda y: class_mapping[y])
                train_datasets.append(SubDataset(train_dataset, labels, task_id, target_transform=target_transform))
                test_datasets.append(SubDataset(test_dataset, labels, task_id, target_transform=target_transform))

        elif dataset == 'PermutedMNIST':
            # Reduce the training sets to X samples
            ind = np.random.permutation(len(train_dataset))[:n_samples] if n_samples is not None else np.arange(len(train_dataset))
            train_dataset = Subset(train_dataset, indices=ind)

            labels_per_task = [list(np.array(range(classes_per_task))) for task_id in range(n_tasks)]
            
            classes_per_task = 10 # same as original MNIST
            # generate permutations, first task is original MNIST
            permutations = [None] + [np.random.permutation(img_size**2) for _ in range(n_tasks-1)]
            if shuffle_labels:
                rs = np.random.RandomState(dataset_seed)
                perm_tasks = list(rs.permutation(len(permutations)))
                permutations = [permutations[i] for i in perm_tasks] #list(permutations)
                self.task_shuffle = perm_tasks
            else:
                self.task_shuffle = list(range(len(permutations)))

            # prepare datasets per task
            train_datasets = []
            test_datasets = []
            for task_id, perm in enumerate(permutations):
                target_transform = transforms.Lambda(lambda y, x=task_id: y + x*classes_per_task) if scenario in ('task', 'class') else None
                train_datasets.append(TransformedDataset(
                    train_dataset, task_id, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
                test_datasets.append(TransformedDataset(
                    test_dataset, task_id, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))

        elif dataset == 'RotatedMNIST':
            # Reduce the training sets to X samples
            ind = np.random.permutation(len(train_dataset))[:n_samples] if n_samples is not None else np.arange(len(train_dataset))
            train_dataset = Subset(train_dataset, indices=ind)
            
            per_task_rotation = 180.0 / n_tasks 
            labels_per_task = [list(np.array(range(classes_per_task))) for task_id in range(n_tasks)]
            # prepare datasets per task
            train_datasets = []
            test_datasets = []
            rotations = [RotationTransform(t * per_task_rotation) for t in range(n_tasks)]
            if shuffle_labels:
                rs = np.random.RandomState(dataset_seed)
                perm_tasks = rs.permutation(len(rotations))
                rotations = [rotations[i] for i in perm_tasks]#list(rotations)
                self.task_shuffle = perm_tasks
            else:
                self.task_shuffle = list(range(len(rotations)))

            for task_id, rot in enumerate(rotations):
                target_transform = transforms.Lambda(lambda y, x=task_id: y + x*classes_per_task) if scenario in ('task', 'class') else None
                train_datasets.append(TransformedDataset(
                    train_dataset, task_id, transform=rot, target_transform=target_transform
                ))
                test_datasets.append(TransformedDataset(
                    test_dataset, task_id, transform=rot, target_transform=target_transform
                ))
            
        else:
            raise RuntimeError('Given undefined dataset: {}'.format(dataset))

        # Get validation set
        valid_datasets = []
        for task_id, train_set in enumerate(train_datasets):
            split = int(np.floor(pc_valid * len(train_set)))
            train_split, valid_split = torch.utils.data.random_split(train_set,
                                                                    lengths=[len(train_set) - split, split],
                                                                    generator=torch.Generator().manual_seed(dataset_seed))
            train_datasets[task_id] = train_split
            valid_datasets.append(valid_split)

        # Keep datasets
        self.train_set = train_datasets 
        self.valid_set = valid_datasets
        self.test_set = test_datasets
        # save other args
        self.task_ids = labels_per_task
        self.n_tasks = n_tasks 
        self.input_size = img_size 
        self.classes_per_task = classes_per_task
        self.dataset = dataset
        self.dataset_seed = dataset_seed
        # change task ids if shuffle labels is enabled
        if shuffle_labels:
            self.shuffle_labels = True
            self.class_mapping = None if dataset in ['PermutedMNIST', 'RotatedMNIST'] else class_mapping
        else:
            self.shuffle_labels = False 
            self.class_mapping = None             

    def get_dataset_for_task(self, task_id):
        dataset = {}
        dataset['train'] = self.train_set[task_id]
        dataset['valid'] = self.valid_set[task_id]
        dataset['test'] = self.test_set[task_id]
        if self.dataset in ['PermutedMNIST', 'RotatedMNIST']:
            dataset['name'] = '{}Task-{}-{}'.format(self.n_tasks, self.dataset, task_id)#, self.task_ids[task_id])
        else:
            dataset['name'] = '{}Split-{}-{}-{}'.format(self.n_tasks, self.dataset, task_id, self.task_ids[task_id])
        return dataset


def get_dataset(dataset, data_dir, train, transforms, target_transforms=None):
    """ Get torchvision datasets.
        Args:
            dataset (str): Dataset name.
            data_dir (str): Directory where dataset is downloaded to.
            train (bool): Get training or test data?
            transforms (transforms.Compose): Composition of data transforms. 
            target_transforms (...): Transforms of target label.
        Returns:
            (torch.Dataset): The wanted dataset.
    """

    # Support for *some* pytorch default loaders is provided. Code is made such that adding new datasets is super easy, given they are in ImageFolder format.        
    if dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']:
        return getattr(torchvision.datasets, dataset)(root=data_dir, train=train, download=True,
                                                        transform=transforms, target_transform=target_transforms)
    elif dataset in ['PermutedMNIST', 'RotatedMNIST']:
        return getattr(torchvision.datasets, 'MNIST')(root=data_dir, train=train, download=True,
                                                        transform=transforms, target_transform=target_transforms)
    elif dataset=='SVHN':
        split = 'train' if train else 'test'
        return getattr(torchvision.datasets, dataset)(root=data_dir, split=split, download=True, transform=transforms, target_transform=target_transforms) 
    else: # e.g. ['notMNIST', 'miniImagenet]
        subfolder = 'train' if train else 'test' # ImageNet 'val' is labled as 'test' here.
        return torchvision.datasets.ImageFolder(data_dir+'/'+dataset+'/'+subfolder, transform=transforms, target_transform=target_transforms)

#----------------------------------------------------------------------------------------------------------#

class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, task_id, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        self.task_id = task_id 
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        (inp, target) = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(target)
            #sample = (sample[0], target)
        sample = (inp, target) #sample = (inp, target, self.task_id)
        return sample

class TransformedDataset(Dataset):
    '''Modify existing dataset with transform; for creating multiple MNIST-permutations w/o loading data every time.'''

    def __init__(self, original_dataset, task_id, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.task_id = task_id 
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (inp, target) = self.dataset[index]
        if self.transform:
            inp = self.transform(inp)
        if self.target_transform:
            target = self.target_transform(target)
        sample = (inp, target) #sample = (inp, target, self.task_id)
        return sample #(input, target, self.task_id) # (input, target)

class RotationTransform:
    """
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """
    def __init__(self, angle):
        self.angle = angle
    
    def __call__(self, x):
        return TF.rotate(x, self.angle, fill=(0,))

def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image