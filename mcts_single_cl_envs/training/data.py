
import os
import copy
import pickle
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset, Subset
import torch, torchvision


def get_multitask_experiment(config):
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
    scenario = config['training']['scenario']
    img_size = config['data']['img_size']
    total_num_classes = config['data']['n_classes'] 
    shuffle_labels = config['data']['shuffle_labels']
    #shuffle_seed = config['data']['shuffle_seed']
    pc_valid = config['data']['pc_valid'] # percentage of training data to validation set
    seed = config['session']['seed'] # used for train/val split

    # Sets parameters of the dataset. For adding new datasets, please add the dataset details in `get_statistics` function.
    #mean, std, total_num_classes, inp_size, in_channels = get_statistics(dataset)

    # Generates the standard data augmentation transforms, if needed for dataset
    #train_augment, test_augment = get_augment_transforms(dataset=dataset, inp_sz=inp_size)
    
    # Transforms
    if dataset in ['notMNIST']:
        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),])
    else:
        data_transforms = transforms.ToTensor()
    train_transforms = data_transforms
    test_transforms = data_transforms
    #train_transforms = transforms.Compose(train_augment + [transforms.ToTensor(),])
    #test_transforms = transforms.Compose(test_augment + [transforms.ToTensor(),])
    
    #train_transforms = transforms.Compose(train_augment + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    #test_transforms = transforms.Compose(test_augment + [transforms.ToTensor(), transforms.Normalize(mean, std)])

    """
    # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
    if shuffle_labels:
        rs = np.random.RandomState(shuffle_seed)
        labels = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        rs.shuffle(labels)
        permutation = labels.flatten()
        #permutation = rs.permutation(list(range(10))) #if shuffle_labels==True else  np.array(list(range(10)))
        target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
        print('label permutation: ', permutation)
    else:
    """
    target_transform = None

    # Get original datasets
    train_dataset = get_dataset(dataset, data_dir, train=True, transforms=train_transforms, target_transforms=target_transform)
    test_dataset = get_dataset(dataset, data_dir, train=False, transforms=test_transforms, target_transforms=target_transform)

    # Get fixed parameters for datasets
    if dataset in ['MNIST', 'CIFAR10', 'FashionMNIST', 'notMNIST', ]:
        n_tasks = 5
        classes_per_task = 2
    elif dataset in ['CIFAR100', 'miniImagenet',]:
        n_tasks = 20
        classes_per_task = 5
    elif dataset in ['GroceryStore',]:
        # use given n_tasks
        classes_per_task = int(np.floor(total_num_classes / n_tasks))

    # Get subdatasets for continual learning
    if dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'notMNIST', 'GroceryStore', 'miniImagenet']:
        # check for number of tasks
        if (n_tasks > total_num_classes):
            raise ValueError("Experiment %s cannot have more than %d tasks!" %(dataset, total_num_classes))
        #classes_per_task = int(np.floor(total_num_classes / n_tasks))
        # generate labels-per-task
        labels_per_task = [
            list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(n_tasks)
        ]
        print('Labels for each task: ', labels_per_task)
        # split them up into sub-tasks
        train_datasets = []
        test_datasets = []
        for labels in labels_per_task:
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y - x) if scenario=='domain' else None
            train_datasets.append(SubDataset(train_dataset, labels, target_transform=target_transform))
            test_datasets.append(SubDataset(test_dataset, labels, target_transform=target_transform))

    elif dataset == 'PermutedMNIST':
        # Reduce the training sets to 10k samples
        ind = np.random.permutation(len(train_dataset))[:10000]
        train_dataset = Subset(train_dataset, indices=ind)
        
        classes_per_task = 10 # same as original MNIST
        # generate permutations, first task is original MNIST
        permutations = [None] + [np.random.permutation(img_size**2) for _ in range(n_tasks-1)]
        # prepare datasets per task
        train_datasets = []
        test_datasets = []
        for task_id, perm in enumerate(permutations):
            target_transform = transforms.Lambda(lambda y, x=task_id: y + x*classes_per_task) if scenario in ('task', 'class') else None
            train_datasets.append(TransformedDataset(
                train_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
            test_datasets.append(TransformedDataset(
                test_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
    else:
        raise RuntimeError('Given undefined dataset: {}'.format(dataset))

    # Get validation set
    valid_datasets = []
    for task_id, train_set in enumerate(train_datasets):
        split = int(np.floor(pc_valid * len(train_set)))
        train_split, valid_split = torch.utils.data.random_split(train_set,
                                                                lengths=[len(train_set) - split, split],
                                                                generator=torch.Generator().manual_seed(seed))
        train_datasets[task_id] = train_split
        valid_datasets.append(valid_split)

    return train_datasets, valid_datasets, test_datasets, classes_per_task

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
    if dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'KMNIST', 'FashionMNIST']:
        return getattr(torchvision.datasets, dataset)(root=data_dir, train=train, download=True,
                                                        transform=transforms, target_transform=target_transforms)
    elif dataset =='PermutedMNIST':
        return getattr(torchvision.datasets, 'MNIST')(root=data_dir, train=train, download=True,
                                                        transform=transforms, target_transform=target_transforms)
    elif dataset=='SVHN':
        split = 'train' if train else 'test'
        return getattr(torchvision.datasets, dataset)(root=data_dir, split=split, download=True, transform=transforms, target_transform=target_transforms) 
    elif dataset == 'GroceryStore':
        split = 'train' if train else 'test'
        X = np.load(data_dir+'/'+dataset+'/' + '%s.features.resnet50.npy' %(split))
        y = np.load(data_dir+'/'+dataset+'/' + '%s.labels.resnet50.npy' %(split))
        X_norm = normalize_data(X) # normalize data to have unit norm
        return torch.utils.data.TensorDataset(torch.Tensor(X_norm), torch.LongTensor(y))
    else: # e.g. ['notMNIST', 'miniImagenet]
        subfolder = 'train' if train else 'test' # ImageNet 'val' is labled as 'test' here.
        return torchvision.datasets.ImageFolder(data_dir+'/'+dataset+'/'+subfolder, transform=transforms, target_transform=target_transforms)

#----------------------------------------------------------------------------------------------------------#

class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
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
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample

class TransformedDataset(Dataset):
    '''Modify existing dataset with transform; for creating multiple MNIST-permutations w/o loading data every time.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)

#----------------------------------------------------------------------------------------------------------#

def normalize_data(X):
    """
    Make each feature have unit norm (divide by L2 norm).
    :param X: data
    :return: normalized data

    Taken from https://github.com/tyler-hayes/ExStream/blob/a51235383b92f3939fbc23e90cf35caaa3d2176b/utils.py#L65
    """
    norm = np.kron(np.ones((X.shape[1], 1)), np.linalg.norm(X, axis=1)).T
    X = X / norm
    return X

################## PERMUTED MNIST CODE ##################

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


"""
def get_augment_transforms(dataset, inp_sz):
    
    #Returns appropriate augmentation given dataset size and name
    ##########Arguments:
        ###############3indices (sequence): a sequence of indices

    ### Taken from GDumb paper 
    ### https://github.com/drimpossible/GDumb/blob/master/src/dataloader.py
    
    if inp_sz == 32 or inp_sz == 28 or inp_sz == 64:
       train_augment = [torchvision.transforms.RandomCrop(inp_sz, padding=4)]
       test_augment = []
    else:
       train_augment = [torchvision.transforms.RandomResizedCrop(inp_sz)]
       test_augment = [torchvision.transforms.Resize(inp_sz+32), torchvision.transforms.CenterCrop(inp_sz)] 
    
    if dataset not in ['MNIST', 'SVHN', 'KMNIST']:
        train_augment.append(torchvision.transforms.RandomHorizontalFlip()) 

    return train_augment, test_augment

def get_statistics(dataset):
    '''
    #Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here

    ### Taken from GDumb paper 
    ### https://github.com/drimpossible/GDumb/blob/master/src/dataloader.py
    '''
    assert(dataset in ['MNIST', 'KMNIST', 'EMNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'CINIC10', 'ImageNet100', 'ImageNet', 'TinyImagenet'])
    mean = {
            'MNIST':(0.1307,),
            'KMNIST':(0.1307,),
            'EMNIST':(0.1307,),
            'FashionMNIST':(0.1307,),
            'SVHN':  (0.4377,  0.4438,  0.4728),
            'CIFAR10':(0.4914, 0.4822, 0.4465),
            'CIFAR100':(0.5071, 0.4867, 0.4408),
            'CINIC10':(0.47889522, 0.47227842, 0.43047404),
            'TinyImagenet':(0.4802, 0.4481, 0.3975),
            'ImageNet100':(0.485, 0.456, 0.406),
            'ImageNet':(0.485, 0.456, 0.406),
        }

    std = {
            'MNIST':(0.3081,),
            'KMNIST':(0.3081,),
            'EMNIST':(0.3081,),
            'FashionMNIST':(0.3081,),
            'SVHN': (0.1969,  0.1999,  0.1958),
            'CIFAR10':(0.2023, 0.1994, 0.2010),
            'CIFAR100':(0.2675, 0.2565, 0.2761),
            'CINIC10':(0.24205776, 0.23828046, 0.25874835),
            'TinyImagenet':(0.2302, 0.2265, 0.2262),
            'ImageNet100':(0.229, 0.224, 0.225),
            'ImageNet':(0.229, 0.224, 0.225),
        }

    classes = {
            'MNIST': 10,
            'KMNIST': 10,
            'EMNIST': 49,
            'FashionMNIST': 10,
            'SVHN': 10,
            'CIFAR10': 10,
            'CIFAR100': 100,
            'CINIC10': 10,
            'TinyImagenet':200,
            'ImageNet100':100,
            'ImageNet': 1000,
        }

    in_channels = {
            'MNIST': 1,
            'KMNIST': 1,
            'EMNIST': 1,
            'FashionMNIST': 1,
            'SVHN': 3,
            'CIFAR10': 3,
            'CIFAR100': 3,
            'CINIC10': 3,
            'TinyImagenet':3,
            'ImageNet100':3,
            'ImageNet': 3,
        }

    inp_size = {
            'MNIST': 28,
            'KMNIST': 28,
            'EMNIST': 28,
            'FashionMNIST': 28,
            'SVHN': 32,
            'CIFAR10': 32,
            'CIFAR100': 32,
            'CINIC10': 32,
            'TinyImagenet':64,
            'ImageNet100':224,
            'ImageNet': 224,
        }
    return mean[dataset], std[dataset], classes[dataset],  inp_size[dataset], in_channels[dataset]
"""

#----------------------------------------------------------------------------------------------------------#
"""
################## PERMUTED MNIST CODE ##################

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
"""



"""
# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
}


### Code that would go into get_multitask_experiment if we would like to use Permuted MNIST
# depending on experiment, get and organize the datasets
if name == 'permMNIST':
    # configurations
    config = DATASET_CONFIGS['mnist']
    classes_per_task = 10
    if not only_config:
        # prepare dataset
        train_dataset = get_dataset('mnist', type="train", permutation=None, dir=data_dir,
                                    target_transform=None, verbose=verbose)
        test_dataset = get_dataset('mnist', type="test", permutation=None, dir=data_dir,
                                    target_transform=None, verbose=verbose)
        # generate permutations
        if exception:
            permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
        else:
            permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
        # prepare datasets per task
        train_datasets = []
        test_datasets = []
        for task_id, perm in enumerate(permutations):
            target_transform = transforms.Lambda(
                lambda y, x=task_id: y + x*classes_per_task
            ) if scenario in ('task', 'class') else None
            train_datasets.append(TransformedDataset(
                train_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
            test_datasets.append(TransformedDataset(
                test_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))

"""