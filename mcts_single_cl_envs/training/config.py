
import yaml
from torch import optim
from os import path
from training.models import model_dict
#from training.train import toggle_grad

# General config
def load_config(path, default_path):
    ''' Loads config file.
    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        # Add item if not yet in dict1
        if k not in dict1:
            dict1[k] = None
        # Update
        if isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def build_models(config):

    dataset = config['data']['name']
    net_name = config['model']['net']
    # Get classes
    ModelClass = model_dict[config['model']['net']]
    # Build models
    if 'mlp' in net_name:
        model = ModelClass(
            out_dim=config['data']['n_classes'],
            in_channel=config['data']['in_channel'],
            img_sz=config['data']['img_size'],
            n_layers=config['model']['n_layers'],
            )

    elif 'convnet' in net_name:
        input_dim_classifier = 256 if (dataset in ['CIFAR10', 'CIFAR100']) else 64
        model = ModelClass(
            num_input_channels=config['data']['in_channel'],
            k_way=config['data']['n_classes'],
            final_layer_size=input_dim_classifier, # this will have to adjusted if its Omniglot or CIFAR10/100
            )

    elif 'resnet18' in net_name:
        if dataset in ['CIFAR10', 'CIFAR100']:
            input_dim_classifier = 160
        elif dataset in ['miniImagenet',]:
            input_dim_classifier = 640
        model = ModelClass(
            num_classes=config['data']['n_classes'],
            nf=20,
            input_size_linear=input_dim_classifier,
            )
    else:
        raise RuntimeError('Given undefined model: {}'.format(net_name))
    return model

def build_optimizers(net, config):
    opt = config['training']['optimizer']
    lr = config['training']['lr']

    toggle_grad(net, True)
    params = net.parameters()

    # Optimizers
    if opt == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-8)
    elif opt == 'adam':
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif opt == 'sgd':
        optimizer = optim.SGD(params, lr=lr)
    elif opt == 'momentum':
        momentum = config['training']['momentum']
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    else:
        raise ValueError('Optimizer {} is not an option right now.'.format(opt))

    return optimizer

def build_lr_scheduler(optimizer, config, last_epoch=-1):

    if config['training']['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_anneal_every'],
            gamma=config['training']['lr_anneal'],
            last_epoch=last_epoch
        )
    elif config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['training']['n_epochs'])

    else:
        return None

    return scheduler

# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)