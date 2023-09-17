
import yaml
from torch import optim
from os import path

import torch
import torch.nn as nn
import numpy as np

from training.models import mlp
from training.models import convnet

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

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(m.bias)

def my_weights_init_uniform(net):
    weight_dict = net.state_dict()
    new_weight_dict = {}
    for param_key in weight_dict:
        # custom initialization in new_weight_dict,
        # You can initialize partially i.e only some of the variables and let others stay as it is
        with torch.no_grad():
            if 'weight' in param_key:
                w = weight_dict[param_key]
                stdv = 1. / np.sqrt(w.size(1))
                new_weight_dict[param_key] = w.uniform_(-stdv, stdv)
                b_param_key = param_key.replace('weight', 'bias')
                b = weight_dict[b_param_key]
                new_weight_dict[b_param_key] = b.uniform_(-stdv, stdv)
    net.load_state_dict(new_weight_dict)
    return net

def build_models(args):

    net_name = args.cl.net
    # Build models
    if 'mlp' in net_name:
        model = mlp.MLP(
            out_dim=args.cl.n_classes,
            img_sz=args.input_size[1],
            in_channel=args.input_size[0],
            n_layers=args.cl.n_layers,
            hidden_dim=args.cl.units,
            )
    elif 'convnet' in net_name:
        input_dim_classifier = 256 # for CIFAR10/100. If Omniglot then 64
        model = convnet.ConvNet(
            num_input_channels=3,#args.inputsize[0],
            k_way=args.cl.n_classes,
            final_layer_size=input_dim_classifier, # this will have to adjusted if its Omniglot or CIFAR10/100
            )
    else:
        raise RuntimeError('Given undefined model: {}'.format(net_name))
    return model


def build_models_using_rng(args, rng):

    net_name = args.cl.net

    def weights_init_uniform_with_rng(m):
        if isinstance(m, nn.Linear):
            stdv = 1. / np.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv, generator=rng)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv, generator=rng)

    # Build models
    if 'mlp' in net_name:
        model = mlp.MLP(
            out_dim=args.cl.n_classes,
            img_sz=args.input_size[1],
            in_channel=args.input_size[0],
            n_layers=args.cl.n_layers,
            hidden_dim=args.cl.units,
            )
        model.apply(weights_init_uniform_with_rng)
        """
        for name, param in model.named_parameters():
            print(name, param)
            break
        print()
        """
    else:
        raise RuntimeError('Given undefined model: {}'.format(net_name))
    return model

def build_optimizers(net, args):
    optimizer = args.cl.optimizer
    lr = args.cl.lr 

    toggle_grad(net, True)
    params = net.parameters()

    # Optimizers
    if optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-8)
    elif optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer == 'sgd':
        #weight_decay = config['training']['weight_decay']
        #momentum = config['training']['momentum']
        optimizer = optim.SGD(params, lr=lr,) #weight_decay=weight_decay, momentum=momentum)

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