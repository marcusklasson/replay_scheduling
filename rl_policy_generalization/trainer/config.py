
import torch 
from torch import optim

from trainer.networks.mlp import MLP
from trainer.networks.convnet import ConvNet
from trainer.networks.resnet import ResNet18

def build_models(args):

    net_name = args.cl.net
    # Build models
    if 'mlp' in net_name:
        model = MLP(out_dim=args.cl.n_classes, in_channel=1, img_sz=28, 
                        n_layers=args.cl.n_layers, hidden_dim=args.cl.units,
                        multi_head=args.cl.multi_head, 
                        n_classes_per_task=args.classes_per_task) 
    elif 'convnet' in net_name:
        input_dim_classifier = 256 # for CIFAR10/100. If Omniglot then 64
        model = ConvNet(
            num_input_channels=3,#args.inputsize[0],
            k_way=args.cl.n_classes,
            final_layer_size=input_dim_classifier, # this will have to adjusted if its Omniglot or CIFAR10/100
            multi_head=args.cl.multi_head,
            n_classes_per_task=args.classes_per_task)
    elif 'resnet18' in net_name:
        model = ResNet18(
            num_classes=args.cl.n_classes,
            nf=20, # reduced number of filters
            multi_head=args.cl.multi_head,
            n_classes_per_task=args.classes_per_task)
    else:
        raise RuntimeError('Given undefined model: {}'.format(net_name))
    return model

def build_optimizers(net, config):
    optimizer = config.cl.optimizer
    lr = config.cl.lr 

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

    if config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_anneal_every'],
            gamma=config['lr_anneal'],
            last_epoch=last_epoch
        )
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['n_epochs'])
    elif config['lr_scheduler'] == 'cyclic':
        cycle_momentum = False if 'momentum' not in optimizer.defaults else True
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=config['base_lr'], 
            max_lr=config['max_lr'],
            step_size_up=config['step_size_up'],
            mode=config['cyclic_lr_mode'],
            cycle_momentum=cycle_momentum)
    else:
        return None

    return scheduler

# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)