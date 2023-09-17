

import torch
import torch.nn as nn

from trainer.utils import select_valid_outputs

class MLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=28, n_layers=2, hidden_dim=256,
                    multi_head=False, n_classes_per_task=2):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.multi_head = multi_head
        self.n_classes_per_task = n_classes_per_task
        lower_modules = []
        in_dim = self.in_dim
        for i in range(n_layers): 
            lower_modules.append(nn.Linear(in_dim, hidden_dim))
            lower_modules.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.net = nn.Sequential(*lower_modules)
        self.linear = nn.Linear(in_dim, out_dim)

    def embed(self, x):
        x = self.net(x.view(-1, self.in_dim))
        return x        

    def forward(self, x, t=None):
        x = self.embed(x)
        x = self.linear(x)
        if self.multi_head and (t is not None):
            x = select_valid_outputs(x, t, self.n_classes_per_task)
        return x

def MLP400(out_dim=10, in_channel=1, img_sz=28, n_layers=2):
    return MLP(out_dim, in_channel, img_sz, n_layers, hidden_dim=400)

def MLP256(out_dim=10, in_channel=1, img_sz=28, n_layers=2):
    return MLP(out_dim, in_channel, img_sz, n_layers, hidden_dim=256)

def MLP150(out_dim=10, in_channel=1, img_sz=28, n_layers=2):
    return MLP(out_dim, in_channel, img_sz, n_layers, hidden_dim=150)

def MLP100(out_dim=10, in_channel=1, img_sz=28, n_layers=2):
    return MLP(out_dim, in_channel, img_sz, n_layers, hidden_dim=100)
