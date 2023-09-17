"""
Grabbed this code from https://github.com/oscarknagg/few-shot/blob/master/few_shot/models.py
which is the model used in "Matching Networks for OneShot Learning" by Vinyals et al. (2016).

This ConvNet was also used in CLAW paper by Adel et al. (ICLR 2020) and many other
Continual Learning models used for Omniglot dataset.
"""

import torch
import torch.nn as nn

from trainer.utils import select_valid_outputs

def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class ConvNet(nn.Module):
    def __init__(self, num_input_channels: int, k_way: int, final_layer_size: int = 64,
                    multi_head: bool =False, n_classes_per_task: int = 5):
        """Creates a few shot classifier as used in MAML.
        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.
        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 256 for CIFAR10/100, 1600 for miniImageNet
        """
        super(ConvNet, self).__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)

        self.logits = nn.Linear(final_layer_size, k_way)

        self.multi_head = multi_head
        self.n_classes_per_task = n_classes_per_task

    def embed(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.view(x.size(0), -1)

    def forward(self, x, t=None):
        x = self.embed(x)
        out = self.logits(x)
        if self.multi_head and (t is not None):
            out = select_valid_outputs(out, t, self.n_classes_per_task)
        return out