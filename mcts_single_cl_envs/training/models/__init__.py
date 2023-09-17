from training.models import (mlp, convnet, resnet)

model_dict = {
    'mlp400': mlp.MLP400,
    'mlp256': mlp.MLP256,
    'mlp150': mlp.MLP150,
    'mlp100': mlp.MLP100,
    'convnet': convnet.ConvNet, # 4 3x3 conv blocks with Relu and MaxPool, before classification layer
    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34,
}