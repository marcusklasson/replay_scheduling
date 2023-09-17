
import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=28, n_layers=2, hidden_dim=400):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        lower_modules = []
        in_dim = self.in_dim
        for i in range(n_layers): 
            lower_modules.append(nn.Linear(in_dim, hidden_dim))
            lower_modules.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.net = nn.Sequential(*lower_modules)
        self.last = nn.Linear(in_dim, out_dim)

    def features(self, x):
        x = self.net(x.view(-1, self.in_dim))
        return x

    def embed(self, x): # for data summary in trainer.summary
        return self.features(x)

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def get_params(self):
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                    + 100 * output_size + output_size)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params):
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (input_size * 100
                    + 100 + 100 * 100 + 100 + 100 * output_size + output_size)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress + torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

def MLP400(out_dim=10, in_channel=1, img_sz=28, n_layers=2):
    return MLP(out_dim, in_channel, img_sz, n_layers, hidden_dim=400)

def MLP256(out_dim=10, in_channel=1, img_sz=28, n_layers=2):
    return MLP(out_dim, in_channel, img_sz, n_layers, hidden_dim=256)

def MLP150(out_dim=10, in_channel=1, img_sz=28, n_layers=2):
    return MLP(out_dim, in_channel, img_sz, n_layers, hidden_dim=150)

def MLP100(out_dim=10, in_channel=1, img_sz=28, n_layers=2):
    return MLP(out_dim, in_channel, img_sz, n_layers, hidden_dim=100)