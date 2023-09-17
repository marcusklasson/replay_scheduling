
import torch
from torch.utils.data import DataLoader

def get_data_loader(dataset, batch_size, num_workers, pin_memory, shuffle, rng=None):
    if rng is None:
        data_loader = DataLoader(dataset, 
                                batch_size=batch_size,
                                num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, 
                                batch_size=batch_size,
                                num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                shuffle=shuffle,
                                generator=rng)
    return data_loader

def select_valid_outputs(output, t, n_classes_per_task):
    #https://github.com/imirzadeh/CL-Gym/blob/main/cl_gym/backbones/base.py
    for i, tt in enumerate(t):
        offset1 = int((tt) * n_classes_per_task)
        offset2 = int((tt+1) * n_classes_per_task)
        output[i, :offset1].data.fill_(-10e10)
        output[i, offset2:].data.fill_(-10e10)
    return output

