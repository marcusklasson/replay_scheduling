# https://github.com/imirzadeh/CL-Gym/blob/302bf7f0d9a96df61c32dec1dd4cacd19746179d/cl_gym/algorithms/utils.py#L22

import numpy as np
import torch
from torch.utils.data import DataLoader

def flatten_grads(model):
    # Reorganize the gradient of a model (e.g. after '.backward()' call) as a single vector
    all_grads = []
    for p in model.parameters():
        if p.requires_grad:
            all_grads.append(p.grad.view(-1))
    return torch.cat(all_grads)

def assign_grads(model, grads):
    """
    Similar to `assign_weights` but this time, manually assign `grads` vector to a model.
    :param model: PyTorch Model.
    :param grads: Gradient vectors.
    :return:
    """
    state_dict = model.state_dict(keep_vars=True)
    index = 0
    for param in state_dict.keys():
        # ignore batchnorm params
        if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
            continue
        param_count = state_dict[param].numel()
        param_shape = state_dict[param].shape
        state_dict[param].grad = grads[index:index+param_count].view(param_shape).clone()
        index += param_count
    model.load_state_dict(state_dict)
    return model

def update_reservoir(current_image, current_label, episodic_images, episodic_labels, M, N):
    """
    Update the episodic memory with current example using the reservoir sampling
    """
    if M > N:
        episodic_images[N] = current_image
        episodic_labels[N] = current_label
    else:
        j = np.random.randint(0, N)
        if j < M:
           episodic_images[j] = current_image
           episodic_labels[j] = current_label

def update_reservoir_der(current_image, current_label, current_logit,
                        episodic_images, episodic_labels, episodic_logits, M, N):
    """
    Update the episodic memory with current example using the reservoir sampling
    """
    if M > N:
        episodic_images[N] = current_image
        episodic_labels[N] = current_label
        episodic_logits[N] = current_logit
    else:
        j = np.random.randint(0, N)
        if j < M:
            episodic_images[j] = current_image
            episodic_labels[j] = current_label
            episodic_logits[j] = current_logit

def update_fifo_buffer(current_images, current_labels, episodic_images, episodic_labels, count_cls, 
                        memories_per_class, episodic_filled_counter, cl_scenario):
    for er_x, er_y in zip(current_images, current_labels):
        label = er_y.item()
        # Write the example at the location pointed by count_cls[label]
        with_in_task_offset = memories_per_class * label
        if cl_scenario == 'domain':
            mem_index = count_cls[label] + with_in_task_offset + episodic_filled_counter
        else:
            mem_index = count_cls[label] + with_in_task_offset 
        episodic_images[mem_index] = er_x
        episodic_labels[mem_index] = er_y
        count_cls[label] = (count_cls[label] + 1) % memories_per_class

def update_fifo_buffer_der(current_images, current_labels, current_logits, 
                            episodic_images, episodic_labels, episodic_logits, count_cls, 
                        memories_per_class, episodic_filled_counter, cl_scenario):
    for er_x, er_y, er_logits in zip(current_images, current_labels, current_logits):
        label = er_y.item()
        # Write the example at the location pointed by count_cls[label]
        with_in_task_offset = memories_per_class * label
        if cl_scenario == 'domain':
            mem_index = count_cls[label] + with_in_task_offset + episodic_filled_counter
        else:
            mem_index = count_cls[label] + with_in_task_offset 
        episodic_images[mem_index] = er_x
        episodic_labels[mem_index] = er_y
        episodic_logits[mem_index] = er_logits
        count_cls[label] = (count_cls[label] + 1) % memories_per_class

#def zeropad_list(l, size, padding=0.0):
#    return l + [padding] * abs((len(l)-size))

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


def pre_select_memory_inds(summarizer, datasets, memories_per_class, method='uniform'):
    chosen_inds = []
    for dataset in datasets:
        X = np.stack([img.numpy() for img, _ in dataset], axis=0)
        y = np.stack([label for _, label in dataset], axis=0)
        for y_ in np.unique(y):
            er_x = X[y == y_] 
            er_y = y[y == y_]
            chosen_inds_y_ = summarizer.build_summary(er_x, er_y, memories_per_class, method=method)
            chosen_inds.append(chosen_inds_y_)
    return chosen_inds 
