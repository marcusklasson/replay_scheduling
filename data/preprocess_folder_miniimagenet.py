import os
import shutil
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

np.random.seed(1234)

# we want 500 for training, 100 for test for each class
n = 500

root_dir = './data/mini_imagenet/'
target_dir = './src/datasets/miniImagenet'
if not os.path.exists(os.path.join(target_dir, 'train')):
    os.makedirs(os.path.join(target_dir, 'train'))
if not os.path.exists(os.path.join(target_dir, 'test')):
    os.makedirs(os.path.join(target_dir, 'test'))


images = []
labels = []
data = {'images': [], 'labels': []}
class_dict = {}
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        filepath = subdir + os.sep + file
        images.append(filepath)
        label = subdir.split('/')[-1]
        if label not in class_dict.keys():
            class_dict[label] = []
        class_dict[label].append(filepath)

# split images into train/test and copy images to corresponding class dir in train/test
for k, v in class_dict.items():
    print(k)
    v = np.array(v)
    np.random.shuffle(v)
    x_train = v[:n]
    if not os.path.exists(os.path.join(target_dir, 'train', str(k))):
        os.makedirs(os.path.join(target_dir, 'train', str(k)))
    dst = os.path.join(target_dir, 'train', str(k))
    for filepath in x_train:
        shutil.copy2(filepath, dst)


    x_test = v[n:]
    if not os.path.exists(os.path.join(target_dir, 'test', str(k))):
        os.makedirs(os.path.join(target_dir, 'test', str(k)))
    dst = os.path.join(target_dir, 'test', str(k))
    for filepath in x_test:
        shutil.copy2(filepath, dst)
    
