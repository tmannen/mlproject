import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy

from train_test import train_model

datas = [h5py.File(f"data//{f}", 'r') for f in os.listdir("data") if "hdf5" in f]

"""
what to predict:

- maybe x and y coordinate from 0 to 9? ground truth would be the keys to the hdf5 file.
- direction would just be south, east etc. with numbers 0-3 maybe.
- randomly skip some squares to get test set?
"""

rows = 10
columns = 10

directions = ['north', 'east', 'south', 'west']
dir_to_number = {
                'north':0, 
                'east': 1, 
                'south': 2, 
                'west': 3
                }

def index_to_xy(index):
    """ Dataset index to xy coordinates """
    x = index % columns
    y = index // columns
    return x, y

def xy_to_index(x, y):
    """ xy coordinates to dataset index """
    index = y * columns + x
    return index

def display_images(data, x, y):
    index = xy_to_index(x, y)
    sensors = data['sensors']

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(sensors['north'][index])
    axarr[0,1].imshow(sensors['east'][index])
    axarr[1,0].imshow(sensors['south'][index])
    axarr[1,1].imshow(sensors['west'][index])
    plt.show()

images = []
targets_index = []
targets_xy = []
targets_direction = []

for data in datas:
    sensors = data['sensors']
    for d in directions:
        images.append(sensors[d][()])
        idx = np.arange(100)
        x_target = idx % columns
        y_target = idx // columns
        xy_targets = np.vstack([x_target, y_target]).T
        targets_index.append(idx)
        targets_xy.append(xy_targets)
        dirs = np.repeat(dir_to_number[d], 100)
        targets_direction.append(dirs)

images = np.concatenate(images, axis=0)
targets_index = np.concatenate(targets_index)
targets_xy = np.concatenate(targets_xy)
targets_direction = np.concatenate(targets_direction)

train_idxs = np.arange(3600)
val_idxs = np.arange(3600, 4000)
test_idxs = np.arange(4000, 4400)

