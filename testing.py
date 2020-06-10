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

import models
from train_test import train_model

if __name__ == '__main__': 
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

    images = torch.Tensor(np.concatenate(images, axis=0)).permute(0, 3, 1, 2)
    targets_index = torch.LongTensor(np.concatenate(targets_index))
    targets_xy = torch.Tensor(np.concatenate(targets_xy))
    targets_direction = torch.Tensor(np.concatenate(targets_direction))

    step = len(images)//10
    train_idxs = np.arange(8*step)
    val_idxs = np.arange(8*step, 9*step)
    test_idxs = np.arange(9*step, step*10)


    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {}
    image_datasets['train'] = TensorDataset(images[train_idxs], targets_index[train_idxs])
    image_datasets['val'] = TensorDataset(images[val_idxs], targets_index[val_idxs])
    image_datasets['test'] = TensorDataset(images[test_idxs], targets_index[test_idxs])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.create_basic_model(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, device,
                        num_epochs=25)

    torch.save(model_ft.state_dict(), "models/basic_model.pt")