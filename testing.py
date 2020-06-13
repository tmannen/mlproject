import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy

import create_dataset

import models
from train_test import train_model

if __name__ == '__main__': 
    """
    what to predict:

    - maybe x and y coordinate from 0 to 9? ground truth would be the keys to the hdf5 file.
    - direction would just be south, east etc. with numbers 0-3 maybe.
    - randomly skip some squares to get test set?
    """



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets = create_dataset.create_dataset()

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    model_ft = models.create_basic_model(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, device,
                        num_epochs=25)

    torch.save(model_ft.state_dict(), "models/basic_model.pt")