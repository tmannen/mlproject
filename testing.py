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
import argparse

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--phase', type=str, default="train")

    args = parser.parse_args()
    model_name = args.model
    phase = args.phase

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amount = 30
    if model_name in ['viewpool', 'multiple']:
        image_datasets = create_dataset.create_dataset(multiple=True, amount=amount)
    else:
        image_datasets = create_dataset.create_dataset(amount=amount)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    print(dataset_sizes)
    if model_name == 'viewpool':
        model_ft = models.MVCNN_max()
    elif model_name == 'multiple':
        model_ft = models.MVCNN()
    else:
        model_ft = models.create_basic_model(device)

    model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    if phase == "train":
        model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, device, model_name,
                            num_epochs=20)

        torch.save(model_ft.state_dict(), f"models/{model_name}.pt")
    else:
        model_ft.load_state_dict(torch.load(f"models/{model_name}.pt"))
        model_ft.eval()
        preds = []
        gts = []
        for im, trgt in image_datasets['test']:
            pred = torch.argmax(model_ft(im.unsqueeze(0).to(device)))
            preds.append(pred.to('cpu'))
            gts.append(trgt)
        
        preds = torch.Tensor(preds)
        gts = torch.Tensor(gts)
        correct = preds == gts
        correct_sum = torch.sum(correct).to(dtype=torch.float)

        print("%f of datapoints were accurately predicted." % (correct_sum/len(preds)).item())