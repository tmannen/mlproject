import h5py
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, Dataset
from torchvision import datasets, models, transforms

class ImageDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.fromarray(self.images[idx])
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target

def index_to_xy(index):
    """ Dataset index to xy coordinates """
    x = index % columns
    y = index // columns
    return x, y

def xy_to_index(x, y):
    """ xy coordinates to dataset index """
    index = y * columns + x
    return index

def create_dataset():
    # Data augmentation and normalization for training
    # Just normalization for validation
    datas = [h5py.File(f"data//{f}", 'r') for f in os.listdir("data") if "hdf5" in f]

    rows = 10
    columns = 10

    directions = ['north', 'east', 'south', 'west']
    dir_to_number = {
                    'north':0, 
                    'east': 1, 
                    'south': 2, 
                    'west': 3
                    }

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
    targets_index = torch.LongTensor(np.concatenate(targets_index))
    targets_xy = torch.Tensor(np.concatenate(targets_xy))
    targets_direction = torch.Tensor(np.concatenate(targets_direction))

    data_transforms = {
        'train': transforms.Compose([
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    step = len(images)//10
    train_idxs = np.arange(8*step)
    val_idxs = np.arange(8*step, 9*step)
    test_idxs = np.arange(9*step, step*10)

    image_datasets = {}
    image_datasets['train'] = ImageDataset(images[train_idxs], targets_index[train_idxs], data_transforms['train'])
    image_datasets['val'] = ImageDataset(images[val_idxs], targets_index[val_idxs], data_transforms['val'])
    image_datasets['test'] = ImageDataset(images[test_idxs], targets_index[test_idxs], data_transforms['test'])

    return image_datasets