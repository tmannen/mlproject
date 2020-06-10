from torchvision import models
import torch.nn as nn

def create_basic_model(device):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 100)
    model_ft.to(device)
    return model_ft