import torch
import torch.nn as nn
from torchvision import models

def create_basic_model(device):
    # 
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 100)
    return model_ft

class MVCNN(nn.Module):
    """
    Multi-View Convolutional Neural Network (MVCNN)
    Initializes a model with the architecture of a MVCNN with a ResNet18 base.

    Modified from: https://github.com/SAMY-ER/Multi-View-Image-Classification/blob/master/network.py

    Uses a backbone to calculate feature vectors from 4 images, concatenates them and makes predictions.
    """
    def __init__(self, num_classes=100, pretrained=True):
        super(MVCNN, self).__init__()
        resnet = models.resnet18(pretrained = pretrained)
        fc_in_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(fc_in_features*4, 2048),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(inplace=True),
            nn.Linear(fc_in_features*4, num_classes)
        )

    def forward(self, inputs): # inputs.shape = samples x views x height x width x channels
        #inputs = inputs.transpose(0, 1)
        view_features = [] 
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch)   
        
        all_features = torch.cat(view_features).view(inputs.shape[0], -1)
        outputs = self.classifier(all_features)
        return outputs

class MVCNN_max(nn.Module):
    """
    Multi-View Convolutional Neural Network (MVCNN)
    Initializes a model with the architecture of a MVCNN with a ResNet34 base.
    """
    def __init__(self, num_classes=100, pretrained=True):
        super(MVCNN_max, self).__init__()
        resnet = models.resnet18(pretrained = pretrained)
        fc_in_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            # nn.Linear(fc_in_features, num_classes),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(inplace=True),
            nn.Linear(fc_in_features, num_classes)
        )

    def forward(self, inputs): # inputs.shape = samples x views x height x width x channels
        inputs = inputs.transpose(0, 1)
        view_features = [] 
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            #print(view_batch.shape)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch)   
        
        #print(torch.stack(view_features).shape)
        pooled_views, _ = torch.max(torch.stack(view_features), 0)
        #print(pooled_views.shape)
        outputs = self.classifier(pooled_views)
        return outputs