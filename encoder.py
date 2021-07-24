import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
from torchvision import models
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import PIL

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
#         self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)        
        features = features.view(features.size(0), -1)        
#         features = self.embed(features)        
        return features