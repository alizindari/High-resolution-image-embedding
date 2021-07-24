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

class PreProcessing:
    def __init__(self,transform,encoder):
        self.transform = transform
        self.encoder = encoder
        
    def applyTransform(self,image):
        return self.transform(image)
    
    def cropImage(self,image):
        cropped_parts = []
        for i in range(14):
            for j in range(14):
                im = image[i*224:(i*224)+224,j*224:(j*224)+224,:]
                im = np.expand_dims(im,axis=0)
                cropped_parts.append(im)
        return np.concatenate(cropped_parts,axis=0)
    
    
    def generateEmbedding(self,batch_image,device):
        
        data = []
        self.encoder.eval()
        embeddings = []
        for i in range(len(batch_image)):
            im = batch_image[i]
            im = self.applyTransform(im)
            im = im.unsqueeze(0)
            data.append(im)
        data = torch.cat(data,dim=0)
        data = data.to(device)
        encodded_image = self.encoder(data)
        
        return encodded_image