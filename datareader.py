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


class DataLoader:
    
    def __init__(self,flooded_path,nonflooded_path):
        self.flooded_path = flooded_path
        self.nonflooded_path = nonflooded_path
        
    def getData(self):
        labels = []
        flooded_images = []
        non_flooded_images = []
        
        for name in os.listdir(self.flooded_path):
            image = plt.imread(self.flooded_path+name)
            image = cv2.resize(image,(3136,3136))
            flooded_images.append(image)
            labels.append(1)
            
        for name in os.listdir(self.nonflooded_path):
            image = plt.imread(self.nonflooded_path+name)
            image = cv2.resize(image,(3136,3136))
            non_flooded_images.append(image)
            labels.append(0)
            
        return np.array(flooded_images+non_flooded_images),np.array(labels)
    

        