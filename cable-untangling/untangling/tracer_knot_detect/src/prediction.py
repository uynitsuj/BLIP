import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import numpy as np

class Prediction:
    def __init__(self, model, num_keypoints, img_height, img_width, use_cuda=True, parallelize=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        if parallelize:
            self.model = nn.DataParallel(model)
            # self.model = nn.parallel.DistributedDataParallel(model)
            self.model.to(self.device)

        self.num_keypoints = num_keypoints
        self.img_height = img_height
        self.img_width = img_width
        self.use_cuda = use_cuda
        
    def predict(self, imgs):
        # img: torch.Tensor(3, height, width) 
        if len(imgs.shape) == 4:
            imgs = imgs.view(-1, imgs.shape[1], imgs.shape[2], imgs.shape[3])    
        elif len(imgs.shape) == 3:
            imgs = imgs.view(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
            
        heatmap = self.model.forward(Variable(imgs))
        return heatmap

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

