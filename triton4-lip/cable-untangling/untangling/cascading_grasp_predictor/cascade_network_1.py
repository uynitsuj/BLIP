import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from untangling.cascading_grasp_predictor.resnet_dilated import Resnet34_8s
# from resnet_dilated import Resnet34_8s

class CascadeNetwork1(nn.Module):
	def __init__(self):
		super(CascadeNetwork1, self).__init__()
		self.num_outputs = 1
		self.resnet = Resnet34_8s()
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		output = self.resnet(x) 
		heatmaps = self.sigmoid(output[:,:self.num_outputs, :, :])
		return heatmaps
