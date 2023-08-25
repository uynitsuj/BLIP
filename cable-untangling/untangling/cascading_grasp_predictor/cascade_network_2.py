import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from untangling.cascading_grasp_predictor.resnet_dilated import Resnet34_8s

class CascadeNetwork2(nn.Module):
	def __init__(self, img_height=200, img_width=200, channels=4):
		super(CascadeNetwork2, self).__init__()
		self.num_outputs = 1
		self.img_height = img_height
		self.img_width = img_width
		self.resnet = Resnet34_8s(channels=channels, pretrained=False)
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
		output = self.resnet(x) 
		heatmaps = self.sigmoid(output[:,:self.num_outputs, :, :])
		return heatmaps
