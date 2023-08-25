import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.insert(0, '/host/src')
from untangling.endpoint_classifier.resnet import resnet9

class EndpointClassifier(nn.Module):
	def __init__(self, pretrained=False, channels=2, num_classes=3, img_height=480, img_width=640, dropout=False):
		super(EndpointClassifier, self).__init__()
		self.img_height = img_height
		self.img_width = img_width
		self.resnet = resnet9(fully_conv=False,
                                       channels=channels,
                                       pretrained=pretrained,
                                       output_stride=8,
									   num_classes=num_classes, 
									   dropout = dropout,
                                       remove_avg_pool_layer=False)
		self.softmax = torch.nn.Softmax()

	def forward(self, x):
		output = self.resnet(x) 
		output = self.softmax(output)
		return output