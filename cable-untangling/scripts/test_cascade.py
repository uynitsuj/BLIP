import torch
import numpy as np
print(torch.cuda.is_available())
from untangling.cascading_grasp_predictor import get_points

point_network = get_points.NetworkGetPoints()

# in_img = torch.zeros((1, 3, 200, 200), dtype=torch.float32, device='cuda:0')
in_img = np.zeros((3, 200, 200), dtype=np.float32) #torch.tensor(np.zeros((1, 3, 200, 200)), dtype=torch.float32) #, device='cuda:0')

point_network.query(in_img)