import cv2
import numpy as np
import torch
from untangling.cascading_grasp_predictor.cascade_network_1 import CascadeNetwork1
from untangling.cascading_grasp_predictor.cascade_network_2 import CascadeNetwork2
from untangling.utils.loop_box import get_loop_box
from torchvision import transforms
import matplotlib.pyplot as plt

# model1_path = "/home/justin/yumi/cable-untangling/untangling/cascading_grasp_predictor/cascade_network_1.py"
# model2_path = "/home/justin/yumi/cable-untangling/untangling/cascading_grasp_predictor/cascade_network_2.py"

model1_path = "/home/justin/yumi/cable-untangling/untangling/cascading_grasp_predictor/model_cascade_1.pth"
model2_path = "/home/justin/yumi/cable-untangling/untangling/cascading_grasp_predictor/model_cascade_2.pth"

transform = transforms.Compose([transforms.ToTensor()])

class NetworkGetPoints:
    def __init__(self,model1_path=model1_path, model2_path=model2_path, vis=True):
        self.model1=CascadeNetwork1().cuda()
        self.model1.load_state_dict(torch.load(model1_path))
        self.model1.eval()
        self.model2=CascadeNetwork2().cuda()
        self.model2.load_state_dict(torch.load(model2_path))
        self.model2.eval()
        self.img_width = 200
        self.img_height = 200
        self.gauss_sigma = 8
        self.vis=vis
        self.transform = transform
        
    def prepare_image(self,img):
        cv2.imwrite("temp.png", img)
        img_cv2 = cv2.imread("temp.png")
        resized = cv2.resize(img_cv2, (200,200))
        return resized

    def gauss_2d_batch(self, width, height, sigma, U, V, normalize_dist=False, single=False):
        if not single:
            U.unsqueeze_(1).unsqueeze_(2)
            V.unsqueeze_(1).unsqueeze_(2)
        X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
        X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
        G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
        if normalize_dist:
            return self.normalize(G).double()
        return G.double()

    def normalize(x):
        return F.normalize(x, p=1)

    def query(self,crop):
        print(crop.shape)
        img = cv2.resize(crop, (200,200))
        img = self.transform(img)
        img = img.unsqueeze(0).float().cuda()
        # img_temp = img.cpu().numpy().squeeze()
        # print(img_temp.shape)
        # plt.imshow(img.cpu().numpy().squeeze().transpose(1,2,0))
        # plt.show()

        heatmap = self.model1(img).cpu()
        heatmap = heatmap.detach().cpu().squeeze().numpy()
        # largest_box = get_loop_box(heatmap)
        # if largest_box is None:
        #     return None, None
        # pt1 = [(largest_box[1] + (largest_box[1] + largest_box[3])) // 2, (largest_box[0] + (largest_box[0] + largest_box[2])) // 2][::-1]
        # h_ratio = largest_box[2]/200
        # v_ratio = largest_box[3]/200
        # pt1 = [int(pt1[0] * h_ratio), int(pt1[1] * v_ratio)]
        pt1_y, pt1_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        pt1 = [pt1_x, pt1_y]
        # print(pt1)

        given = pt1
        given_gauss = self.gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, torch.tensor(given[0]).cuda(), torch.tensor(given[1]).cuda(), single=True)
        given_gauss = torch.unsqueeze(given_gauss, 0).cuda()
        # plt.imshow(given_gauss.cpu().squeeze().numpy())
        # plt.show()
        combined = torch.cat((img.squeeze().cuda().double(), given_gauss), dim=0).float()
        heatmap2 = self.model2(combined.unsqueeze(0)).cpu()
        heatmap2 = heatmap2.detach().cpu().squeeze().numpy()
        # largest_box2 = get_loop_box(heatmap2.detach().cpu().numpy())
        # if largest_box2 is None:
        #     return None, None
        # pt2 = [(largest_box2[1] + (largest_box2[1] + largest_box2[3])) // 2, (largest_box2[0] + (largest_box2[0] + largest_box2[2])) // 2][::-1]
        # h_ratio = largest_box2[2]/200
        # v_ratio = largest_box2[3]/200
        # pt2 = [int(pt2[0] * h_ratio), int(pt2[1] * v_ratio)]
        pt2_y, pt2_x = np.unravel_index(heatmap2.argmax(), heatmap2.shape)
        pt2 = [pt2_x, pt2_y]
        print(pt2)

        # concatenate the heatmaps horizontally and also return
        # the heatmaps
        heatmap = heatmap.squeeze()
        heatmap2 = heatmap2.squeeze()
        heatmap_tot = np.concatenate((heatmap, heatmap2), axis=-1)

        pt1 = [int(pt1[0] * crop.shape[1]/200), int(pt1[1] * crop.shape[0]/200)]
        pt2 = [int(pt2[0] * crop.shape[1]/200), int(pt2[1] * crop.shape[0]/200)]
        return pt1[::-1], pt2[::-1], heatmap_tot