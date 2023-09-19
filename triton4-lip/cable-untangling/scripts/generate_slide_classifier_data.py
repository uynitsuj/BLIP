from typing import no_type_check
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt 
from autolab_core import RigidTransform, Point
from phoxipy.phoxi_sensor import PhoXiSensor
from yumiplanning.yumi_kinematics import YuMiKinematics as YK

# load np images in data-justin that start with "color" and save as jpg
justin_dir = "./keep_going_test"
threshold = True #img and depth thresholding
folder = "dataset_thresh" if threshold else "img_no_thresh"
images_path = justin_dir + "/" + folder

if os.path.exists(images_path):
    remove_command = 'rm -r ' + images_path
    os.system(remove_command)
os.mkdir(images_path)

counter_file_name = 0
for file in np.sort(os.listdir(justin_dir)):
    if file.startswith("color") and not file.endswith(".jpg"):
        print("Processing:", os.path.join(justin_dir, file))
        img = np.load(os.path.join(justin_dir, file))
        img = img.astype(np.uint8)
        file_num = int(file[file.index("_") + 1: file.index(".")])
        depth_file = "depth_" + str(file_num) + ".npy" 
        right_grip_pose_file = "r_pose" + str(file_num) + ".tf"
        right_pos = RigidTransform.load(os.path.join(justin_dir, right_grip_pose_file))
        p = Point(right_pos.translation,frame=YK.base_frame)
        T_CAM_BASE = RigidTransform.load("/home/justin/yumi/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")
        T_BASE_CAM = T_CAM_BASE.inverse()
        intr = PhoXiSensor.create_intr(img.shape[1], img.shape[0])
        right_grip_pixel_coord = intr.project(T_BASE_CAM*p)
        print(right_grip_pixel_coord)
        right_x = right_grip_pixel_coord[0]
        right_y = right_grip_pixel_coord[1]
        depth_og = np.load(os.path.join(justin_dir, depth_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        depth_og = np.squeeze(depth_og)
        thresh = 0.7
        if threshold:
            gray[depth_og>thresh]=0
            gray[depth_og==0]=0
        img = gray
        depth_og[depth_og>thresh] = 0
        depth = depth_og
        # plt.imshow(depth)
        # plt.show()

        #img = np.where(img > 90, img, 0) #img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]
        img[int(4*img.shape[0]/5):, :] = 0
        depth[int(4*depth.shape[0]/5):, :] = 0
        img_crop = img.copy()[right_y-220:right_y-20, right_x-80: right_x+120]
        depth_crop = depth.copy()[right_y-220:right_y-20, right_x-80: right_x+120]
        # img_resized = cv2.resize(img, (640,480))
        # depth_resized = cv2.resize(depth, (640, 480))
        depth_crop = np.resize(depth_crop, (depth_crop.shape[1], depth_crop.shape[0]))
        _,axs=plt.subplots(1,2)
        axs[0].set_title(file)
        axs[0].imshow(depth_crop)
        axs[1].imshow(img_crop)
        plt.show()

        combined = np.stack((img_crop, depth_crop)) 
        # format file name to have 5 digits
        # np.save(f"{justin_dir}/{folder}/" + str(counter_file_name).zfill(5) + ".npy", combined)

        #cv2.imwrite(f"{justin_dir}/dataset/" + str(counter_file_name).zfill(5) + ".png", img)
        counter_file_name += 1
