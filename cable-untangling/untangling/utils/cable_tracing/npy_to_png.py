import numpy as np
import cv2
import os

input_folder = '/home/justin/yumi/cable-untangling/scripts/oct_12_more_double_overhand'
output_folder = '/home/justin/yumi/cable-untangling/scripts/oct_12_more_double_overhand_png'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

counter = 0
for file in os.listdir(input_folder):
    # if 'color' not in file:
    #     continue
    img = np.load(os.path.join(input_folder, file),allow_pickle=True)[:, :, :3]
    out_path = os.path.join(output_folder, file.replace('.npy', '.png'))
    cv2.imwrite(out_path, img)