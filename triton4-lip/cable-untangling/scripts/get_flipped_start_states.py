import shutil
from shutil import move
from untangling.utils.interface_rws import Interface
import numpy as np
from untangling.utils.tcps import *
import time
import matplotlib.pyplot as plt
import os
from untangling.utils.interface_rws import getH
import cv2

if __name__ == "__main__":
    start_states = '/home/justin/yumi/cable-untangling/scripts/start_states'
    img_save_dir = '/home/justin/yumi/cable-untangling/scripts/start_states_flipped_img2'
    if os.path.exists(img_save_dir):
        shutil.rmtree(img_save_dir)
    os.mkdir(img_save_dir)
    for file in np.sort(os.listdir(start_states)):
        if file[:10] == "2023-01-27":
            continue
        print(file[:-4])
        img = np.load(os.path.join(start_states, file))
        color_img = img[:, :, :3]
        color_img = np.flip(color_img)
        cv2.imwrite(os.path.join(img_save_dir, file[:-4] + '.png'), color_img)