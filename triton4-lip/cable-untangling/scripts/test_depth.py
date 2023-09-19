from shutil import move
from untangling.utils.interface_rws import Interface
from untangling.utils.grasp import GraspSelector
import numpy as np
from untangling.utils.tcps import *
from untangling.point_picking import *
import time
import matplotlib.pyplot as plt
import os
from untangling.utils.interface_rws import getH

def test_depth():
    SPEED = (0.6, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    # iface.open_grippers()
    iface.open_arms()
    iface.sync()
    img = iface.take_image()

    depth = img.depth._data / img.depth._data.max()
    color = img.color._data / img.color._data.max()

    images_of_interest = [depth, color]
    for i, image_of_interest in enumerate(images_of_interest):
        if i == 0:
            image = (image_of_interest * (color > 100/255)[:, :, :1])
        else:
            image = image_of_interest
        image = image[205:255, 580:625]
        # find min nonzero value
        image -= np.min(image[np.nonzero(image)])
        image = np.clip(image, 0, 1)
        image *= (1 / image.max())
        plt.clf()
        plt.imshow(image)
        plt.colorbar()
        plt.show()
if __name__ == '__main__':
    test_depth()

