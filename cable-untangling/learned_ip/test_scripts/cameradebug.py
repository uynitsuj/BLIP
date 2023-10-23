from untangling.point_picking import click_points_simple
import cv2
import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage

if __name__ == "__main__":
    SPEED = (0.4, 6 * np.pi) # speed of the yumi
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    ) # initialize the interface, set the motion speed of the yumi, tell yumi how to go to initial pose
    iface.open_grippers()
    time.sleep(4)
    iface.home()
    iface.sync() # bug: sometimes skips first command, so sync before moving to second command --> goal of sync but not working
    time.sleep(4) # small fix bc sync not working

    # img = iface.take_image() # camera gets snapshot: rgb (color, actually grayscale on photoneo rip) and depth
    img = iface.cam.read()
    plt.imshow(img.color._data)
    plt.show()