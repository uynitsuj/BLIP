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

def test_cage_pinch():
    SPEED = (0.6, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    iface.open_grippers()
    iface.open_arms()
    time.sleep(3)
    iface.sync()
    iface.close_grippers()
    iface.sync()
    img = iface.take_image()
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    while True:
        try:
            left_coords, right_coords = click_points(img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
            l_grasp, r_grasp = g.double_grasp(
                    left_coords, right_coords, .0085, .0085, iface.L_TCP, iface.R_TCP)
            iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp)
            iface.sync()
            iface.partial_pull_apart(1, slide_left=False, slide_right=True)
            iface.sync()
            iface.open_grippers()
            time.sleep(3)
        except:
            continue

if __name__ == '__main__':
    test_cage_pinch()

