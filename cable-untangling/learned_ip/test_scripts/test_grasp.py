from untangling.point_picking import click_points_simple
import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage

def calculate_grasps(grasp_left, grasp_right, g, iface):   
    g.gripper_pos = 0.024
    #originally 0.0085
    if grasp_left is not None and grasp_right is not None:
        l_grasp, r_grasp = g.double_grasp(tuple(grasp_left), tuple(grasp_right), 0.004, 0.001, iface.L_TCP, iface.R_TCP, slide0=True, slide1=True)
        # l_grasp, r_grasp = g.double_grasp(tuple(grasp_left), tuple(grasp_right), 0.1, 0.1, iface.L_TCP, iface.R_TCP, slide0=True, slide1=True)
        
        #setting to be thicker bc rope 
        l_grasp.gripper_pos = 0.02
        r_grasp.gripper_pos = 0.02
                
    elif grasp_left is not None:
        l_grasp = g.single_grasp(tuple(grasp_left), 0.004, iface.L_TCP)
        l_grasp.gripper_pos = 0.02
        r_grasp = None
            
    elif grasp_right is not None:
        r_grasp = g.single_grasp(tuple(grasp_right), 0.001, iface.R_TCP)
        r_grasp.gripper_pos = 0.02
        l_grasp = None

    return l_grasp, r_grasp



if __name__ == "__main__":
    SPEED = (0.4, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    iface.open_grippers()
    iface.home()
    iface.sync()
    time.sleep(4)


    img = iface.take_image()
    print("Choose pick points")
    pick_left, pick_right= click_points_simple(img)

    if pick_left is not None:
        plt.scatter(pick_left[0], pick_left[1], c='b')
    if pick_right is not None:
        plt.scatter(pick_right[0], pick_right[1], c='r')
    plt.imshow(img.color._data)
    plt.show()

    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    grasp_left, grasp_right = calculate_grasps(pick_left, pick_right, g, iface)
    iface.grasp(l_grasp=grasp_left, r_grasp=grasp_right)
    iface.sync()
    time.sleep(5)



    