from untangling.point_picking import click_points_simple
import cv2
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

    img = iface.take_image() # camera gets snapshot: rgb (color, actually grayscale on photoneo rip) and depth
    print(img.color._data.shape)
    print("Choose place points")
    pick_img, _ = click_points_simple(img) #outputs: left and right click

    pick_x, pick_y = list(pick_img) # 0.031 inches/pixel
    place_box_img_W, place_box_img_H = 194, 194

    '''
    quad_place_UL, quad_place_UR = (int(pick_x - place_box_img_W/2), int(pick_y - place_box_img_H/2)), (int(pick_x + place_box_img_W/2), int(pick_y - place_box_img_H/2))
    quad_place_BL, quad_place_BR = (int(pick_x - place_box_img_W/2), int(pick_y + place_box_img_H/2)), (int(pick_x + place_box_img_W/2), int(pick_y + place_box_img_H/2))
    # color = (255, 0, 0)
    '''
    
    '''
    ## SANITY FOR SQUARE AROUND PICK POINT
    plt.scatter(pick_img[0], pick_img[1], c='b')
    plt.scatter(quad_place_UL[0], quad_place_UL[1], c='r')
    plt.scatter(quad_place_UR[0], quad_place_UR[1], c='r')
    plt.scatter(quad_place_BL[0], quad_place_BL[1], c='r')
    plt.scatter(quad_place_BR[0], quad_place_BR[1], c='r')
    plt.imshow(img.color._data)
    plt.show()
    '''
    

    quad_transform = {'UL': [[-1, 0], [-1, 0]], 'UR': [[0, 1], [-1, 0]], 'BL': [[-1, 0], [0, 1]], 'BR': [[0, 1], [0, 1]]}
    lowest_density, lowest_quad, best_center = np.inf, None, None

    for k, v in quad_transform.items():
        min_x, max_x = pick_x + v[0][0]*place_box_img_W/2, pick_x + v[0][1]*place_box_img_W/2
        min_y, max_y = pick_y + v[1][0]*place_box_img_H/2, pick_y + v[1][1]*place_box_img_H/2

        curr_density = dst[int(min_y):int(max_y), int(min_x):int(max_x)].sum()

        if curr_density < lowest_density:
            lowest_density = curr_density
            lowest_quad = k
            best_center = (int((min_x + max_x)/2), int((min_y + max_y)/2))


    # print(lowest_quad)

    '''
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi
    grasp_left, grasp_right = calculate_grasps(pick_img, None, g, iface)
    iface.grasp(l_grasp=grasp_left, r_grasp=grasp_right)
    iface.sync()
    time.sleep(5)

    # place_world = g.ij_to_point(place_img).data # convert pixel coordinates to 3d coordinates
    # place_transform = RigidTransform(
    #     translation=place_world,
    #     rotation= iface.GRIP_DOWN_R,
    #     from_frame=YK.l_tcp_frame,
    #     to_frame="base_link",
    # )
    # # iface.go_pose_plan(r_target=place1_transform)
    # iface.go_cartesian(
    #     l_targets=[place_transform], removejumps=[6])
    # iface.sync()
    '''

