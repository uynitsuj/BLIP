from untangling.point_picking import click_points_simple
import cv2
import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
import math


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

def find_circle_based_place(img, trace_point, center_point, viz=True):
    radius = math.sqrt((trace_point[0] - center_point[0])**2 + (trace_point[1] - center_point[1])**2)
    start_theta = (math.atan((trace_point[1] - center_point[1])/(trace_point[0] - center_point[0])))
    if trace_point[0] < center_point[0]:
        start_theta += math.pi
    print(start_theta)
    x = center_point[0] + radius*math.cos(start_theta)
    y = center_point[1] + radius*math.sin(start_theta)
    plt.scatter(x, y, c='g', s = 3)
    plt.imshow(img.color._data)
    plt.show()


    # plot the circle
    # for theta in np.linspace(start_theta, start_theta + 2*math.pi, 100):
    #     print(theta)
    #     x = center_point[0] + radius*math.cos(theta)
    #     y = center_point[1] + radius*math.sin(theta)
    #     plt.scatter(x, y, c='r', s=1)
    # plt.imshow(img.color._data)
    # plt.show()

    # convert image to a binary mask
    thresh, maxValue = 127, 255
    _, binary = cv2.threshold(img.color._data[:,:,0], thresh, maxValue, cv2.THRESH_BINARY)

    step = 2*math.pi/100


    # go clockwise
    i = 0
    clockwise_neighbor = None
    for theta in np.linspace(start_theta, start_theta + 2*math.pi, 100):
        x = center_point[0] + radius*math.cos(theta)
        y = center_point[1] + radius*math.sin(theta)

        if binary[int(y), int(x)] == 255 and i > 1:
            # plt.scatter(x, y, c='r', s = 3)
            clockwise_neighbor = theta
            break

        # plt.scatter(x, y, c='b', s = 3)
        i += 1
        
    # plt.imshow(binary)
    # plt.show()


    

    # go counterclockwise
    i = 0
    counterclockwise_neighbor = None
    for theta in np.linspace(start_theta, start_theta - 2*math.pi, 100):
        x = center_point[0] + radius*math.cos(theta)
        y = center_point[1] + radius*math.sin(theta)

        if binary[int(y), int(x)] == 255 and i > 1:
            # plt.scatter(x, y, c='r', s = 3)
            counterclockwise_neighbor = theta
            break
        # plt.scatter(x, y, c='b', s = 3)
        i += 1
        
    # plt.imshow(binary)
    # plt.show()


    # find the midpoint
    midpoint_theta = (clockwise_neighbor + counterclockwise_neighbor)/2

    midpoint_x = center_point[0] + radius*math.cos(midpoint_theta)
    midpoint_y = center_point[1] + radius*math.sin(midpoint_theta)
    clockwise_x = center_point[0] + radius*math.cos(clockwise_neighbor)
    clockwise_y = center_point[1] + radius*math.sin(clockwise_neighbor)
    counterclockwise_x = center_point[0] + radius*math.cos(counterclockwise_neighbor)
    counterclockwise_y = center_point[1] + radius*math.sin(counterclockwise_neighbor)

    if viz:
        plt.scatter(midpoint_x, midpoint_y, c='r', s = 3)
        plt.scatter(clockwise_x, clockwise_y, c='b', s = 3)
        plt.scatter(counterclockwise_x, counterclockwise_y, c='b', s = 3)
        plt.imshow(img.color._data)
        plt.show()

    return [int(midpoint_x), int(midpoint_y)]





    



def find_grasp_ip_place_point(img, pick_img, place_box_img_W, viz=True):
    pick_x, pick_y = list(pick_img)
    thresh, maxValue = 127, 255
    _, dst = cv2.threshold(img.color._data, thresh, maxValue, cv2.THRESH_BINARY)

    quad_transform = {'UL': [[-1, 0], [-1, 0]], 'UR': [[0, 1], [-1, 0]], 'BL': [[-1, 0], [0, 1]], 'BR': [[0, 1], [0, 1]]}
    lowest_density, lowest_quad, best_center = np.inf, None, None

    for k, v in quad_transform.items():
        min_x, max_x = pick_x + v[0][0]*place_box_img_W/2, pick_x + v[0][1]*place_box_img_W/2
        min_y, max_y = pick_y + v[1][0]*place_box_img_W/2, pick_y + v[1][1]*place_box_img_W/2

        curr_density = dst[int(min_y):int(max_y), int(min_x):int(max_x)].sum()

        if curr_density < lowest_density:
            lowest_density = curr_density
            lowest_quad = k
            best_center = (int((min_x + max_x)/2), int((min_y + max_y)/2))

    place_img = [best_center[0], best_center[1]]

    if viz:
        plt.scatter(pick_img[0], pick_img[1], c='b')
        plt.scatter(place_img[0], place_img[1], c='r')
        plt.imshow(img.color._data)
        plt.show()
    return place_img

def place_circle_grasp_ip(pick_point, place_point, iface):
    # check which arm is better
    avg_point = (pick_point + place_point)/2
    if avg_point[0] < 590:
        grasp_left, grasp_right, place_left, place_right = pick_point, None, place_point, None
    else:
        grasp_left, grasp_right, place_left, place_right = None, pick_point, None, place_point


    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi
    grasp_left, grasp_right = calculate_grasps(grasp_left, grasp_right, g, iface)
    iface.grasp(l_grasp=grasp_left, r_grasp=grasp_right)
    iface.sync()

    if place_left is not None:
        place_world = g.ij_to_point(place_left).data # convert pixel coordinates to 3d coordinates
        place_transform = RigidTransform(
            translation=place_world,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        # iface.go_pose_plan(r_target=place_transform)
        iface.go_cartesian(
            # l_targets=[place_transform], removejumps=[6])
            l_targets = [place_transform], removejumps=[5, 6])
        iface.sync()

        
        iface.open_grippers()
        iface.sync()
        
        iface.home()
        iface.sync()
    else:
        place_world = g.ij_to_point(place_right).data # convert pixel coordinates to 3d coordinates
        place_transform = RigidTransform(
            translation=place_world,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        # iface.go_pose_plan(r_target=place_transform)
        iface.go_cartesian(
            # l_targets=[place_transform], removejumps=[6])
            r_targets = [place_transform], removejumps=[5, 6])
        iface.sync()

        
        iface.open_grippers()
        iface.sync()
        
        iface.home()
        iface.sync()




if __name__ == "__main__":
    SPEED = (0.4, 6 * np.pi) # speed of the yumi
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    img = None
    iface.open_grippers()
    iface.home()
    iface.sync() 

    img = iface.take_image() 
    print(img.color._data.shape)
    # print("Choose pick points")
    # pick_img, _ = click_points_simple(img) #outputs: left and right click

    # pick_x, pick_y = list(pick_img) # 0.031 inches/pixel
    # place_box_img_W, place_box_img_H = 194, 194
    # place_box_img_W = 300
    # place_img = find_grasp_ip_place_point(img, pick_img, 300, viz=True)

    print("Choose center points")
    center_point, _ = click_points_simple(img) 
    print("Choose trace radius points")
    trace_point, _ = click_points_simple(img) 
    center_point = np.array([center_point[0], center_point[1]])
    trace_point = np.array([trace_point[0], trace_point[1]])
    place_img = find_circle_based_place(img, trace_point, center_point, viz=True)
    place_circle_grasp_ip(trace_point, place_img, iface)



    # g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi
    # grasp_left, grasp_right = calculate_grasps(pick_img, None, g, iface)
    # iface.grasp(l_grasp=grasp_left, r_grasp=grasp_right)
    # iface.sync()
    # #time.sleep(2)

    # place_world = g.ij_to_point(place_img).data # convert pixel coordinates to 3d coordinates
    # place_transform = RigidTransform(
    #     translation=place_world,
    #     rotation= iface.GRIP_DOWN_R,
    #     from_frame=YK.l_tcp_frame,
    #     to_frame="base_link",
    # )
    # # iface.go_pose_plan(r_target=place_transform)
    # iface.go_cartesian(
    #     # l_targets=[place_transform], removejumps=[6])
    #     l_targets = [place_transform], removejumps=[5, 6])
    # iface.sync()

    
    # iface.open_grippers()
    # iface.sync()
    
    # iface.home()
    # iface.sync()

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
