from untangling.point_picking import click_points_simple, click_points_closest
from blip_pipeline.add_noise_to_img import run_tracer_with_transform
import cv2
import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, Point, RgbdImage, DepthImage, ColorImage
from scripts.full_pipeline_trunk import FullPipeline

from collections import OrderedDict
import argparse
import logging
import numpy as np
import colorsys

DIST_TO_TABLE_LEFT = 1.025 # meters
DIST_TO_TABLE_RIGHT = 0.99 # meters
T_CAM_BASE = RigidTransform.load("/home/mallika/triton4-lip/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")

CAMERA_CALIBRATION_OFFSET = 24e-2

def find_dist_to_table(x, y):
    # return -0.0001357143*x  + 1.087143
    # return -0.0000698*x + 1.0493
    # return -1.07490104e-04*x + -6.14310790e-06*y + 1.07035846 # 1.0475 was better but not enough

    # return -1.07451205e-04*x -4.36124962e-06*y +1.06936426
    # return -1.07985894e-04*x-4.80273450e-06*y+1.06983336e+00

    #return -1.13533896e-04*x -9.31199774e-06*y + 1.07163214e+00
    # return -1.06121824e-04*x + -8.13607113e-06*y + 1.07033272e+00
    # ### NEW CALIBRATION
    # return 2.26352019e-04*x -4.88326330e-04*y + 1.06551469e+00

    ### NEW FOAM SETUP
    return -1.06241415e-04*x -1.03883880e-05*y + 1.07033272e+00

    # return 1

def get_world_coord_from_pixel_coord(pixel_coord, cam_intrinsics):
    '''
    pixel_coord: [x, y] in pixel coordinates
    cam_intrinsics: 3x3 camera intrinsics matrix
    '''
    pixel_coord = np.array(pixel_coord)
    point_3d_cam = find_dist_to_table(pixel_coord[0], pixel_coord[1])* np.linalg.inv(cam_intrinsics._K).dot(np.r_[pixel_coord, 1.0])
    point_3d_world = T_CAM_BASE.matrix.dot(np.r_[point_3d_cam, 1.0])
    print('homogenous = ', point_3d_world)
    point_3d_world = point_3d_world[:3]/point_3d_world[3]
    point_3d_world[-1] = point_3d_world[-1] + CAMERA_CALIBRATION_OFFSET
    print('non-homogenous = ', point_3d_world)
    return point_3d_world

def visualize_trace(img, trace):
    img = img.copy()
    def color_for_pct(pct):
        return colorsys.hsv_to_rgb(pct, 1, 1)[0] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[1] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[2] * 255
    for i in range(len(trace) - 1):
        # if trace is ordered dict, use below logic
        if not isinstance(trace, OrderedDict):
            pt1 = tuple(trace[i].astype(int))
            pt2 = tuple(trace[i+1].astype(int))
        else:
            trace_keys = list(trace.keys())
            pt1 = trace_keys[i]
            pt2 = trace_keys[i + 1]
        cv2.line(img, pt1[::-1], pt2[::-1], color_for_pct(i/len(trace)), 4)
    plt.title("Trace Visualized")
    plt.imshow(img)
    plt.show()
    

# def get_world_coord_from_pixel_coord(arm, pixel_coord, cam_intrinsics):
#     if arm == "left":
#         pixel_coord = np.array(pixel_coord)
#         point_3d_cam = DIST_TO_TABLE_LEFT * np.linalg.inv(cam_intrinsics._K).dot(np.r_[pixel_coord, 1.0])
#         point_3d_world = T_CAM_BASE.matrix.dot(np.r_[point_3d_cam, 1.0])
#         point_3d_world = point_3d_world[:3]/point_3d_world[3]
#         return point_3d_world
#     else:
#         pixel_coord = np.array(pixel_coord)
#         point_3d_cam = DIST_TO_TABLE_RIGHT * np.linalg.inv(cam_intrinsics._K).dot(np.r_[pixel_coord, 1.0])
#         point_3d_world = T_CAM_BASE.matrix.dot(np.r_[point_3d_cam, 1.0])
#         point_3d_world = point_3d_world[:3]/point_3d_world[3]
#         return point_3d_world


def gaussian_2d(width, height):
    x, y = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width))
    # x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 0.6, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    
    return g

def get_closest_trace_idx(pix, trace):
    distances = np.linalg.norm(np.array(trace) - np.array(pix)[None, ...], axis=1)
    return np.argmin(distances)

def is_valid_cc(cc_stats, cc_labels, cc):
    print(cc_stats[cc][4])
    return is_large_cc(cc_stats, cc) and cc > 0

def is_large_cc(cc_stats, cc):
    return cc_stats[cc][4] > 300 # minimum cc pixel area of 1000 may be tuned

# checks that the CC doesn't go from one end of the image to the other
def not_touching_both_ends_cc(cc_lables, cc):
    if cc in cc_lables[0, :] and cc in cc_lables[-1, :]:
        return False
    if cc in cc_lables[:, 0] and cc in cc_lables[:, -1]:
        return False
    return True

# Point of inaccessability is the point within a polygon that is furthest from the boundary of the polygon
def get_pole_of_inaccessibility(labels, label):
    poles = None

    #for label in range(num_labels):
    padded_labels = np.pad(labels, ((1,1),(1,1)), mode='constant', constant_values=0)

    binary_img = (padded_labels == label).astype(np.uint8)
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    # adjusted_max_loc = (max_loc[0]-1, max_loc[1]-1)
    # poles = adjusted_max_loc

    # print(np.max(dist_transform))
    # plt.clf()
    # plt.scatter(poles[0], poles[1])
    # plt.imshow(padded_labels+dist_transform)
    # plt.show()
    center_bias = np.multiply(gaussian_2d(padded_labels.shape[0], padded_labels.shape[1]), dist_transform)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(center_bias)
    adjusted_max_loc = (max_loc[0]-1, max_loc[1]-1)
    poles = adjusted_max_loc
    # plt.clf()
    # plt.scatter(poles[0], poles[1])
    # plt.imshow(padded_labels+center_bias)
    # plt.show()
    
    return poles


def get_tangent_vec_poi(push_coord, trace):
    
    poi_trace = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)]
    poi_trace = [poi_trace[1], poi_trace[0]]
    poi_closest = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)+1]
    poi_closest = [poi_closest[1], poi_closest[0]]
    return poi_trace, poi_closest

def get_push_vec(poi_trace, poi_closest, img_rgb, place_box_img_W, type="poles", viz=True):
    
    return 
    
def get_poi_and_vec_for_push(push_coord, img_rgb, trace, place_box_img_W=100, type="poles", viz=True):
    push_coord = [push_coord[1], push_coord[0]]
    poi_trace = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)]
    poi_trace = [poi_trace[1], poi_trace[0]]
    poi_closest = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)+1]
    poi_closest = [poi_closest[1], poi_closest[0]]

    if viz:
        plt.scatter(poi_trace[0], poi_trace[1])
        plt.scatter(poi_closest[0], poi_closest[1])
        plt.imshow(img_rgb)
        plt.show()
    
    poi_tan_vec = (np.array(poi_trace) - np.array(poi_closest))/np.linalg.norm(np.array(poi_trace) - np.array(poi_closest)) #unit vector

    x_start = max(0,poi_trace[1]-place_box_img_W//2)
    x_end = min(poi_trace[1]+place_box_img_W//2, img_rgb.shape[0])
    y_start = max(0, poi_trace[0]-place_box_img_W//2)
    y_end = min(img_rgb.shape[1], poi_trace[0]+place_box_img_W//2)

    cropped_img = img_rgb[x_start:x_end, y_start:y_end, 0]
    poi_cropped = [poi_trace[0]-y_start, poi_trace[1]-x_start]

    thresh, maxValue = 127, 255
    _, binary_img = cv2.threshold(cropped_img, thresh, maxValue, cv2.THRESH_BINARY)

    #if viz:
    
    #plt.imshow(binary_img, cmap='gray')
    #plt.show()
    dilate_size = 8
    dilate_kernel = np.ones((dilate_size,dilate_size), np.uint8)
    dilated_binary_img = cv2.dilate(binary_img, dilate_kernel)
    #plt.imshow(dilated_binary_img, cmap='gray')
    #plt.show()

    for i in range(dilated_binary_img.shape[0]):
        for j in range(dilated_binary_img.shape[1]):
            if dilated_binary_img[i][j] == 255:
                dilated_binary_img[i][j] = 0
            else:
                dilated_binary_img[i][j] = 1

    num_labels, labels, stats, centroids= cv2.connectedComponentsWithStats(dilated_binary_img)
    
    if viz:
        plt.scatter(poi_cropped[0],poi_cropped[1])

    maxdotprod = 0
    maxpoint = None
    maxlabel = -1
    for label in range(num_labels):
        if is_valid_cc(stats, labels, label): # minimum cc pixel area of 1000 may be tuned

            if type == "poles":
                center_point = get_pole_of_inaccessibility(labels=labels, label = label)
                plt.title("Per-Region Poles of Inaccessibility")
            else:
                center_point = centroids[label]
                plt.title("Per-Region Centroid")

            plt.scatter(center_point[0],center_point[1])
            # pointerest_center_vec = (np.array(center_point) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))/np.linalg.norm(np.array(center_point) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
            pointerest_center_vec = (np.array(center_point) - poi_cropped)/np.linalg.norm(np.array(center_point) - poi_cropped)
            
            print("vector: ",pointerest_center_vec)
            print(np.dot(poi_tan_vec, pointerest_center_vec))
            plt.text(center_point[0]-5,center_point[1]-5, str(round(np.dot(poi_tan_vec, pointerest_center_vec), 2)), fontsize=14)
            
            poidotprod = np.abs(np.dot(poi_tan_vec, pointerest_center_vec))
            if poidotprod >= maxdotprod:
                maxdotprod = poidotprod
                maxlabel = label
                maxpoint = center_point
    #if maxpoint == None:

    if viz:
        plt.imshow(labels)
        plt.show()

    # cen_poi_vec = (np.array(maxpoint) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
    cen_poi_vec = (np.array(maxpoint) - poi_cropped)
    return poi_trace, cen_poi_vec

def get_poi_and_vec_for_push2(push_coord, img_rgb, trace, place_box_img_W=100, viz=True):
    push_coord = [push_coord[1], push_coord[0]]
    poi_trace = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)]
    poi_trace = [poi_trace[1], poi_trace[0]]
    poi_closest = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)+1]
    poi_closest = [poi_closest[1], poi_closest[0]]

    if viz:
        plt.scatter(poi_trace[0], poi_trace[1])
        plt.scatter(poi_closest[0], poi_closest[1])
        plt.imshow(img_rgb)
        plt.show()
    
    poi_tan_vec = (np.array(poi_trace) - np.array(poi_closest))/np.linalg.norm(np.array(poi_trace) - np.array(poi_closest)) #unit vector

    x_start = max(0,poi_trace[1]-place_box_img_W//2)
    x_end = min(poi_trace[1]+place_box_img_W//2, img_rgb.shape[0])
    y_start = max(0, poi_trace[0]-place_box_img_W//2)
    y_end = min(img_rgb.shape[1], poi_trace[0]+place_box_img_W//2)

    cropped_img = img_rgb[x_start:x_end, y_start:y_end, 0]
    poi_cropped = [poi_trace[0]-y_start, poi_trace[1]-x_start]

    thresh, maxValue = 127, 255

    _, binary_img_inv = cv2.threshold(cropped_img, thresh, maxValue, cv2.THRESH_BINARY_INV)
    _, binary_img = cv2.threshold(cropped_img, thresh, maxValue, cv2.THRESH_BINARY)

    dilate_size = 8
    dilate_kernel = np.ones((dilate_size,dilate_size), np.uint8)
    dilated_binary_img = cv2.dilate(binary_img, dilate_kernel)

    for i in range(dilated_binary_img.shape[0]):
        for j in range(dilated_binary_img.shape[1]):
            if dilated_binary_img[i][j] == 255:
                dilated_binary_img[i][j] = 0
            else:
                dilated_binary_img[i][j] = 1

    for i in range(binary_img_inv.shape[0]):
        for j in range(binary_img_inv.shape[1]):
            if binary_img_inv[i][j] == 255:
                binary_img_inv[i][j] = 0
            else:
                binary_img_inv[i][j] = 1


    print(binary_img_inv)


    cropped_low_dense = get_pole_of_inaccessibility(labels=binary_img_inv, label=0)



    print(cropped_low_dense)

    plt.imshow(dilated_binary_img)
    plt.scatter(cropped_low_dense[0],cropped_low_dense[1])
    plt.show()
    
    num_labels, labels, stats, centroids= cv2.connectedComponentsWithStats(dilated_binary_img)

    rejectlabel = labels[cropped_low_dense[0]][cropped_low_dense[1]]
    print(rejectlabel)
    plt.imshow(labels)
    plt.scatter(cropped_low_dense[0],cropped_low_dense[1])
    plt.show()
    
    if viz:
        plt.scatter(poi_cropped[0],poi_cropped[1])

    maxdotprod = -1
    maxpoint = None
    maxlabel = -1
    mindistzero = np.inf
    for label in range(num_labels):
        if is_valid_cc(stats, labels, label): # minimum cc pixel area of 1000 may be tuned
            center_point = get_pole_of_inaccessibility(labels=labels, label = label)
            plt.title("Per-Region Poles of Inaccessibility")
            plt.scatter(center_point[0],center_point[1])
            # pointerest_center_vec = (np.array(center_point) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))/np.linalg.norm(np.array(center_point) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
            pointerest_center_vec = (np.array(center_point) - poi_cropped)/np.linalg.norm(np.array(center_point) - poi_cropped)
            
            print("vector: ",pointerest_center_vec)
            print(np.dot(poi_tan_vec, pointerest_center_vec))
            plt.text(center_point[0]-5,center_point[1]-5, str(round(np.dot(poi_tan_vec, pointerest_center_vec), 2)), fontsize=14)
            
            poidotprod = np.abs(np.dot(poi_tan_vec, pointerest_center_vec))
            if np.abs(poidotprod) < mindistzero:
                mindistzero = np.abs(poidotprod)
                maxlabel = label
                maxpoint = center_point
 
    if viz:
        plt.imshow(labels)
        plt.show()

    # cen_poi_vec = (np.array(maxpoint) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
    cen_poi_vec = (np.array(maxpoint) - poi_cropped)
    return poi_trace, cen_poi_vec

def perform_push_through(poi_trace, cen_poi_vec, img, fullPipeline, viz=True):
    
    waypoint1 = np.array([poi_trace[0], poi_trace[1]]) + cen_poi_vec
    waypoint2 = np.array([poi_trace[0], poi_trace[1]]) - cen_poi_vec*0.65

    # if viz:
    #     plt.text(waypoint1[0]+7,waypoint1[1]+7, 'pt 1')
    #     plt.text(waypoint2[0]+7,waypoint2[1]+7, 'pt 2')
    #     plt.scatter(waypoint1[0],waypoint1[1])
    #     plt.scatter(waypoint2[0],waypoint2[1])
    #     plt.imshow(img.color._data)
    #     plt.show()

    # waypoint1 = [int(waypoint1[0]), int(waypoint1[1])]
    # waypoint2 = [int(waypoint2[0]), int(waypoint2[1])]

    if viz:
        plt.text(waypoint1[0]+7,waypoint1[1]+7, 'pt 1')
        plt.text(waypoint2[0]+7,waypoint2[1]+7, 'pt 2')
        plt.scatter(waypoint1[0],waypoint1[1])
        plt.scatter(waypoint2[0],waypoint2[1])
        plt.imshow(img.color._data)
        plt.show()

    iface = fullPipeline.iface
    iface.close_grippers()
    iface.sync()
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi
    h = img.shape[0]
    print(h)
    waypoint1 = [int(waypoint1[0]), int(waypoint1[1])]
    waypoint2 = [int(waypoint2[0]), int(waypoint2[1])]

    print('waypoint1: ', waypoint1)
    print('waypoint2: ', waypoint2)
    

    average_place = (np.array(waypoint1) + np.array(waypoint2))/2
    fixed_depth = 0.0408
    #fixed_depth = 0.0935
    if average_place[0] > 590:
        print('waypoint1 again: ', waypoint1)
        print('waypoint2 again: ', waypoint2)
        # # WORLD COORDINATE ESTIMATION WITH DEPTH
        place1 = g.ij_to_point(waypoint1).data # convert pixel coordinates to 3d coordinates
        place2 = g.ij_to_point(waypoint2).data # convert pixel coordinates to 3d coordinates
        # print('with depth calculation: ')
        # print(place1)
        # print(place2)
        place1 = [place1[0], place1[1], fixed_depth]
        place2 = [place2[0], place2[1], fixed_depth]
        # intermediate_place1 = place1 + np.array([0, 0, 0.05])
        # print('without depth calculation fixed z: ')
        print(place1)
        print(place2)
        # place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        # place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        intermediate_place1 = place1 + np.array([0, 0, 0.05])
        intermediate_place2 = place2 + np.array([0, 0, 0.05])
        # right arm
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

        intermediate_place2_transform = RigidTransform(
            translation=intermediate_place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

        place1_transform = RigidTransform(
            translation=place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        place2_transform = RigidTransform(
            translation=place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        iface.go_cartesian(r_targets=[intermediate_place1_transform], removejumps=[6])
        iface.sync()

        iface.go_cartesian(r_targets=[place1_transform], removejumps=[6])
        iface.sync()

        iface.go_cartesian(r_targets=[place2_transform], removejumps=[6])
        iface.sync()

        iface.go_cartesian(r_targets=[intermediate_place2_transform], removejumps=[6])
        iface.sync()

    else:
        # left arm
        # WORLD COORDINATE ESTIMATION WITH DEPTH
        place1 = g.ij_to_point(waypoint1).data # convert pixel coordinates to 3d coordinates
        place2 = g.ij_to_point(waypoint2).data # convert pixel coordinates to 3d coordinates
        # print('with depth calculation: ')
        # print(place1)
        # print(place2)
        place1 = [place1[0], place1[1], fixed_depth]
        place2 = [place2[0], place2[1], fixed_depth]
        # intermediate_place1 = place1 + np.array([0, 0, 0.05])
        # print('without depth calculation fixed z: ')
        print(place1)
        print(place2)

        # place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        # place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        intermediate_place1 = place1 + np.array([0, 0, 0.05])
        intermediate_place2 = place2 + np.array([0, 0, 0.05])
        
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        intermediate_place2_transform = RigidTransform(
            translation=intermediate_place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        place1_transform = RigidTransform(
            translation=place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        place2_transform = RigidTransform(
            translation=place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        iface.go_cartesian(l_targets=[intermediate_place1_transform], removejumps=[6])
        iface.sync()

        iface.go_cartesian(l_targets=[place1_transform], removejumps=[6])
        iface.sync()

        iface.go_cartesian(l_targets=[place2_transform], removejumps=[6])
        iface.sync()

        iface.go_cartesian(l_targets=[intermediate_place2_transform], removejumps=[6])
        iface.sync()

    iface.open_grippers()
    iface.sync()
    iface.home()
    iface.sync()
    time.sleep(2)

def perform_push_through2(poi_trace, cen_poi_vec, img, fullPipeline, viz=True):
    
    waypoint1 = np.array([poi_trace[0], poi_trace[1]]) + cen_poi_vec
    waypoint2 = np.array([poi_trace[0], poi_trace[1]]) - cen_poi_vec*0.65


    # if viz:
    #     plt.text(waypoint1[0]+7,waypoint1[1]+7, 'pt 1')
    #     plt.text(waypoint2[0]+7,waypoint2[1]+7, 'pt 2')
    #     plt.scatter(waypoint1[0],waypoint1[1])
    #     plt.scatter(waypoint2[0],waypoint2[1])
    #     plt.imshow(img.color._data)
    #     plt.show()

    waypoint1 = [int(waypoint1[0]), int(waypoint1[1])]
    waypoint2 = [int(waypoint2[0]), int(waypoint2[1])]

    if viz:
        plt.text(waypoint1[0]+7,waypoint1[1]+7, 'pt 1')
        plt.text(waypoint2[0]+7,waypoint2[1]+7, 'pt 2')
        plt.scatter(waypoint1[0],waypoint1[1])
        plt.scatter(waypoint2[0],waypoint2[1])
        plt.imshow(img.color._data)
        plt.show()

    iface = fullPipeline.iface
    iface.close_grippers()
    iface.sync()
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi

    waypoint1 = [int(waypoint1[0]), int(waypoint1[1])]
    waypoint2 = [int(waypoint2[0]), int(waypoint2[1])]

    print('waypoint1: ', waypoint1)
    print('waypoint2: ', waypoint2)
    ## WORLD COORDINATE ESTIMATION WITH DEPTH
    # place1 = g.ij_to_point(waypoint1).data # convert pixel coordinates to 3d coordinates
    # place2 = g.ij_to_point(waypoint2).data # convert pixel coordinates to 3d coordinates
    # print('with depth calculation: ')
    # print(place1)
    # print(place2)
    ## WORLD COORDINATE ESTIMATION WITHOUT DEPTH
    # place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
    # place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
    # print('without depth calculation: ')
    # print(place1)
    # print(place2)

    average_place = (np.array(waypoint1) + np.array(waypoint2))/2
    fixed_depth = 0.0408
    if average_place[0] > 590:
        print('waypoint1 again: ', waypoint1)
        print('waypoint2 again: ', waypoint2)
        place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        intermediate_place1 = place1 + np.array([0, 0, 0.1])
        intermediate_place2 = place2 + np.array([0, 0, 0.1])
        # place1 = [place1[0], place1[1], fixed_depth]
        # place2 = [place2[0], place2[1], fixed_depth]
        # print('without depth calculation: ')
        print(place1)
        print(place2)
        # right arm
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        intermediate_place2_transform = RigidTransform(
            translation=intermediate_place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

        place1_transform = RigidTransform(
            translation=place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        place2_transform = RigidTransform(
            translation=place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        try:
            iface.go_cartesian(r_targets=[intermediate_place1_transform], removejumps=[6])
            iface.sync()

            iface.go_cartesian(r_targets=[place1_transform], removejumps=[6])
            iface.sync()

            iface.go_cartesian(r_targets=[place2_transform], removejumps=[6])
            iface.sync()

            iface.go_cartesian(r_targets=[intermediate_place2_transform], removejumps=[6])
            iface.sync()
        except RuntimeError:
            print("stuck?") 
    else:
        # left arm
        place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        intermediate_place1 = place1 + np.array([0, 0, 0.1])
        intermediate_place2 = place2 + np.array([0, 0, 0.1])
        # place1 = [place1[0], place1[1], fixed_depth]
        # place2 = [place2[0], place2[1], fixed_depth]
        # print('without depth calculation: ')
        print(place1)
        print(place2)
        
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        intermediate_place2_transform = RigidTransform(
            translation=intermediate_place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )


        place1_transform = RigidTransform(
            translation=place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        place2_transform = RigidTransform(
            translation=place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        try:
            iface.go_cartesian(l_targets=[intermediate_place1_transform], removejumps=[6])
            iface.sync()

            iface.go_cartesian(l_targets=[place1_transform], removejumps=[6])
            iface.sync()

            iface.go_cartesian(l_targets=[place2_transform], removejumps=[6])
            iface.sync()

            iface.go_cartesian(l_targets=[intermediate_place2_transform], removejumps=[6])
            iface.sync()
        except RuntimeError:
            print("stuck?")

    iface.open_grippers()
    iface.sync()
    iface.home()
    iface.sync()
    time.sleep(2)

if __name__ == "__main__":
 
    fullPipeline = FullPipeline(viz=False, loglevel=logging.INFO, initialize_iface=True)
    fullPipeline.iface.speed = (0.2, 6 * np.pi)
    fullPipeline.iface.open_grippers()
    fullPipeline.iface.sync()
    fullPipeline.iface.home()
    fullPipeline.iface.sync()
    viz = True

    img = fullPipeline.iface.take_image()

    fullPipeline.get_endpoints()
    print("Choose endpoint")
    
    chosen_endpoint, _ = click_points_closest(img.color._data, fullPipeline.endpoints)
    print("Chosen endpoint:", chosen_endpoint)

    starting_pixels, analytic_trace_problem = fullPipeline.get_trace_from_endpoint(chosen_endpoint)
    starting_pixels = np.array(starting_pixels)
    trace_t, _ = run_tracer_with_transform(img.color._data, 0, starting_pixels, endpoints = fullPipeline.endpoints)

    visualize_trace(img.color._data, trace_t[0])
    push_coord, _ = click_points_simple(img) 

    poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img.color._data, trace_t[0], type="poles", viz=True)
    perform_push_through(poi_trace, cen_poi_vec, img, fullPipeline)
