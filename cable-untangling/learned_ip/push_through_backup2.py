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
from autolab_core.transformations import rotation_matrix
from scripts.full_pipeline_trunk import FullPipeline

from collections import OrderedDict
import argparse
import logging
import numpy as np
import colorsys

DIST_TO_TABLE_LEFT = 1.025 # meters
DIST_TO_TABLE_RIGHT = 0.99 # meters
T_CAM_BASE = RigidTransform.load("/home/mallika/triton4-lip/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")

Z_OFFSET = 1.4e-3

def get_world_coord_from_pixel_coord(pixel_coord, cam_intrinsics):
    '''
    pixel_coord: [x, y] in pixel coordinates
    cam_intrinsics: 3x3 camera intrinsics matrix
    '''
    height_samples = [0.043108, 0.041911, 0.0420478, 0.04362216]

    pixel_coord = np.array(pixel_coord)
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(np.r_[pixel_coord, 1.0])
    # print(T_CAM_BASE.matrix)
    # PHOXI_TO_WORLD = np.eye(4)
    # PHOXI_TO_WORLD[:3,:3] = np.array([[0,-1,0],[-1,0,0],[0,0,-1]])
    # PHOXI_TO_WORLD[:,3] = np.array([T_CAM_BASE.matrix[0,3], T_CAM_BASE.matrix[1,3], T_CAM_BASE.matrix[2,3], 1.0])
    # print(PHOXI_TO_WORLD)
    point_3d_world = T_CAM_BASE.matrix.dot(np.r_[point_3d_cam, 1.0])
    # print(T_CAM_BASE.matrix)
    # print(point_3d_world)
    point_3d_world = point_3d_world[:3]/point_3d_world[3]
    # print(point_3d_world)
    point_3d_world[-1] = np.mean(height_samples)
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
    # print(cc_stats[cc][4])
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
def get_pole(labels, label):
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

    #plot gaussian
    gaussian = gaussian_2d(padded_labels.shape[0], padded_labels.shape[1])
    # plt.clf()
    # plt.imshow(gaussian)
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

    
def get_poi_and_vec_for_push(push_coord, img_rgb, trace, place_box_img_W=150, edge_case=False, viz=True, y_buffer=0, x_buffer=0):
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

    poidotproducts = []
    candidate_pts = []
    for label in range(num_labels):
        if is_valid_cc(stats, labels, label): # minimum cc pixel area of 1000 may be tuned
            center_point = get_pole(labels=labels, label = label)
            plt.title("Per-Region Poles of Inaccessibility")

            plt.scatter(center_point[0],center_point[1])
            pointerest_center_vec = (np.array(center_point) - poi_cropped)/np.linalg.norm(np.array(center_point) - poi_cropped)
            
            plt.text(center_point[0]-5,center_point[1]-5, str(round(np.dot(poi_tan_vec, pointerest_center_vec), 2)), fontsize=14)
            
            poidotproducts.append(np.dot(poi_tan_vec, pointerest_center_vec))
            candidate_pts.append(center_point)

    if viz:
        plt.imshow(labels)
        plt.show()

    def check_in_buffer(pt):
        pt = [pt[1], pt[0]]
        if pt[0] > (img_rgb.shape[0] - y_buffer):
            return True
        if pt[0] < y_buffer:
            return True
        if pt[1] > (img_rgb.shape[1] - x_buffer):
            return True
        if pt[1] < x_buffer:
            return True
        return False

    if edge_case is True:
        keep_candidate_pts = []
        keep_poidotproducts = []
        for pt, poidotproduct in zip(candidate_pts, poidotproducts):
            if check_in_buffer(pt):
                keep_candidate_pts.append(pt)
                keep_poidotproducts.append(poidotproduct)
        candidate_pts = keep_candidate_pts
        poidotproducts = keep_poidotproducts
        chosen_pt = candidate_pts[poidotproducts.index(min(poidotproducts))]
    else:
        chosen_pt = candidate_pts[poidotproducts.index(max(poidotproducts))]

    cen_poi_vec = (np.array(chosen_pt) - poi_cropped)
    return poi_trace, cen_poi_vec


def get_pindown_gripper_rot(poi, pt1, pt2):
    slope = (pt2[1] - pt1[1])/(pt2[0] - pt1[0])
    yaw = np.arctan(slope)
    gripper_rot = rotation_matrix(-yaw, [0, 0, 1], poi)[:3,:3]
    return gripper_rot

def color_for_pct(pct):
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(pct, 1, 1))

def visualize_trace(img, trace, save=False):
    img_copy = img.copy()
    for i in range(len(trace) - 1):
        pt1, pt2 = get_trace_points(trace, i)
        cv2.line(img_copy, pt1[::-1], pt2[::-1], color_for_pct(i / len(trace)), 4)
    display_image(img_copy, trace, save)

def get_trace_points(trace, idx):
    if not isinstance(trace, OrderedDict):
        return tuple(trace[idx].astype(int)), tuple(trace[idx+1].astype(int))
    trace_keys = list(trace.keys())
    return trace_keys[idx], trace_keys[idx + 1]

def display_image(img, trace, save):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Trace Visualized")
    if save:
        plt.savefig("multicable_baseline.png")
    else:
        plt.show()

def perform_pindown(trace, push_coord, normalized_cable_density, img, fullPipeline):

    iface = fullPipeline.iface
    iface.close_grippers()
    iface.sync()

    pin_arm = None

    if push_coord[1] > 590:
        pin_arm = 'left'

        iface.open_gripper('left')
        iface.sync()
        radoe = [np.linalg.norm(trace[i]-push_coord) for i in range(len(trace))]
        # print(radoe)
        idx_keep = [i for i in range(len(trace)) if (trace[i][1] <= 590 and radoe[i]<230 and radoe[i]>150)]
        if len(idx_keep)==0:
            return None
        keep_normalized_cable_density = [normalized_cable_density[i] for i in idx_keep]
        trace_idx = idx_keep[keep_normalized_cable_density.index(min(keep_normalized_cable_density))]
        pindown_pt = trace[trace_idx]
        pindown_pt = [int(pindown_pt[1]), int(pindown_pt[0])]

        # visualize_trace(img.color._data, [trace[i] for i in idx_keep])
        # plt.imshow(img.color._data)
        # for i in [trace[i] for i in idx_keep]:
        #     plt.scatter(i[1],i[0], c='b', s=1)
        # plt.scatter(pindown_pt[0], pindown_pt[1], c='r')
        # plt.title('pindown pt')
        # plt.show()

        place1 = get_world_coord_from_pixel_coord(pindown_pt, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates

        x1, x2 = get_world_coord_from_pixel_coord(trace[trace_idx-1], iface.cam.intrinsics), get_world_coord_from_pixel_coord(trace[trace_idx+1], iface.cam.intrinsics)
        gripper_rot = get_pindown_gripper_rot(place1, x1, x2)

        intermediate_place1 = place1 + np.array([0, 0, 0.07])

        gripper_rot = gripper_rot @ iface.GRIP_DOWN_R

        # left arm
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= gripper_rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        place1_transform = RigidTransform(
            translation=place1,
            rotation= gripper_rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        iface.go_cartesian(l_targets=[intermediate_place1_transform,place1_transform], removejumps=[6])
        iface.sync()

        # iface.go_cartesian(l_targets=[place1_transform], removejumps=[6])
        # iface.sync()

        iface.close_gripper('left')
        iface.sync()

    else:
        pin_arm = 'right'

        iface.open_gripper('right')
        iface.sync()
        radoe = [np.linalg.norm(trace[i]-push_coord) for i in range(len(trace))]
        idx_keep = [i for i in range(len(trace)) if (trace[i][1] > 590 and radoe[i]<220 and radoe[i]>150)]
        # idx_keep = [i for i in range(len(trace)) if trace[i][1] > 590]
        if len(idx_keep)==0:
            return None
        keep_normalized_cable_density = [normalized_cable_density[i] for i in idx_keep]
        trace_idx = idx_keep[keep_normalized_cable_density.index(min(keep_normalized_cable_density))]
        
        pindown_pt = trace[trace_idx]
        pindown_pt = [int(pindown_pt[1]), int(pindown_pt[0])]

        # plt.imshow(img.color._data)
        # plt.scatter(pindown_pt[0], pindown_pt[1])
        # plt.title('pindown pt')
        # plt.show()

        place1 = get_world_coord_from_pixel_coord(pindown_pt, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        intermediate_place1 = place1 + np.array([0, 0, 0.07])

        x1, x2 = get_world_coord_from_pixel_coord(trace[trace_idx-1], iface.cam.intrinsics), get_world_coord_from_pixel_coord(trace[trace_idx+1], iface.cam.intrinsics)
        gripper_rot = get_pindown_gripper_rot(place1, x1, x2)

        intermediate_place1 = place1 + np.array([0, 0, 0.07])

        gripper_rot = gripper_rot @ iface.GRIP_DOWN_R

        # right arm
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= gripper_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

        place1_transform = RigidTransform(
            translation=place1,
            rotation= gripper_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        iface.go_cartesian(r_targets=[intermediate_place1_transform,place1_transform], removejumps=[6])
        iface.sync()

        # iface.go_cartesian(r_targets=[place1_transform], removejumps=[6])
        # iface.sync()

        iface.close_gripper('right')
        iface.sync()
    
    iface.sync()
    time.sleep(1)

    return pin_arm


def perform_push_through(poi_trace, cen_poi_vec, img, fullPipeline, viz=False, pin_arm=None):
    
    waypoint1 = np.array([poi_trace[0], poi_trace[1]]) + cen_poi_vec
    waypoint2 = np.array([poi_trace[0], poi_trace[1]]) - cen_poi_vec*0.7

    if viz:
        plt.text(waypoint1[0]+7,waypoint1[1]+7, 'pt 1')
        plt.text(waypoint2[0]+7,waypoint2[1]+7, 'pt 2')
        plt.scatter(waypoint1[0],waypoint1[1])
        plt.scatter(waypoint2[0],waypoint2[1])
        plt.imshow(img.color._data)
        plt.show()

    iface = fullPipeline.iface
    if pin_arm is not None:
        iface.close_grippers()
        iface.sync()
    # print(h)
    waypoint1 = [int(waypoint1[0]), int(waypoint1[1])]
    waypoint2 = [int(waypoint2[0]), int(waypoint2[1])]

    print('waypoint1: ', waypoint1)
    print('waypoint2: ', waypoint2)
    

    average_place = (np.array(waypoint1) + np.array(waypoint2))/2
    fixed_depth = 0.0419
    #fixed_depth = 0.0935
    if average_place[0] > 590:
        place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], fixed_depth]
        place2 = [place2[0], place2[1], fixed_depth]
        print(place1)
        print(place2)
        intermediate_place1 = place1 + np.array([0, 0, 0.07])
        intermediate_place2 = place2 + np.array([0, 0, 0.07])
        intermediate_place3 = np.array([.4, -.25, 0.12])
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
        intermediate_place3_transform = RigidTransform(
            translation=intermediate_place3,
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
        iface.go_cartesian(r_targets=[intermediate_place1_transform, place1_transform, place2_transform, intermediate_place2_transform, intermediate_place3_transform], removejumps=[6])
        iface.sync()

        # iface.go_cartesian(r_targets=[place1_transform], removejumps=[6])
        # iface.sync()

        # iface.go_cartesian(r_targets=[place2_transform], removejumps=[6])
        # iface.sync()

        # iface.go_cartesian(r_targets=[intermediate_place2_transform], removejumps=[6])
        # iface.sync()

    else:
        place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], fixed_depth]
        place2 = [place2[0], place2[1], fixed_depth]
        print(place1)
        print(place2)
        intermediate_place1 = place1 + np.array([0, 0, 0.07])
        intermediate_place2 = place2 + np.array([0, 0, 0.07])
        intermediate_place3 = np.array([0.4, 0.25, 0.12])
        
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
        intermediate_place3_transform = RigidTransform(
            translation=intermediate_place3,
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

        iface.go_cartesian(l_targets=[intermediate_place1_transform, place1_transform, place2_transform, intermediate_place2_transform, intermediate_place3_transform], removejumps=[6])
        iface.sync()

        # iface.go_cartesian(l_targets=[place1_transform], removejumps=[6])
        # iface.sync()

        # iface.go_cartesian(l_targets=[place2_transform], removejumps=[6])
        # iface.sync()

        # iface.go_cartesian(l_targets=[intermediate_place2_transform], removejumps=[6])
        # iface.sync()

    if pin_arm is not None:
        time.sleep(1)
        iface.open_gripper(pin_arm)
        iface.sync()
    
    iface.home()
    iface.sync()
    time.sleep(1)
    iface.close_grippers()
    iface.sync()
    time.sleep(1)

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
    place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
    place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
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
