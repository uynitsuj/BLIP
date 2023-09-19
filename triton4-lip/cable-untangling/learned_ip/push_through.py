from untangling.point_picking import click_points_simple
import cv2
import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
from scripts.full_pipeline_trunk import FullPipeline

from scripts.full_pipeline_trunk import FullPipeline
import argparse
import logging
import numpy as np


def get_closest_trace_idx(pix, trace):
    distances = np.linalg.norm(np.array(trace) - np.array(pix)[None, ...], axis=1)
    return np.argmin(distances)

def is_valid_cc(cc_stats, cc_labels, cc):
    return is_large_cc(cc_stats, cc) and cc > 0

def is_large_cc(cc_stats, cc):
    return cc_stats[cc][4] > 1000 # minimum cc pixel area of 1000 may be tuned

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
    #plt.imshow(dist_transform)
    #plt.show()

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    adjusted_max_loc = (max_loc[0]-1, max_loc[1]-1)
    poles = adjusted_max_loc

    return poles


def get_poi_and_vec_for_push(fullPipeline, type="poles", viz=True):
    img = fullPipeline.iface.take_image()
     
    fullPipeline.get_endpoints()
    starting_pixels, analytic_trace_problem = fullPipeline.get_trace_from_endpoint(fullPipeline.endpoints[0])
    starting_pixels = np.array(starting_pixels)
    traces = []    
    fullPipeline.tkd._set_data(fullPipeline.img.color._data, starting_pixels)
    perception_result = fullPipeline.tkd.perception_pipeline(endpoints=fullPipeline.endpoints, viz=False, vis_dir=None) #do_perturbations=True
    traces.append(fullPipeline.tkd.pixels)
    trace = traces[0]

    pick_coord, _ = click_points_simple(img) 
    poi_trace = trace[get_closest_trace_idx([pick_coord[1],pick_coord[0]], trace)]
    poi_trace = [poi_trace[1], poi_trace[0]]
    poi_closest = trace[get_closest_trace_idx([pick_coord[1],pick_coord[0]], trace)+1]
    poi_closest = [poi_closest[1], poi_closest[0]]

    if viz:
        plt.scatter(poi_trace[0], poi_trace[1])
        plt.scatter(poi_closest[0], poi_closest[1])
        plt.imshow(img.color._data)
        plt.show()

    
    poi_tan_vec = (np.array(poi_trace) - np.array(poi_closest))/np.linalg.norm(np.array(poi_trace) - np.array(poi_closest)) #unit vector
    place_box_img_W = 300
    cropped_img = img.color._data[poi_trace[1]-place_box_img_W//2:poi_trace[1]+place_box_img_W//2, poi_trace[0]-place_box_img_W//2:poi_trace[0]+place_box_img_W//2, 0]

    thresh, maxValue = 127, 255
    _, binary_img = cv2.threshold(cropped_img, thresh, maxValue, cv2.THRESH_BINARY)
    if viz:
        plt.imshow(binary_img)
        plt.show()

    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if binary_img[i][j] == 255:
                binary_img[i][j] = 0
            else:
                binary_img[i][j] = 1

    num_labels, labels, stats, centroids= cv2.connectedComponentsWithStats(binary_img)
    
    if viz:
        plt.scatter(int(binary_img.shape[0]/2),int(binary_img.shape[1]/2))

    maxdotprod = 0
    maxpoint = None
    maxlabel = -1
    for label in range(num_labels):
        if is_valid_cc(stats, labels, label): # minimum cc pixel area of 1000 may be tuned

            if type == "poles":
                center_point = get_pole_of_inaccessibility(labels=labels, label = label)
                plt.title("Per-Region Centroid")
            else:
                center_point = centroids[label]
                plt.title("Per-Region Poles of Inaccessibility")
            

            plt.scatter(center_point[0],center_point[1])
            pointerest_center_vec = (np.array(center_point) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))/np.linalg.norm(np.array(center_point) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))

            
            print("vector: ",pointerest_center_vec)
            print(np.dot(poi_tan_vec, pointerest_center_vec))

            
            plt.text(center_point[0]-5,center_point[1]-5, str(round(np.dot(poi_tan_vec, pointerest_center_vec), 2)), fontsize=14)
            
            
            poidotprod = np.abs(np.dot(poi_tan_vec, pointerest_center_vec))
            if poidotprod > maxdotprod:
                maxdotprod = poidotprod
                maxlabel = label
                maxpoint = center_point
    if viz:
        plt.imshow(labels)
        plt.show()

    
    cen_poi_vec = (np.array(maxpoint) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
    return poi_trace, cen_poi_vec





if __name__ == "__main__":
    fullPipeline = FullPipeline(viz=False, loglevel=logging.INFO, initialize_iface=True)
    fullPipeline.iface.speed = (0.2, 6 * np.pi)
    fullPipeline.iface.open_grippers()
    fullPipeline.iface.home()
    fullPipeline.iface.sync()
    viz = True

    poi_trace, cen_poi_vec = get_poi_and_vec_for_push(fullPipeline, type="poles", viz=viz)
    waypoint1 = np.array([poi_trace[0], poi_trace[1]]) + cen_poi_vec
    waypoint2 = np.array([poi_trace[0], poi_trace[1]]) - cen_poi_vec

    if viz:
        plt.text(waypoint1[0]+7,waypoint1[1]+7, 'pt 1')
        plt.text(waypoint2[0]+7,waypoint2[1]+7, 'pt 2')
        plt.scatter(waypoint1[0],waypoint1[1])
        plt.scatter(waypoint2[0],waypoint2[1])
        plt.imshow(fullPipeline.img.color._data)
        plt.show()

    iface = fullPipeline.iface
    iface.close_grippers()
    iface.sync()
    g = GraspSelector(fullPipeline.img, iface.cam.intrinsics, iface.T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi

    waypoint1 = [int(waypoint1[0]), int(waypoint1[1])]
    waypoint2 = [int(waypoint2[0]), int(waypoint2[1])]

    place1 = g.ij_to_point(waypoint1).data # convert pixel coordinates to 3d coordinates
    place2 = g.ij_to_point(waypoint2).data # convert pixel coordinates to 3d coordinates

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


    iface.go_cartesian(
       r_targets=[place1_transform], removejumps=[6])
    iface.sync()

    iface.go_cartesian(
        r_targets=[place2_transform], removejumps=[6])
    iface.sync()

    
    



