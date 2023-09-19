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
    #plt.clf()
    #plt.imshow(dist_transform)
    #plt.show()

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    adjusted_max_loc = (max_loc[0]-1, max_loc[1]-1)
    poles = adjusted_max_loc

    return poles



if __name__ == "__main__":
    fullPipeline = FullPipeline(viz=False, loglevel=logging.INFO, initialize_iface=True)
    fullPipeline.iface.open_grippers()
    fullPipeline.iface.home()
    fullPipeline.iface.sync()

    img = fullPipeline.iface.take_image() # camera gets snapshot: rgb (color, actually grayscale on photoneo rip) and depth
    #np.save("img2.npy", img.color._data)
    # img = iface.cam.read()
    #print(img.color._data.shape)
    #print("Choose pick points")


    fullPipeline.get_endpoints()
    starting_pixels, analytic_trace_problem = fullPipeline.get_trace_from_endpoint(fullPipeline.endpoints[0])
    starting_pixels = np.array(starting_pixels)
    traces = []    
    fullPipeline.tkd._set_data(fullPipeline.img.color._data, starting_pixels)
    perception_result = fullPipeline.tkd.perception_pipeline(endpoints=fullPipeline.endpoints, viz=False, vis_dir=None) #do_perturbations=True
    traces.append(fullPipeline.tkd.pixels)
    trace = traces[0]

    #print("trace", trace)

    pick_coord, _ = click_points_simple(img) 

    print(pick_coord)
    print(trace[get_closest_trace_idx([pick_coord[1],pick_coord[0]], trace)])
    poi_trace = trace[get_closest_trace_idx([pick_coord[1],pick_coord[0]], trace)]
    poi_trace = [poi_trace[1], poi_trace[0]]

    poi_closest = trace[get_closest_trace_idx([pick_coord[1],pick_coord[0]], trace)+1]
    poi_closest = [poi_closest[1], poi_closest[0]]

    plt.scatter(poi_trace[0], poi_trace[1])
    plt.scatter(poi_closest[0], poi_closest[1])
    plt.imshow(img.color._data)
    plt.show()

    poi_tan_vec = (np.array(poi_trace) - np.array(poi_closest))/np.linalg.norm(np.array(poi_trace) - np.array(poi_closest)) #unit vector


    place_box_img_W = 300
    #cropped_img = img.color._data[pick_coord[1]-place_box_img_W//2:pick_coord[1]+place_box_img_W//2, pick_coord[0]-place_box_img_W//2:pick_coord[0]+place_box_img_W//2, 0]
    cropped_img = img.color._data[poi_trace[1]-place_box_img_W//2:poi_trace[1]+place_box_img_W//2, poi_trace[0]-place_box_img_W//2:poi_trace[0]+place_box_img_W//2, 0]

    thresh, maxValue = 127, 255
    _, binary_img = cv2.threshold(cropped_img, thresh, maxValue, cv2.THRESH_BINARY)

    print(binary_img)

    plt.imshow(binary_img)
    plt.show()

    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if binary_img[i][j] == 255:
                binary_img[i][j] = 0
            else:
                binary_img[i][j] = 1
    #print(binary_img)
    num_labels, labels, stats, centroids= cv2.connectedComponentsWithStats(binary_img)
    print("num_labels", num_labels)
    #print(labels)
    plt.scatter(int(binary_img.shape[0]/2),int(binary_img.shape[1]/2))
    print("tangent vector: ", poi_tan_vec)
    print(stats)
    maxdotprod = 0
    # maxcentroid = None
    maxpoint = None
    maxlabel = -1
    for label in range(num_labels):
        if is_valid_cc(stats, labels, label): # minimum cc pixel area of 1000 may be tuned
            #print("Centroid for label " + str(label) + ": ")
            #print(centroids[label])
            #print("POInacc. for label " + str(label) + ": ")
            poinacc = get_pole_of_inaccessibility(labels=labels, label = label)
            centroid = centroids[label]
            point = poinacc # replace w poinacc or centroid
            #plt.title("Per-Region Centroid")
            plt.scatter(point[0],point[1])
            #plt.scatter(poinacc[0], poinacc[1])

            pointerest_center_vec = (np.array(point) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))/np.linalg.norm(np.array(point) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
            #pointerest_centroid_vec = (np.array(centroids[label]) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))/np.linalg.norm(np.array(centroids[label]) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
            # pointerest_poinacc_vec = (np.array(poinacc) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))/np.linalg.norm(np.array(poinacc) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
            
            print("vector: ",pointerest_center_vec)
            print(np.dot(poi_tan_vec, pointerest_center_vec))

            
            plt.text(point[0]-5,point[1]-5, str(round(np.dot(poi_tan_vec, pointerest_center_vec), 2)), fontsize=14)


            #plt.scatter(centroids[label][0],centroids[label][1])
            #plt.text(centroids[label][0]-5,centroids[label][1]-5, str(round(np.dot(poi_tan_vec, pointerest_centroid_vec), 2)))


            # print("poinacc vector: ")
            # print(pointerest_poinacc_vec)

            # plt.text(poinacc[0]-5, poinacc[1]-5, str(round(np.dot(poi_tan_vec, pointerest_poinacc_vec), 2)), fontsize=14)
            
            
            poidotprod = np.abs(np.dot(poi_tan_vec, pointerest_center_vec))
            if poidotprod > maxdotprod:
                maxdotprod = poidotprod
                maxlabel = label
                maxpoint = point
    
    plt.imshow(labels)
    plt.show()
    #cen_poi_vec = (np.array(maxcentroid) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
    #pt2 = (np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)])-np.array(cen_poi_vec))
    #print(pt2)
    
    cen_poi_vec = (np.array(maxpoint) - np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)]))
    # pt2 = (np.array([int(binary_img.shape[0]/2),int(binary_img.shape[1]/2)])-np.array(cen_poi_vec))
    # print(pt2)
    
    # plt.text(pt2[0]+7,pt2[1]+7, 'pt 2')

    # plt.text(maxcentroid[0]+7,maxcentroid[1]+7, 'pt 1')
    # plt.scatter(pt2[0],pt2[1])
    # plt.scatter(maxcentroid[0],maxcentroid[1])
    # plt.imshow(labels)
    # plt.show()
    waypoint1 = np.array([poi_trace[0], poi_trace[1]]) + cen_poi_vec
    waypoint2 = np.array([poi_trace[0], poi_trace[1]]) - cen_poi_vec
    plt.text(waypoint1[0]+7,waypoint1[1]+7, 'pt 1')

    plt.text(waypoint2[0]+7,waypoint2[1]+7, 'pt 2')
    plt.scatter(waypoint1[0],waypoint1[1])
    plt.scatter(waypoint2[0],waypoint2[1])
    plt.imshow(img.color._data)
    plt.show()

    iface = fullPipeline.iface
    # img = iface.take_image() 
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
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )
    place2_transform = RigidTransform(
        translation=place2,
        rotation= iface.GRIP_DOWN_R,
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )


    #iface.go_cartesian(
    #    l_targets=[place1_transform], removejumps=[6])
    #iface.sync()

    # iface.go_cartesian(
    #     l_targets=[place2_transform], removejumps=[6])
    # iface.sync()

    
    



