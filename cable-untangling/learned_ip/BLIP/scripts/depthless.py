from untangling.point_picking import click_points_simple
import sys
import numpy as np
import argparse
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
import time
import cv2

from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
sys.path.append('/home/mallika/triton4-lip/lip_tracer/') 
sys.path.append('/home/mallika/triton4-lip/cable-untangling/learned_ip/') 
from push_through_backup2 import get_world_coord_from_pixel_coord
from learn_from_demos_ltodo import DemoPipeline
import logging
T_CAM_BASE = RigidTransform.load("/home/mallika/triton4-lip/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")

def click_points_simple(img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    left_coords,right_coords = None,None
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        # lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
        nonlocal left_coords,right_coords
        if(event.button==1):
            left_coords=coords
        elif(event.button==3):
            right_coords=coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords, right_coords

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print more debug statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()

def initialize_pipeline(logLevel):
    pipeline = DemoPipeline(viz=False, loglevel=logLevel, initialize_iface=True)
    pipeline.iface.open_grippers()
    pipeline.iface.sync()
    pipeline.iface.home()
    pipeline.iface.sync()
    return pipeline

def L_rigid_tf(world_tf):
    place_transform = RigidTransform(
        translation=world_tf,
        rotation= iface.GRIP_DOWN_R,
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )
    return place_transform

def R_rigid_tf(world_tf):
    place_transform = RigidTransform(
        translation=world_tf,
        rotation= iface.GRIP_DOWN_R,
        from_frame=YK.r_tcp_frame,
        to_frame="base_link",
    )
    return place_transform


def acquire_image(pipeline):
    img_rgbd = pipeline.iface.take_image()
    return img_rgbd, img_rgbd.color._data

if __name__ == "__main__":
    arm = 'right'
    SPEED = (0.4, 6 * np.pi)

    args = parse_args()
    logLevel = args.loglevel
    start_time = time.time()
    fullPipeline = initialize_pipeline(logLevel)
    
    iface = fullPipeline.iface
    iface.home()
    iface.sync()
    iface.close_grippers()
    time.sleep(1)

    rgbd, img = acquire_image(fullPipeline)
    
    CHECKERBOARD = (6,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        chessboard = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    cv2.imshow('img',chessboard)
    cv2.waitKey(1)
    place_1 = corners[0][0]
    print(place_1)
    place_2 = corners[5][0]
    place_3 = corners[-6][0]
    place_4 = corners[-1][0]
    print(place_2)
    print(T_CAM_BASE)
    place1_depthless = get_world_coord_from_pixel_coord(place_1, iface.cam.intrinsics)
    place2_depthless = get_world_coord_from_pixel_coord(place_2, iface.cam.intrinsics)
    place3_depthless = get_world_coord_from_pixel_coord(place_3, iface.cam.intrinsics)
    place4_depthless = get_world_coord_from_pixel_coord(place_4, iface.cam.intrinsics)
    print("Depthless", place1_depthless)

    if arm == 'left':
        place1_transform = L_rigid_tf(place1_depthless)
        place2_transform = L_rigid_tf(place2_depthless)
        place3_transform = L_rigid_tf(place3_depthless)
        place4_transform = L_rigid_tf(place4_depthless)
        iface.go_cartesian(
            l_targets=[place1_transform, place2_transform,place4_transform, place3_transform, place1_transform], removejumps=[6])
        iface.sync()
    else:
        place1_transform = R_rigid_tf(place1_depthless)
        place2_transform = R_rigid_tf(place2_depthless)
        place3_transform = R_rigid_tf(place3_depthless)
        place4_transform = R_rigid_tf(place4_depthless)
        iface.go_cartesian(
            r_targets=[place1_transform, place2_transform,place4_transform, place3_transform, place1_transform], removejumps=[6])
        iface.sync()

    iface.sync()
    time.sleep(5)
