from untangling.point_picking import click_points_simple
import sys
import numpy as np
import argparse
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
import cv2

from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
sys.path.append('/home/mallika/triton4-lip/lip_tracer/') 
sys.path.append('/home/mallika/triton4-lip/cable-untangling/learned_ip/') 
from push_through_backup2 import get_world_coord_from_pixel_coord
from learn_from_demos_ltodo import DemoPipeline
import logging
T_CAM_BASE = RigidTransform.load("/home/mallika/triton4-lip/phoxipy/tools/arducam_to_world_bww.tf").as_frames(from_frame="arducam", to_frame="base_link")
CONST = 1.25
ARDUCAM_INTRINSICS = np.array([[758.923378*CONST,   0.,         960],
 [  0.,         757.65464286*CONST, 540],
 [  0.,           0.,           1.        ]])

def get_world_coord_from_pixel_coord(pixel_coord, cam_intrinsics):
    '''
    pixel_coord: [x, y] in pixel coordinates
    cam_intrinsics: 3x3 camera intrinsics matrix
    '''
    height_samples = [0.043108, 0.041911, 0.0420478, 0.04362216]

    pixel_coord = np.array(pixel_coord)
    print(pixel_coord)
    point_3d_cam = np.linalg.inv(cam_intrinsics).dot(np.r_[pixel_coord, 1.0])
    print(point_3d_cam)

    point_3d_world = T_CAM_BASE.matrix.dot(np.r_[point_3d_cam, 1.0])

    point_3d_world = point_3d_world[:3]/point_3d_world[3]

    point_3d_world[-1] = np.mean(height_samples)
    print('non-homogenous = ', point_3d_world)
    return point_3d_world

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

if __name__ == "__main__":
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

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, img = cam.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    # print(corners)
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
    cv2.waitKey(0)
    place_1 = corners[0][0]
    print(place_1)
    place_2 = corners[5][0]
    place_3 = corners[-6][0]
    place_4 = corners[-1][0]
    print(place_2)
    print(T_CAM_BASE)
    place1_depthless = get_world_coord_from_pixel_coord(place_1, ARDUCAM_INTRINSICS)
    place2_depthless = get_world_coord_from_pixel_coord(place_2, ARDUCAM_INTRINSICS)
    place3_depthless = get_world_coord_from_pixel_coord(place_3, ARDUCAM_INTRINSICS)
    place4_depthless = get_world_coord_from_pixel_coord(place_4, ARDUCAM_INTRINSICS)
    print("Depthless", place1_depthless)

    place1_transform = L_rigid_tf(place1_depthless)
    place2_transform = L_rigid_tf(place2_depthless)
    place3_transform = L_rigid_tf(place3_depthless)
    place4_transform = L_rigid_tf(place4_depthless)
    # iface.go_pose_plan(l_target=place1_transform)
    iface.go_cartesian(
        l_targets=[place1_transform, place2_transform,place4_transform, place3_transform, place1_transform], removejumps=[6])
    iface.sync()
    # time.sleep(1)
    # iface.home()
    iface.sync()
    time.sleep(5)
