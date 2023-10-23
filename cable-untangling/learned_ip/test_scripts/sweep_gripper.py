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
T_PHOXI_BASE = RigidTransform.load("/home/mallika/triton4-lip/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")

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

def execute2(img, waypoint1, waypoint2):
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
    g = GraspSelector(img, iface.cam.intrinsics, T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi
    w = img.shape[1]
    h = img.shape[0]
    print(h)
    waypoint1 = [int(waypoint1[0]), int(waypoint1[1])]
    waypoint2 = [int(waypoint2[0]), int(waypoint2[1])]

    print('waypoint1: ', waypoint1)
    print('waypoint2: ', waypoint2)
    

    average_place = (np.array(waypoint1) + np.array(waypoint2))/2
    average_place = (np.array(waypoint1) + np.array(waypoint2))/2
    fixed_depth = 0.0478

    if average_place[0] > 590:
        print('waypoint1 again: ', waypoint1)
        print('waypoint2 again: ', waypoint2)
        place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        intermediate_place1 = place1 + np.array([0, 0, 0.1])
        # place1 = [place1[0], place1[1], fixed_depth]
        # place2 = [place2[0], place2[1], fixed_depth]
        print('without depth calculation: ')
        print(place1)
        print(place2)
        # right arm
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
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

    else:
        # left arm
        place1 = get_world_coord_from_pixel_coord(waypoint1, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        intermediate_place1 = place1 + np.array([0, 0, 0.1])
        # place1 = [place1[0], place1[1], fixed_depth]
        # place2 = [place2[0], place2[1], fixed_depth]
        print('without depth calculation: ')
        print(place1)
        print(place2)
        
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
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

    iface.open_grippers()
    iface.sync()
    iface.home()
    iface.sync()
    time.sleep(2)

def execute(img, waypoint1, waypoint2):
    #fixed_depth = 0.0478

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
    g = GraspSelector(img, iface.cam.intrinsics, T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi
    w = img.shape[1]
    h = img.shape[0]
    print(h)
    waypoint1 = [int(waypoint1[0]), int(waypoint1[1])]
    waypoint2 = [int(waypoint2[0]), int(waypoint2[1])]

    print('waypoint1: ', waypoint1)
    print('waypoint2: ', waypoint2)
    

    average_place = (np.array(waypoint1) + np.array(waypoint2))/2
    fixed_depth = 0.0418
    if average_place[0] > 590:
        print("right")
        print('waypoint1 again: ', waypoint1)
        print('waypoint2 again: ', waypoint2)
        # WORLD COORDINATE ESTIMATION WITH DEPTH
        place1 = g.ij_to_point(waypoint1).data # convert pixel coordinates to 3d coordinates
        place2 = g.ij_to_point(waypoint2).data # convert pixel coordinates to 3d coordinates
        print('with depth calculation: ')
        print(place1)
        print(place2)
        place1 = [place1[0], place1[1], fixed_depth]
        place2 = [place2[0], place2[1], fixed_depth]
        intermediate_place1 = place1 + np.array([0, 0, 0.05])
        print('without depth calculation fixed z: ')
        print(place1)
        print(place2)
        # right arm
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
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

    else:
        print("left")
        # left arm
        # WORLD COORDINATE ESTIMATION WITH DEPTH

        print('waypoint1 again: ', waypoint1)
        print('waypoint2 again: ', waypoint2)
        place1 = g.ij_to_point(waypoint1).data # convert pixel coordinates to 3d coordinates
        place2 = g.ij_to_point(waypoint2).data # convert pixel coordinates to 3d coordinates
        print('with depth calculation: ')
        print(place1)
        print(place2)
        place1 = [place1[0], place1[1], fixed_depth]
        place2 = [place2[0], place2[1], fixed_depth]
        intermediate_place1 = place1 + np.array([0, 0, 0.05])
        print('without depth calculation fixed z: ')
        print(place1)
        print(place2)
        
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
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

    iface.sync()
    #iface.home()
    iface.sync()
    time.sleep(2)

if __name__ == "__main__":
 
    fullPipeline = FullPipeline(viz=False, loglevel=logging.INFO, initialize_iface=True)
    fullPipeline.iface.speed = (0.2, 6 * np.pi)
    fullPipeline.iface.open_grippers()
    fullPipeline.iface.sync()
    fullPipeline.iface.home()
    fullPipeline.iface.sync()
    iface = fullPipeline.iface
    viz = True
    img = fullPipeline.iface.take_image()
    waypoint1 = (800, 300)
    waypoint2 = (750, 300)

    execute2(img, waypoint1, waypoint2)


    # place1 = [0.389376, 0.095717, 0.0408]
    # place1_transform = RigidTransform(
    #         translation=place1,
    #         rotation= iface.GRIP_DOWN_R,
    #         from_frame=YK.l_tcp_frame,
    #         to_frame="base_link",
    #     )
    # iface.go_cartesian(l_targets=[place1_transform], removejumps=[6])
    # iface.sync()