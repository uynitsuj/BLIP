from untangling.point_picking import click_points_simple
import sys
import numpy as np
import argparse
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
sys.path.append('/home/mallika/triton4-lip/lip_tracer/') 
sys.path.append('/home/mallika/triton4-lip/cable-untangling/learned_ip/') 
from learn_from_demos_ltodo import DemoPipeline
import logging


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

if __name__ == "__main__":
    SPEED = (0.4, 6 * np.pi)

    args = parse_args()
    logLevel = args.loglevel
    start_time = time.time()
    fullPipeline = initialize_pipeline(logLevel)
    
    iface = fullPipeline.iface

    iface.home()
    iface.sync()
    time.sleep(4)

    img = fullPipeline.iface.take_image()
    print("Choose place points")
    place_1, _= click_points_simple(img)


    plt.scatter(place_1[0], place_1[1], c='r')
    plt.imshow(img.color._data)
    plt.show()

    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    place1_point = g.ij_to_point(place_1).data
    print(place1_point)

    place1_transform = RigidTransform(
        translation=place1_point,
        rotation= iface.GRIP_DOWN_R,
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )
    # iface.go_pose_plan(l_target=place1_transform)
    iface.go_cartesian(
        l_targets=[place1_transform], removejumps=[6])
    iface.sync()
    iface.sync()
    time.sleep(5)

