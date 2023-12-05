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
    time.sleep(1)

    input("Manually move left YuMi end effector to top left, touching foam, then press enter to get frame")
    top_left = iface.y.left.get_pose()
    print("Top Left: ", top_left.translation)
    input("Manually move left YuMi end effector to bottom left, touching foam, then press enter to get frame")
    bottom_left = iface.y.left.get_pose()
    print("Bottom Left: ", bottom_left.translation)
    input("Manually move left YuMi end effector to top right, touching foam, then press enter to get frame")
    top_right = iface.y.right.get_pose()
    print("Top Right: ", top_right.translation)
    input("Manually move left YuMi end effector to bottom right, touching foam, then press enter to get frame")
    bottom_right = iface.y.right.get_pose()
    print("Bottom Right: ", bottom_right.translation)
