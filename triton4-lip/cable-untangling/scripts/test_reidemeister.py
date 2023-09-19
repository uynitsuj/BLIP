from codecs import IncrementalDecoder
from pickle import FALSE
from turtle import done, left
import analysis as loop_detectron
from untangling.utils.grasp import GraspSelector, GraspException
from untangling.utils.interface_rws import Interface
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
import numpy as np
from untangling.utils.tcps import *
# from untangling.utils.circle_BFS import trace, trace_white
from cable_tracing.tracers.simple_uncertain_trace import trace
# from untangling.utils.cable_tracing.tracers.simple_uncertain_trace import trace
#from untangling.slide import FCNNetworkStopCond, knot_in_hand, endpoint_in_hand
from untangling.shake import shake
import time
from untangling.point_picking import *
import matplotlib.pyplot as plt
from fcxvision.kp_wrapper import KeypointNetwork
from untangling.spool import init_endpoint_orientation,execute_spool
from untangling.keypoint_untangling import closest_valid_point_to_pt
import torch
import cv2
import threading
from queue import Queue
import signal
import os
import os.path as osp
import datetime
import logging 
import argparse
import matplotlib.pyplot as plt
from full_pipeline_trunk import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print more statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO,
    )

    args = parser.parse_args()
    logLevel = args.loglevel
    fullPipeline = FullPipeline(viz=True, loglevel=logLevel)

    SPEED = (0.6, 2 * np.pi) #(0.6, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )

    while True:
        fullPipeline.iface.open_grippers()
        fullPipeline.iface.open_arms()
        time.sleep(1)
        fullPipeline.img = fullPipeline.take_and_save_image()
        fullPipeline.get_endpoints()
        fullPipeline.run_endpoint_separation_incremental()
        input("Again?")