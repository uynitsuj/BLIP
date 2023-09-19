# 3 lines of detectron imports
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
plt.set_loglevel('info')
# plt.style.use('seaborn-darkgrid')

torch.cuda.empty_cache()

def l_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(
        translation=trans,
        rotation=rot,
        from_frame=YK.l_tcp_frame,
        to_frame=YK.base_frame,
    )

def r_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(
        translation=trans,
        rotation=rot,
        from_frame=YK.r_tcp_frame,
        to_frame=YK.base_frame,
    )

def run_pipeline():
    global t2, _FINISHED
    SPEED = (0.6, 2 * np.pi) #(0.6, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )

    for i in range(100):
        iface.home()
        iface.sync()

        iface.go_cartesian(
            r_targets=[r_p([0.2, 0.1, 0.1], iface.GRIP_DOWN_R)],
            removejumps=[5, 6],
            nwiggles=(2, 2),
            rot=(0.3, 0.3),
        )

        iface.sync()

    # iface.go_cartesian(
    #     r_targets=[r_p([0.4, 0.0, 0.2], iface.GRIP_DOWN_R)],
    #     removejumps=[5, 6],
    #     nwiggles=(2, 2),
    #     rot=(0.3, 0.3),
    # )


    
    
if __name__ == "__main__":
    run_pipeline()