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
from untangling.utils.reveal_endpoint import reveal_endpoint
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

def get_endpoints(img):
    endpoint_boxes, _ = loop_detectron.predict(img.color._data, thresh=0.05, endpoints=True)
    endpoint_boxes = endpoint_boxes['instances'].to('cpu')
    endpoints = []
    for box in endpoint_boxes:
        xmin, ymin, xmax, ymax = box
        x = (xmin + xmax)/2
        y = (ymin + ymax)/2
        endpoints.append([[x,y], [x,y]])      
    logger.debug("endpoints: ", endpoints)
    # (n,2,2) (num endpoints, (head point, neck point), (x,y))
    endpoints = np.array([e for e in endpoints if img.depth._data[e[0][0], e[0][1]] > 0])
    logger.debug("Found {} true endpoints after filtering for depth".format(len(endpoints)))
    return endpoints.astype(np.int32).reshape(-1, 2, 2), None
