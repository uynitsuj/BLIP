from untangling.utils.grasp import GraspSelector, GraspException
from untangling.utils.interface_rws import Interface
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
import numpy as np
from untangling.utils.tcps import *
import time
from untangling.point_picking import *
import matplotlib.pyplot as plt
from untangling.keypoint_untangling import closest_valid_point_to_pt
import torch
import cv2
import threading
from queue import Queue
import signal
import os
import os.path as osp
import datetime
SPEED = (0.6, 6)
iface = Interface(
    "1703005",
    ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
    ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
    speed=SPEED,
)
iface.open_grippers()