from numpy.lib.type_check import imag
from untangling.utils.interface_rws import Interface
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
import numpy as np
from untangling.utils.tcps import *
from untangling.utils.circle_BFS import trace
from untangling.utils.grasp import GraspSelector, GraspException
from untangling.utils.cable_tracer import CableTracer
from phoxipy.phoxi_sensor import PhoXiSensor
from scipy.interpolate import make_interp_spline
import time
from untangling.point_picking import *
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    T_PHOXI_BASE = RigidTransform.load(
        "/home/justin/yumi/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")
    cam = PhoXiSensor("1703005")
    cam.start()
    img = cam.read()
    intr=PhoXiSensor.create_intr(img.width,img.height)
    pt,_ = click_points(img,T_PHOXI_BASE,intr)
    tracer=CableTracer(img,cam.intrinsics,T_PHOXI_BASE)
    tracer.trace_cable(start_pos=pt,start_direction=np.ones(3),do_vis=True,trace_len=.3,vis_floodfill=True)
