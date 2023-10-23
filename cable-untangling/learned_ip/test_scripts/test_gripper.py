from untangling.point_picking import click_points_simple
import cv2
import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
from scripts.full_pipeline_trunk import FullPipeline
import logging

if __name__ == "__main__":
    fullPipeline = FullPipeline(viz=False, loglevel=logging.INFO, topdown_grasp=True, initialize_iface=True)
    fullPipeline.iface.open_grippers()
    fullPipeline.iface.sync()
    fullPipeline.iface.home()
    fullPipeline.iface.sync()
    fullPipeline.iface.close_grippers()
    fullPipeline.iface.sync()