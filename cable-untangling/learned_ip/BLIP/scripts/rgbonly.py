import sys
sys.path.append('/home/mallika/triton4-lip/lip_tracer/') 
sys.path.append('/home/mallika/triton4-lip/cable-untangling/learned_ip/') 
from push_through_backup2 import get_world_coord_from_pixel_coord
from learn_from_demos_ltodo import DemoPipeline
import numpy as np
import argparse
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

def acquire_image(pipeline):
    img_rgbd = pipeline.iface.take_image()
    return img_rgbd, img_rgbd.color._data

args = parse_args()
logLevel = args.loglevel
fullPipeline = initialize_pipeline(logLevel)
img_rgbd, img_rgb = acquire_image(fullPipeline)