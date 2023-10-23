import sys
import numpy as np
import logging
import argparse
import colorsys
import cv2
import matplotlib.pyplot as plt
import time
import os
from collections import OrderedDict
from enum import Enum
# Importing custom modules
sys.path.append('/home/mallika/triton4-lip/lip_tracer/') 

from blip_pipeline.add_noise_to_img import run_tracer_with_transform
from blip_pipeline.divergences import get_divergence_pts
from blip_pipeline.make_endpoint_mapping import get_matching
from learn_from_demos_ltodo import DemoPipeline
from untangling.utils.grasp import GraspSelector
from untangling.point_picking import click_points_simple, click_points_closest
from push_through_backup2 import get_poi_and_vec_for_push, perform_push_through, perform_pindown
from autolab_core import RigidTransform
from untangling.tracer_knot_detect.tracer import TraceEnd
from untangling.utils.tcps import *

MAX_TIME_HORIZON = 5 # maximum number of IP moves you can perform before termination regardless of trace state

def euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_pixel_arc_length(trace):
    return sum(euclidean_dist(trace[i][0], trace[i][1], trace[i+1][0], trace[i+1][1]) for i in range(len(trace) - 1))

def color_for_pct(pct):
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(pct, 1, 1))

def visualize_trace(img, trace, endpoints=None, save=False):
    img_copy = img.copy()
    for i in range(len(trace) - 1):
        pt1, pt2 = get_trace_points(trace, i)
        cv2.line(img_copy, pt1[::-1], pt2[::-1], color_for_pct(i / len(trace)), 4)    
    display_image(img_copy, trace, endpoints, save)

def get_trace_points(trace, idx):
    if not isinstance(trace, OrderedDict):
        return tuple(trace[idx].astype(int)), tuple(trace[idx+1].astype(int))
    trace_keys = list(trace.keys())
    return trace_keys[idx], trace_keys[idx + 1]

def display_image(img, trace, endpoints, save):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if endpoints is not None:
        for ep in endpoints:
            plt.scatter(ep[1], ep[0], c='r', s=4)

    plt.title("Trace Visualized")
    if save:
        plt.savefig("./images/blip"+str(time.time())+".png")
    else:
        plt.show()

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

def choose_endpoint(fullPipeline, step, prev_endpoints, prev_endpoint_idx, img_rgb, choose_ip_point=False, viz=True):
    if choose_ip_point:
            print("Choose IP point")
            pick_img, _ = click_points_simple(img_rgb)
            if pick_img is None:
                refined_pick_pts = []
            else:
                refined_pick_pts = [pick_img]
    else:
        fullPipeline.get_endpoints()
        if step == 0:
            print("Choose endpoint")
            chosen_endpoint, _ = click_points_closest(img_rgb, fullPipeline.endpoints)
            print("Chosen endpoint:", chosen_endpoint)
            for i in range(len(fullPipeline.endpoints)):
                if fullPipeline.endpoints[i][0] == chosen_endpoint[0] and fullPipeline.endpoints[i][1] == chosen_endpoint[1]:
                    prev_endpoint_idx = i
            prev_endpoints = np.array(fullPipeline.endpoints)
        else:
            # use hungarian algorithm to match endpoints
            fullPipeline.get_endpoints()
            old_endpts = np.array(prev_endpoints)
            new_endpts = np.array(fullPipeline.endpoints)
            old_indices, new_indices = get_matching(old_endpts, new_endpts, viz=True, image=img_rgb)
            print("Old indices:", old_indices)
            print("New indices:", new_indices)
            if prev_endpoint_idx not in new_indices:
                print("\n***Endpoint "+prev_endpoint_idx+" left scene***")
                if new_indices.length <= prev_endpoint_idx:
                    extended_indices = np.array(prev_endpoint_idx+1)
                    for i in range(new_indices.length):
                        extended_indices[i] = new_indices[i]
                    extended_indices[prev_endpoint_idx] = prev_endpoint_idx
                    new_indices = extended_indices
            chosen_endpoint = fullPipeline.endpoints[new_indices[prev_endpoint_idx]]
            print("Chosen endpoint:", chosen_endpoint)
            if viz:
                plt.scatter(chosen_endpoint[1], chosen_endpoint[0], s=5, c='r')
                for i, ep in enumerate(new_endpts):
                    plt.text(ep[1]+5, ep[0]+5, new_indices[i])
                plt.imshow(img_rgb)
                plt.show()
            prev_endpoint_idx = new_indices[prev_endpoint_idx]
            prev_endpoints = fullPipeline.endpoints
    return chosen_endpoint, prev_endpoints, prev_endpoint_idx


def main():
    args = parse_args()
    logLevel = args.loglevel
    start_time = time.time()
    fullPipeline = initialize_pipeline(logLevel)

    step, prev_endpoints, prev_endpoint_idx, curr_endpoint_idx = 0, [], [], -1

    img_rgbd, img_rgb = acquire_image(fullPipeline)
    for attempt in range(10):
            try:
                chosen_endpoint, prev_endpoints, prev_endpoint_idx = choose_endpoint(fullPipeline, step, prev_endpoints, prev_endpoint_idx, img_rgb)
            except:
                print("\n***Forgot to choose an endpoint***\n")
            else:
                break

    starting_pixels, _analytic_trace_problem = fullPipeline.get_trace_from_endpoint(chosen_endpoint)

    trace_t, _, _, normalized_covs, normalized_cable_density, y_buffer, x_buffer = run_tracer_with_transform(img_rgb, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=None)
    visualize_trace(img_rgb, trace_t[0], endpoints=fullPipeline.endpoints)
    img_rgbd, img_rgb = acquire_image(fullPipeline)

    # try:
    print(prev_endpoints)
    print(prev_endpoint_idx)
    chosen_endpoint, prev_endpoints, prev_endpoint_idx = choose_endpoint(fullPipeline, 1, prev_endpoints, prev_endpoint_idx, img_rgb)
    # except:
        # print("\n***Issuet***\n")


    starting_pixels, _analytic_trace_problem = fullPipeline.get_trace_from_endpoint(chosen_endpoint)

    trace_t, _, _, normalized_covs, normalized_cable_density, y_buffer, x_buffer = run_tracer_with_transform(img_rgb, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=None)
    visualize_trace(img_rgb, trace_t[0], endpoints=fullPipeline.endpoints)
    img_rgbd, img_rgb = acquire_image(fullPipeline)
    
if __name__ == "__main__":
    main()
