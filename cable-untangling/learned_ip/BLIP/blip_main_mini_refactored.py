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
# Importing custom modules
sys.path.append('/home/mallika/triton4-lip/lip_tracer/') 

from blip_pipeline.add_noise_to_img import run_tracer_with_transform
from blip_pipeline.divergences import get_divergence_pts
from blip_pipeline.make_endpoint_mapping import get_matching
from learn_from_demos_ltodo import DemoPipeline
from untangling.utils.grasp import GraspSelector
from untangling.point_picking import click_points_simple, click_points_closest
from push_through_backup2 import get_poi_and_vec_for_push, perform_push_through
from autolab_core import RigidTransform
from untangling.tracer_knot_detect.tracer import TraceEnd
from untangling.utils.tcps import *

def euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_pixel_arc_length(trace):
    return sum(euclidean_dist(trace[i][0], trace[i][1], trace[i+1][0], trace[i+1][1]) for i in range(len(trace) - 1))

def color_for_pct(pct):
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(pct, 1, 1))

def visualize_trace(img, trace, save=False):
    img_copy = img.copy()
    for i in range(len(trace) - 1):
        pt1, pt2 = get_trace_points(trace, i)
        cv2.line(img_copy, pt1[::-1], pt2[::-1], color_for_pct(i / len(trace)), 4)
    display_image(img_copy, trace, save)

def get_trace_points(trace, idx):
    if not isinstance(trace, OrderedDict):
        return tuple(trace[idx].astype(int)), tuple(trace[idx+1].astype(int))
    trace_keys = list(trace.keys())
    return trace_keys[idx], trace_keys[idx + 1]

def display_image(img, trace, save):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Trace Visualized")
    if save:
        plt.savefig("multicable_baseline.png")
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

def choose_endpoint(fullPipeline, step, prev_endpoints, prev_endpoint_idx, curr_endpoint_idx, img_rgb, choose_ip_point=False, viz=True):
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
            old_indices, new_indices = get_matching(old_endpts, new_endpts, viz=False)
            print("Old indices:", old_indices)
            print("New indices:", new_indices)
            if prev_endpoint_idx not in new_indices:
                print("Endpoint(s) left scene")
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
                plt.imshow(img_rgb)
                plt.show()
            prev_endpoint_idx = new_indices[prev_endpoint_idx]
            prev_endpoints = fullPipeline.endpoints
    return chosen_endpoint, prev_endpoints, prev_endpoint_idx

def plot_covs_and_density(covariances, cable_density):
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.plot(covariances, marker='o', linestyle='-')
                plt.title('Delta Covariances Across Trace')
                plt.xlabel('Trace Points')
                plt.ylabel('Covariance')

                plt.subplot(1, 2, 2)
                plt.plot(cable_density, marker='x', linestyle='--')
                plt.title('Delta Cable Density Across Trace')
                plt.xlabel('Trace Points')
                plt.ylabel('Cable Density')

                plt.tight_layout()
                plt.show()

def run_tracer(fullPipeline, img_rgb, starting_pixels, viz = True):
    trace_t, heatmaps, crops, normalized_covs, normalized_cable_density = run_tracer_with_transform(img_rgb, 10, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=None)
                # trace_t, noisy_traces = run_tracer_with_transform(img_rgb, 10, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=None)
    print('arc length: ' + str(calculate_pixel_arc_length(trace_t[0])))
    print(trace_t[0][-1])
    print(fullPipeline.endpoints)
    if trace_t[1] == TraceEnd.EDGE:
        print('trace at edge')
    elif trace_t[1] == TraceEnd.ENDPOINT:
        print('trace at endpoint')
    elif trace_t[1] == TraceEnd.FINISHED:
        print('trace is finished')
    elif trace_t[1] == TraceEnd.RETRACE:
        print('trace got retraced')
    if viz:
        visualize_trace(img_rgb, trace_t[0])
    delta_covariances = [normalized_covs[j]-normalized_covs[j-1] for j in range(1, len(normalized_covs))]
    delta_densities = [normalized_cable_density[j]-normalized_cable_density[j-1] for j in range(1, len(normalized_cable_density))]
    delta_covariances.insert(0, 0)
    delta_covariances.append(0)
    delta_covariances.append(0)
    delta_densities.insert(0, 0)
    #plot_covs_and_density(delta_covariances, delta_densities)
    
    eps = 0.01
    delta_covariances = [0 if dc < eps and dc > 0 else dc for dc in delta_covariances]
    delta_densities = [0 if dd < eps and dd > 0 else dd for dd in delta_densities]
    plot_covs_and_density(delta_covariances, delta_densities)
    max_idx, max_val, total_val = None, 0, 0
    num_increases = 0
    uncertainty = []
    #find the argmax of 0.125*delta_covariances[i]+0.125*delta_covariances[i+1] + 0.75*delta_density.
    for i, (dc, dd) in enumerate(zip(delta_covariances, delta_densities)):
        if dc > 0 and dd > 0:
            num_increases += 1
            curr_sum = 0.15*delta_covariances[i] + 0.15*delta_covariances[i+1]+ 0.15*delta_covariances[i+2] + 0.55*dd
            uncertainty.append(curr_sum)
            total_val += curr_sum
            if curr_sum > max_val:
                max_val = curr_sum
                max_idx = i + len(starting_pixels)
        else:
            uncertainty.append(0)
    print(len(uncertainty))
    plt.plot(uncertainty, marker='x', linestyle='--')
    plt.title('Uncertainty Across Trace')
    plt.xlabel('Trace Points')
    plt.ylabel('Uncertainty')
    plt.show()
    
    if num_increases:
        print('num increases = ',  num_increases)
        print('total sum = ', total_val)
        print('average divergence = ', total_val/num_increases)
        print('max idx = ', max_idx)
        print('max val = ', max_val)
        print('delta_cov', delta_covariances[max_idx])
        print('delta_den', delta_densities[max_idx])
        push_coord = trace_t[0][max_idx]
        plt.imshow(img_rgb)
        plt.scatter(push_coord[1], push_coord[0], c='r')
        plt.show()
    return trace_t, push_coord

def main():
    args = parse_args()
    logLevel = args.loglevel
    start_time = time.time()
    fullPipeline = initialize_pipeline(logLevel)

    step, prev_endpoints, prev_endpoint_idx, curr_endpoint_idx = 0, [], [], -1

    DONE = False

    while not DONE: 
        img_rgbd, img_rgb = acquire_image(fullPipeline)
        chosen_endpoint, prev_endpoints, prev_endpoint_idx = choose_endpoint(fullPipeline, step, prev_endpoints, prev_endpoint_idx, curr_endpoint_idx, img_rgb)

        starting_pixels, _analytic_trace_problem = fullPipeline.get_trace_from_endpoint(chosen_endpoint)

        if step == 0:
            trace_t, push_coord = run_tracer(fullPipeline, img_rgb, starting_pixels, start_time)
        else:
            trace_t, push_coord = run_tracer(fullPipeline, img_rgb, starting_pixels)
        poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], type="poles", viz=True)

        print("poi_trace:", poi_trace)
        print("cen_poi_vec:", cen_poi_vec)
        perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=True)
        step += 1

if __name__ == "__main__":
    main()
