
from blip_pipeline.add_noise_to_img import run_tracer_with_transform
from blip_pipeline.divergences import get_divergence_pts
# from blip_pipeline.refine_push_location import refine_push_location_pts
from blip_pipeline.make_endpoint_mapping import get_matching
from learn_from_demos_ltodo import DemoPipeline
from untangling.utils.grasp import GraspSelector
from untangling.point_picking import click_points_simple, click_points_closest
from push_through_backup2 import get_poi_and_vec_for_push, perform_push_through
from autolab_core import RigidTransform
from untangling.tracer_knot_detect.tracer import TraceEnd
from untangling.utils.tcps import *

import numpy as np
import logging 
import argparse
import colorsys
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import os

def euclidean_dist(x1, y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def calculate_pixel_arc_length(trace):
    arc_length = 0
    for i in range(len(trace) - 1):
        arc_length += euclidean_dist(trace[i][0], trace[i][1], trace[i+1][0], trace[i+1][1])
    return arc_length

def visualize_trace(img, trace, save=False):
    img = img.copy()
    def color_for_pct(pct):
        return colorsys.hsv_to_rgb(pct, 1, 1)[0] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[1] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[2] * 255
    for i in range(len(trace) - 1):
        # if trace is ordered dict, use below logic
        if not isinstance(trace, OrderedDict):
            pt1 = tuple(trace[i].astype(int))
            pt2 = tuple(trace[i+1].astype(int))
        else:
            trace_keys = list(trace.keys())
            pt1 = trace_keys[i]
            pt2 = trace_keys[i + 1]
        cv2.line(img, pt1[::-1], pt2[::-1], color_for_pct(i/len(trace)), 4)
    plt.title("Trace Visualized")
    plt.imshow(img)
    if not save:
        plt.show()
    else:
        plt.savefig("multicable_baseline.png")


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

    start = time.time()

    fullPipeline = DemoPipeline(viz=False, loglevel=logLevel, initialize_iface=True)
    fullPipeline.iface.open_grippers()
    fullPipeline.iface.sync()
    fullPipeline.iface.home()
    fullPipeline.iface.sync()

    choose_ip_point = False
    viz = True

    step = 0
    prev_endpoints = []
    curr_endpoint_idx = -1
    num_divergence_pts = float('inf')
    while True: # replace with stopping condition (global uncertainty measure)
        img_rgbd = fullPipeline.iface.take_image()
        img_rgb = img_rgbd.color._data

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

            starting_pixels, analytic_trace_problem = fullPipeline.get_trace_from_endpoint(chosen_endpoint)

            # TODO: analytic trace problem deal with this IP

            # while analytic_trace_problem:
            #     poi_trace, cen_poi_vec = get_poi_and_vec_for_push(starting_pixels[0], img_rgb, [starting_pixels, starting_pixels + [2, 2]], type="poles", viz=viz)
            #     perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=viz)
            starting_pixels = np.array(starting_pixels)
            if step == 0:
                trace_t, heatmaps, crops, normalized_covs, normalized_cable_density = run_tracer_with_transform(img_rgb, 10, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=start)
                # trace_t, noisy_traces = run_tracer_with_transform(img_rgb, 10, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=start)
                
                baseline_trace = trace_t[0]
                baseline_image = img_rgb
                ## TODO: SAVE BASELINE IMAGE AND PERTURBED IMAGES + ALL TRACES --> FOR REPRODUCING EXPERIMENTS
                # print('trace state is ' + )
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
                visualize_trace(baseline_image, baseline_trace, save=True)

            
            else:
                trace_t, heatmaps, crops, normalized_covs, normalized_cable_density = run_tracer_with_transform(img_rgb, 10, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=None)
                # trace_t, noisy_traces = run_tracer_with_transform(img_rgb, 10, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=None)
            print('arc length: ' + str(calculate_pixel_arc_length(trace_t[0])))
            if viz:
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
                visualize_trace(img_rgb, trace_t[0])

            # check if original trace is on a retrace
            # if trace_t[1] == TraceEnd.RETRACE:
            #     # redo IP move at the trace pt
            #     poi_trace, cen_poi_vec = get_poi_and_vec_for_push(trace_t[0][-1], img_rgb, trace_t[0], type="poles", viz=viz)
            #     perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=viz)
            #     step += 1
            #     continue


            delta_covariances = [normalized_covs[j]-normalized_covs[j-1] for j in range(1, len(normalized_covs))]
            delta_densities = [normalized_cable_density[j]-normalized_cable_density[j-1] for j in range(1, len(normalized_cable_density))]

            eps = 0.01
            delta_covariances = [0 if dc < eps and dc > 0 else dc for dc in delta_covariances]
            delta_densities = [0 if dd < eps and dd > 0 else dd for dd in delta_densities]

            max_idx, max_val, total_val = None, 0, 0
            num_increases = 0
            for i, (dc, dd) in enumerate(zip(delta_covariances, delta_densities)):
                if dc > 0 and dd > 0:
                    num_increases += 1
                    curr_sum = dc + dd
                    total_val += curr_sum
                    if curr_sum > max_val:
                        max_val = curr_sum
                        max_idx = i
            
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

            # print('trace state is ' + )
            # print(trace_t[0][-1])
            # print(fullPipeline.endpoints)
            # if trace_t[1] == TraceEnd.EDGE:
            #     print('trace at edge')
            # elif trace_t[1] == TraceEnd.ENDPOINT:
            #     print('trace at endpoint')
            # elif trace_t[1] == TraceEnd.FINISHED:
            #     print('trace is finished')
            # elif trace_t[1] == TraceEnd.RETRACE:
            #     print('trace got retraced')

            # if trace_t[1] == TraceEnd.FINISHED:
            #     print('Finshed trace!')
            #     break

            poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], type="poles", viz=viz)

            print("poi_trace:", poi_trace)
            print("cen_poi_vec:", cen_poi_vec)
            perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=viz)

        
            '''if viz:
                for noisy_trace in noisy_traces:
                    visualize_trace(img_rgb, noisy_trace[0])
          
            divergence_pts, num_divergence_pts = get_divergence_pts(trace_t[0], [nt[0] for nt in noisy_traces], prune=False)
            refined_push_pts = refine_push_location_pts(img_rgb[:, :, 0], trace_t[0], divergence_pts, viz=False)

            print("num_divergence_pts:", num_divergence_pts)
            # if num_divergence_pts <= 5: # trace is same as true trace >= (10 - 5) = 5 times
            if num_divergence_pts <= 3: # trace is same as true trace >= (10 - 3) = 7 times
                blip_trace = trace_t[0]
                blip_image = img_rgb
                blip_time = time.time()
                break

            divergence_pts_trace = [[pt[1], pt[0]] for pt in divergence_pts]
            refined_push_pts_trace = [[pt[1], pt[0]] for pt in refined_push_pts]'''



            # push_coord = refined_push_pts_trace[0]
            # poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], type="poles", viz=viz)
            # perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=viz)
        
        step += 1

print("Done!")
# print("baseline time", baseline_time - start)
print("blip time", blip_time - start)

visualize_trace(blip_image, blip_trace, save=False)
