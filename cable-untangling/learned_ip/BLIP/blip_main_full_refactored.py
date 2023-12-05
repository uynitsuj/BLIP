import sys
import numpy as np
import logging
import argparse
import colorsys
import cv2
import matplotlib.pyplot as plt
import time
import os
import os.path as osp
import datetime
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
from push_through_backup2 import get_poi_and_vec_for_push, perform_push_through, perform_pindown, perform_pick_away
from autolab_core import RigidTransform
from untangling.tracer_knot_detect.tracer import TraceEnd
from untangling.utils.tcps import *
from calc_arc_length import get_circle, in_circle, view_circles, get_new_start_pt

MAX_TIME_HORIZON = 8 # maximum number of IP moves you can perform before termination regardless of trace state
CABLE_LENGTH = 49

def euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_pixel_arc_length(trace):
    return sum(euclidean_dist(trace[i][0], trace[i][1], trace[i+1][0], trace[i+1][1]) for i in range(len(trace) - 1))

def color_for_pct(pct):
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(pct, 1, 1))

def visualize_trace(fullPipeline, img, trace, endpoints=None, save=True, dirname = None):
    img_copy = img.copy()
    for i in range(len(trace) - 1):
        pt1, pt2 = get_trace_points(trace, i)
        cv2.line(img_copy, pt1[::-1], pt2[::-1], color_for_pct(i / len(trace)), 4)    
    display_image(fullPipeline, img_copy, trace, endpoints, save, dirname)

def get_trace_points(trace, idx):
    if not isinstance(trace, OrderedDict):
        return tuple(trace[idx].astype(int)), tuple(trace[idx+1].astype(int))
    trace_keys = list(trace.keys())
    return trace_keys[idx], trace_keys[idx + 1]

def display_image(fullPipeline, img, trace, endpoints, save, dirname):
    txt_offset = 2
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if endpoints is not None:
        for i, ep in enumerate(endpoints):
            plt.scatter(ep[1], ep[0], c='r', s=4)

    plt.title("Trace Visualized")
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if save:
        fig = plt.gcf()
        plt.show()

        fig.savefig(osp.join(f'./logs/full_rollouts/{dirname}/images/trace{cur_time}.png'))
        fullPipeline.logger.log(25, 'Image saved to ' + osp.join(f'./logs/full_rollouts/{dirname}/images/trace{cur_time}.png'))
        np.save(osp.join(f'./logs/full_rollouts/{dirname}/traces/trace_coords{cur_time}.npy'), trace)
        fullPipeline.logger.log(25, 'Trace saved to ' + osp.join(f'./logs/full_rollouts/{dirname}/traces/trace_coords{cur_time}.png'))
        np.save(osp.join(f'./logs/full_rollouts/{dirname}/traces/endpoints{cur_time}.npy'), endpoints)
        fullPipeline.logger.log(25, 'Endpoints saved to ' + osp.join(f'./logs/full_rollouts/{dirname}/traces/endpoints{cur_time}.png'))
    else:
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print more debug statements",
        action="store_const", dest="loglevel", const=25,
        default=25,
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

def choose_endpoint(fullPipeline, step, prev_endpoints, prev_endpoint_idx, img_rgb, choose_ip_point=False, viz=False):
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
            old_endpts = np.array(prev_endpoints)
            new_endpts = np.array(fullPipeline.endpoints)
            old_indices, new_indices = get_matching(old_endpts, new_endpts, viz=False)

            # if len(list(range(old_endpts.shape[0]))) - len(old_indices) > 0:
            #     missing_idx = list(range(old_endpts.shape[0])) - old_indices
            #     missing_endpt = old_endpts[missing_idx]
                ## PERFORM IP MOVE IN LOCATION OF MISSING ENDPOINT


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

def plot_covs_and_density(covariances, cable_density):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(covariances, marker='x', linestyle='-')
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

def executeINTERNAL(fullPipeline, img_rgb, img_rgbd, trace_t, normalized_covs, normalized_cable_density):
    # plt.plot(normalized_covs, marker='x', linestyle='-')
    # plt.title('Normalized Covariances Across Trace')
    # plt.xlabel('Trace Points')
    # plt.ylabel('Normalized Covariance')
    # plt.show()

    delta_covariances = [normalized_covs[j]-normalized_covs[j-1] for j in range(1, len(normalized_covs))]
    delta_densities = [normalized_cable_density[j]-normalized_cable_density[j-1] for j in range(1, len(normalized_cable_density))]
    delta_covariances.insert(0, 0)
    delta_covariances.append(0)
    delta_covariances.append(0)
    delta_densities.insert(0, 0)
    
    eps = 0.01
    delta_covariances = [0 if dc < eps and dc > 0 else dc for dc in delta_covariances]
    delta_densities = [0 if dd < eps and dd > 0 else dd for dd in delta_densities]
    # plot_covs_and_density(delta_covariances, delta_densities)
    # max_idx, max_val, total_val = None, 0, 0
    num_increases = 0
    uncertainty = []
    for i, (dc, dd) in enumerate(zip(delta_covariances, delta_densities)):
        if dc > 0 and dd > 0:
            num_increases += 1
            curr_sum = 0.15*delta_covariances[i] + 0.20*delta_covariances[i+1]+ 0.15*delta_covariances[i+2] + 0.55/3*delta_densities[i] + 0.55/3*normalized_cable_density[i+1] + 0.55/3*normalized_cable_density[i+2] 
            uncertainty.append(curr_sum)
            # total_val += curr_sum
            # if curr_sum > max_val:
            #     max_val = curr_sum
            #     max_idx = i
        else:
            uncertainty.append(0)

    # plt.plot(uncertainty, marker='x', linestyle='--')
    # plt.title('Uncertainty Across Trace')
    # plt.xlabel('Trace Points')
    # plt.ylabel('Uncertainty')
    # plt.show()

    # max_idx = uncertainty.index(max(uncertainty))
    # push_coord = trace_t[0][max_idx]
    # # plt.imshow(img_rgb)
    # # plt.scatter(push_coord[1], push_coord[0], c='r')
    # # plt.show()
    # poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], viz=True)
    # pin_arm = None
    # pin_arm = perform_pindown(trace_t[0], push_coord, normalized_cable_density, img_rgbd, fullPipeline)
    # perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=False, pin_arm=pin_arm)
    for max_idx in np.argsort([-x for x in uncertainty]):
        push_coord = trace_t[0][max_idx]
        # print('push_coord: ', push_coord)
        poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], viz=True)
        try:
            pin_arm = None
            pin_arm = perform_pindown(trace_t[0], push_coord, normalized_cable_density, img_rgbd, fullPipeline)
            perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=False, pin_arm=pin_arm)
            break
        except:
            try:
                perform_pick_away(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, trace_t)
            except:
                continue

def executeENDPTPERTURB(fullPipeline, img_rgb, img_rgbd, trace_t, normalized_cable_density):
    # push_coord = trace_t[0][-3]
    # print('push_coord: ', push_coord)
    # poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], viz=False)
    # perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=False)
    for i in range(len(trace_t[0])-3):
        # print('i = ', i)
        push_coord = trace_t[0][-3-i]
        # print('push_coord: ', push_coord)
        poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], viz=False)
        try:
            perform_pick_away(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, trace_t)
            break
        except:
            continue

def executeRETRACE(fullPipeline, img_rgb, img_rgbd, trace_t, normalized_cable_density):
    # push_coord = trace_t[0][-2]
    # poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], viz=False)
    # pin_arm = None
    # pin_arm = perform_pindown(trace_t[0], push_coord, normalized_cable_density, img_rgbd, fullPipeline)
    # perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=False, pin_arm=pin_arm)
    for i in range(len(trace_t[0])-2):
        print('i = ', i)
        push_coord = trace_t[0][-2-i]
        print('push_coord: ', push_coord)
        poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], viz=False)
        try:
            pin_arm = None
            pin_arm = perform_pindown(trace_t[0], push_coord, normalized_cable_density, img_rgbd, fullPipeline)
            perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, viz=False, pin_arm=pin_arm)
            break
        except:
            try:
                perform_pick_away(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, trace_t)
            except:
                continue


def executeEDGE(fullPipeline, img_rgb, img_rgbd, trace_t, y_buffer=50, x_buffer=30):
    # push_coord = trace_t[0][-2]
    # poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], edge_case=True, viz=False, 
    #                                                   y_buffer=y_buffer, x_buffer=x_buffer)
    # perform_pick_away(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, trace_t)
    for i in range(len(trace_t[0])-2):
        push_coord = trace_t[0][-2-i]
        poi_trace, cen_poi_vec = get_poi_and_vec_for_push(push_coord, img_rgb, trace_t[0], edge_case=True, viz=False, 
                                                        y_buffer=y_buffer, x_buffer=x_buffer)
        try:
            perform_pick_away(poi_trace, cen_poi_vec, img_rgbd, fullPipeline, trace_t)
            break
        except:
            continue

class IPMove(Enum):
    INTERNAL = 1
    INTERNAL_RETRACE = 2
    EDGE_PERTURB = 3
    ENDPT_PERTURB = 4


def count_circles(trace, click = False, viz = False, img = None):
    circles = []
    start_pt = trace[0]
    next_idx = 1
    while next_idx != trace.shape[0]-1:
        next_pt = trace[next_idx]
        circle, new_next_idx = get_circle(start_pt, next_pt, next_idx, trace)
        new_start_pt = get_new_start_pt(trace, new_next_idx, circle._center, circle.radius)
        start_pt = new_start_pt
        next_idx = new_next_idx
        circles.append(circle)
    
    if not in_circle(trace[-1], circles[-1]):
        radius = circles[-1].radius
        start, next = start_pt, trace[next_idx]
        theta = np.arctan2((next[1] - start[1]), (next[0] - start[0]))
        center_x = start[0] + np.cos(theta) * radius
        center_y = start[1] + np.sin(theta) * radius
        circle = plt.Circle((center_x, center_y), radius, color='r', fill=False)
        circles.append(circle)

    if viz:
        if click:
            return view_circles(trace, circles, click=True, img = img)
        else:
            return view_circles(trace, circles, img = img)
    print(len(circles))
    return len(circles)


def trace_altered(trace_ct):
# add alteration of endpoints?
    threshold = 2  # Placeholder value, adjust as needed
    trace_altered = abs(trace_ct[-1] - trace_ct[-2]) > threshold
    if trace_altered:
        print("\nTrace altered")
    else:
        print("\nTrace did not alter significantly")
    return trace_altered

def core(fullPipeline, img_rgb, img_rgbd, trace_t, ip_history, normalized_covs, normalized_cable_density, starting_pixels, trace_ct, y_buffer, x_buffer, prev_endpoints, prev_endpoint_idx, global_conf):
    trace_state = trace_t[1]
    print(trace_state)
    if len(ip_history) >= MAX_TIME_HORIZON:
        print("MAX_TIME_HORIZON reached. Click circle to determine correctly traced percentage")
        count_circles(trace_t[0], click=True, viz=True)
        return trace_t, normalized_covs, normalized_cable_density, trace_ct
    if global_conf >= CABLE_LENGTH-1 and global_conf <= CABLE_LENGTH+1 and trace_state == TraceEnd.ENDPOINT:
        print("execute endpt perturb")
        fullPipeline.logger.log(25,"Executing Cable Endpoint Verification (CEV-IP)")
        executeENDPTPERTURB(fullPipeline, img_rgb, img_rgbd, trace_t, normalized_cable_density)
        ip_history.append(IPMove.ENDPT_PERTURB)
    elif trace_state == TraceEnd.ENDPOINT:
        print('execute internal')
        fullPipeline.logger.log(25,'Executing Trace Uncertainty Disambiguation (TUG-IP)')
        executeINTERNAL(fullPipeline, img_rgb, img_rgbd, trace_t, normalized_covs, normalized_cable_density)
        ip_history.append(IPMove.INTERNAL)
        img_rgbd, img_rgb = acquire_image(fullPipeline)
        chosen_endpoint, prev_endpoints, prev_endpoint_idx = choose_endpoint(fullPipeline, 1, prev_endpoints, prev_endpoint_idx, img_rgb)
        starting_pixels, _analytic_trace_problem = fullPipeline.get_trace_from_endpoint(chosen_endpoint)
        trace_t, _, _, normalized_covs, normalized_cable_density, y_buffer, x_buffer = run_tracer_with_transform(img_rgb, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=None)
        # trace_ct.append(len(trace_t[0]))

        trace_ct.append(count_circles(trace_t[0]))
        print(trace_t[1])
        visualize_trace(fullPipeline, img_rgb, trace_t[0], fullPipeline.endpoints, dirname=f'logs_{fullPipeline.date_time}')
        # Check for trace alteration or state change
        if trace_altered(trace_ct) or trace_state != TraceEnd.ENDPOINT:
            print("trace altered or trace moved off endpoint")
            fullPipeline.logger.log(25,"Trace altered or trace moved off endpoint")
            return core(fullPipeline, img_rgb, img_rgbd, trace_t, ip_history, normalized_covs, normalized_cable_density, starting_pixels, trace_ct, y_buffer, x_buffer, prev_endpoints, prev_endpoint_idx, global_conf=trace_ct[-1])
        else:
            print("execute endpt perturb")
            fullPipeline.logger.log(25,"Executing Cable Endpoint Verification (CEV-IP)")
            executeENDPTPERTURB(fullPipeline, img_rgb, img_rgbd, trace_t, normalized_cable_density)
            ip_history.append(IPMove.ENDPT_PERTURB)
    elif trace_state == TraceEnd.RETRACE:
        print('execute internal retrace')
        fullPipeline.logger.log(25,"Executing Retrace Uncertainty Disambiguation (RUG-IP)")
        executeRETRACE(fullPipeline, img_rgb, img_rgbd, trace_t, normalized_cable_density)
        ip_history.append(IPMove.INTERNAL_RETRACE)
    elif trace_state == TraceEnd.EDGE:
        print('execute internal edge')
        fullPipeline.logger.log(25,"Executing Edge Recovery (ER-IP)")
        executeEDGE(fullPipeline, img_rgb, img_rgbd, trace_t)
        ip_history.append(IPMove.EDGE_PERTURB)
    elif trace_state == TraceEnd.UNDETERMINED:
        print("entered UNDETERMINED state, performing retrace ip")
        fullPipeline.logger.log(25,"Executing Trace Uncertainty Disambiguation (TUG-IP)")
        executeINTERNAL(fullPipeline, img_rgb, img_rgbd, trace_t, normalized_covs, normalized_cable_density)
        ip_history.append(IPMove.INTERNAL)
    img_rgbd, img_rgb = acquire_image(fullPipeline)
    chosen_endpoint, prev_endpoints, prev_endpoint_idx = choose_endpoint(fullPipeline, 1, prev_endpoints, prev_endpoint_idx, img_rgb)
    starting_pixels, _analytic_trace_problem = fullPipeline.get_trace_from_endpoint(chosen_endpoint)
    trace_t, _, _, normalized_covs, normalized_cable_density, y_buffer, x_buffer = run_tracer_with_transform(img_rgb, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=None)    
    visualize_trace(fullPipeline, img_rgb, trace_t[0], fullPipeline.endpoints, dirname=f'logs_{fullPipeline.date_time}')
    # trace_ct.append(len(trace_t[0]))
    trace_ct.append(count_circles(trace_t[0]))
    return trace_t, normalized_covs, normalized_cable_density, trace_ct

def main():
    args = parse_args()
    logLevel = args.loglevel
    start_time = time.time()
    fullPipeline = initialize_pipeline(logLevel)

    step, prev_endpoints, prev_endpoint_idx, curr_endpoint_idx = 0, [], [], -1
    start_ct, end_ct = 0,-1
    trace_ct = [] #number of points on trace, append per iteration
    ip_history = [] #list of previous IP moves, append per iteration
    DONE = False

    while (not DONE): 
        fullPipeline.logger.log(25,'IP Moves Executed: ' + str(len(ip_history)))
        # try:
        img_rgbd, img_rgb = acquire_image(fullPipeline)
        if step == 0:
            for attempt in range(10):
                try:
                    chosen_endpoint, prev_endpoints, prev_endpoint_idx = choose_endpoint(fullPipeline, step, prev_endpoints, prev_endpoint_idx, img_rgb)
                except:
                    print("\n***Forgot to choose an endpoint***\n")
                else:
                    break
        starting_pixels, _analytic_trace_problem = fullPipeline.get_trace_from_endpoint(chosen_endpoint)
        if len(ip_history) == 0:
            trace_t, _, _, normalized_covs, normalized_cable_density, y_buffer, x_buffer = run_tracer_with_transform(img_rgb, starting_pixels, endpoints=fullPipeline.endpoints, sample=True, start_time=None)
            trace_ct.append(count_circles(trace_t[0]))
            start_ct = count_circles(trace_t[0], click=True, viz=True, img = img_rgb)
            if start_ct == None: start_ct = count_circles(trace_t[0])
            # start_percent_correct=start_ct/CABLE_LENGTH
            print('Start trace length: ' + str(start_ct))
            print(trace_t[1])
            visualize_trace(fullPipeline, img_rgb, trace_t[0], endpoints=fullPipeline.endpoints, dirname=f'logs_{fullPipeline.date_time}')
            trace_t, normalized_covs, normalized_cable_density, trace_ct = core(fullPipeline, img_rgb, img_rgbd, trace_t, ip_history, normalized_covs, 
                                                                                normalized_cable_density, starting_pixels, trace_ct, y_buffer, x_buffer, prev_endpoints, prev_endpoint_idx, global_conf=trace_ct[-1])
        else:
            if len(ip_history) >= MAX_TIME_HORIZON:
                fullPipeline.logger.log(25, 'Max IP Move Horizon Reached')
                end_ct = count_circles(trace_t[0], click=True, viz=True, img = img_rgb)
                fullPipeline.logger.log(25,'Start trace length: ' + str(start_ct))
                fullPipeline.logger.log(25,'Final trace length: ' + str(end_ct))
                DONE = True
                break
            last_move_was_endpt_perturb = ip_history[-1] == IPMove.ENDPT_PERTURB
            if last_move_was_endpt_perturb and trace_t[1] == TraceEnd.ENDPOINT:
                print(TraceEnd.ENDPOINT)
                if trace_altered(trace_ct):
                    trace_t, normalized_covs, normalized_cable_density, trace_ct = core(fullPipeline, img_rgb, img_rgbd, trace_t, ip_history, normalized_covs, 
                                                                                        normalized_cable_density, starting_pixels, trace_ct, y_buffer, x_buffer, prev_endpoints, prev_endpoint_idx, global_conf=trace_ct[-1])
                else:
                    fullPipeline.logger.log(25,'\n******DONE******\n')
                    print("\n******DONE******\n")
                    # visualize_trace(fullPipeline, img_rgb, trace_t[0], endpoints=fullPipeline.endpoints, dirname=f'logs_{fullPipeline.date_time}')
                    end_ct = count_circles(trace_t[0], click=True, viz=True, img = img_rgb)
                    if end_ct == None: end_ct = count_circles(trace_t[0])
                    fullPipeline.logger.log(25,'Start trace length: ' + str(start_ct))
                    fullPipeline.logger.log(25,'Final trace length: ' + str(end_ct))
                    DONE = True
                    break
            else:
                trace_t, normalized_covs,normalized_cable_density, trace_ct = core(fullPipeline, img_rgb, img_rgbd, trace_t, ip_history, normalized_covs, 
                                                                                normalized_cable_density, starting_pixels, trace_ct, y_buffer, x_buffer, prev_endpoints, prev_endpoint_idx, global_conf=trace_ct[-1])
        if trace_t[1]==TraceEnd.UNDETERMINED:
            print("Trace End state was TraceEnd.UNDETERMINED, evaluate cable trace")
        step += 1

if __name__ == "__main__":
    main()
