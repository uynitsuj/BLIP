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
from cable_tracing.tracers.simple_uncertain_trace import trace
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
from untangling.tracer_knot_detect.tracer_knot_detection import TracerKnotDetector
from untangling.tracer_knot_detect.tracer import TraceEnd
from scripts.full_pipeline_trunk import FullPipeline

class DemoPipeline(FullPipeline):
    def get_closest_trace_point(self, pix, trace):
        distances = np.linalg.norm(np.array(trace) - np.array(pix)[None, ...], axis=1)
        return np.argmin(distances)
    
    def get_trace_dist(self, pix, trace, cumsum):
        return cumsum[self.get_closest_trace_point(pix, trace)]

    def get_trace_idx_for_dist(self, dist, cumsum):
        return np.argmax(cumsum > dist)
    
    def imitate_point(self, pixel_demo, trace_pixels_demo, trace_crossings_demo, trace_pixels_exec,
                      trace_crossings_exec, imit_type='distance', allow_off_cable=False):
        supported_types = ['distance', 'crossing']
        assert imit_type in supported_types, f"Type {imit_type} not supported, must be one of {supported_types}"

        if imit_type == 'distance':
            dist_cumsum_demo = self.tkd.tracer.get_dist_cumsum_array(trace_pixels_demo)
            dist_cumsum_exec = self.tkd.tracer.get_dist_cumsum_array(trace_pixels_exec)
            # find the closest point to the demo point on the trace
            grasp_dist = self.get_trace_dist(pixel_demo, trace_pixels_demo, dist_cumsum_demo)
            # now find the closest point to the demo point on the trace for the execution
            grasp_dist_idx = self.get_trace_idx_for_dist(grasp_dist, dist_cumsum_exec)
            # now get coordinates
            return trace_pixels_exec[grasp_dist_idx]
        elif imit_type == 'crossing':
            demo_trace_index = self.get_closest_trace_idx(pixel_demo, trace_pixels_demo)
            crossing_pixel_indices = [crossing['pixels_idx'] for crossing in trace_crossings_demo]
            crossing_pixel_indices_exec = [crossing['pixels_idx'] for crossing in trace_crossings_exec]

            # find the pixel index of the crossing just before and 
            crossing_after_idx = np.argmax(crossing_pixel_indices > demo_trace_index)
            crossing_before_idx = crossing_after_idx - 1
            crossing_after_trace_idx = crossing_pixel_indices[crossing_after_idx]
            crossing_before_trace_idx = crossing_pixel_indices[crossing_before_idx]
            demo_point_fraction = (demo_trace_index - crossing_before_trace_idx) / (crossing_after_trace_idx - crossing_before_trace_idx)

            exec_crossing_before_trace_idx = crossing_pixel_indices_exec[crossing_before_idx]
            exec_crossing_after_trace_idx = crossing_pixel_indices_exec[crossing_after_idx]
            exec_index = int(exec_crossing_before_trace_idx + demo_point_fraction * (exec_crossing_after_trace_idx - exec_crossing_before_trace_idx))
            anchor_point = trace_pixels_exec[exec_index]

        projection, perpendicular = 0, 0
        perpendicular_unit_vec, direction_vec_exec = np.array([0, 0]), np.array([0, 0])
        if allow_off_cable:  # for place points
            import pdb; pdb.set_trace()
            # find local orientation of the trace at the anchor point
            anchor_point_neighbors = trace_pixels_demo[demo_anchor_idx - 1:demo_anchor_idx + 2]
            # assume nothing leaves the image or is out of bounds
            direction_vec_demo = anchor_point_neighbors[-1] - anchor_point_neighbors[0]
            direction_vec_demo = direction_vec_demo / np.linalg.norm(direction_vec_demo)
            
            displacement_vector = pixel_demo_orig - pixel_demo
            # calculate projection along direction vec and perpendicular to direction vec
            projection = np.linalg.norm(np.dot(displacement_vector, direction_vec_demo))
            perpendicular = np.linalg.norm(displacement_vector - projection * direction_vec_demo)
            perpendicular_unit_vec = (displacement_vector - projection * direction_vec_demo) / perpendicular
            
            direction_vec_exec = trace_pixels_exec[exec_index - 1] - trace_pixels_exec[exec_index + 2]
            direction_vec_exec = direction_vec_exec / np.linalg.norm(direction_vec_exec)
        return anchor_point + perpendicular * perpendicular_unit_vec + projection * direction_vec_exec

    def get_trace_pixels(self):
        print("started perception")
        traces = []
        crossings = []
        trace_uncertain = True
        uncertain_endpoint = None
        for point in self.endpoints:
            starting_pixels, trace_problem = self.get_trace_from_endpoint(point)
            starting_pixels = np.array(starting_pixels)
            if not trace_problem:
                trace_uncertain = False
                self.tkd._set_data(self.img.color._data, starting_pixels)
                perception_result = self.tkd.perception_pipeline(endpoints=self.endpoints, viz=True, do_perturbations=True)
                traces.append(self.tkd.pixels)
                crossings.append(self.tkd.detector.crossings)
            else:
                uncertain_endpoint = starting_pixels
        return traces[0], crossings[0] #, trace_uncertain, uncertain_endpoint

    def run_pipeline(self):
        start_time = int(time.time())
        self.logs_file.write(f'Start time: {start_time}\n')
        count = 0
        diff = int(time.time()) - start_time
        while not done or diff < self.time_limit:
            try:
                time.sleep(1)
                # collect demos
                print("COLLECTING DEMOS")

                demos = []
                while True:
                    input_chars = input("Press enter when ready to take an image, or press x once all demos are collected")
                    if input_chars == 'x':
                        break

                    self.img = self.take_and_save_image(path='/home/justin/yumi/cable-untangling/scripts/oct_11_more_overhand/color_6.npy'); self.get_endpoints()
                    trace_pixels, trace_crossings = self.get_trace_pixels()

                    # gather click points from user
                    left_coords, right_coords = click_points_simple(self.img)
                    left_coords, right_coords = left_coords[::-1], right_coords[::-1]
                    
                    # gather place points from user
                    left_place_coords, right_place_coords = click_points_simple(self.img)
                    left_place_coords, right_place_coords = left_place_coords[::-1], right_place_coords[::-1]

                    demos.append({'img': self.img, 'left_coords': left_coords, 'right_coords': right_coords, 'trace_pixels': trace_pixels,
                    'trace_crossings': trace_crossings, 'left_place_coords': left_place_coords, 'right_place_coords': right_place_coords})

                # now process demos for execution on robot
                input("Press enter when ready to execute policy learned from demos")
                self.img = self.take_and_save_image(path='/home/justin/yumi/cable-untangling/scripts/oct_11_more_overhand/color_7.npy'); self.get_endpoints()
                if self.endpoints.shape[0] == 0:
                    failure = True
                trace_pixels_exec, trace_crossings_exec = self.get_trace_pixels()

                demo_type = "length_relative"
                if demo_type == "length_relative":
                    # get the normalized distance along the trace that the demo points were
                    # for now just use the first demo
                    demo = demos[0]
                    trace_pixels = demo['trace_pixels']
                    dist_cumsum = self.tkd.tracer.get_dist_cumsum_array(trace_pixels)
                    print(dist_cumsum)
                    print("Dist cumsum max", dist_cumsum[-1])
                    dist_cumsum_exec = self.tkd.tracer.get_dist_cumsum_array(trace_pixels_exec)
                    print(dist_cumsum)
                    print("Dist cumsum exec max", dist_cumsum_exec[-1])

                    # find the closest point to the demo point on the trace
                    left_grasp_dist = self.get_trace_dist(demo['left_coords'], demo['trace_pixels'], dist_cumsum)
                    right_grasp_dist = self.get_trace_dist(demo['right_coords'], demo['trace_pixels'], dist_cumsum)

                    print("Grasp distances", left_grasp_dist, right_grasp_dist)

                    # now find the closest point to the demo point on the trace for the execution
                    left_grasp_dist_idx = self.get_trace_idx_for_dist(left_grasp_dist, dist_cumsum_exec)
                    right_grasp_dist_idx = self.get_trace_idx_for_dist(right_grasp_dist, dist_cumsum_exec)
                    print(left_grasp_dist)

                    # now get coordinates
                    grasp1_coord = trace_pixels_exec[left_grasp_dist_idx]
                    grasp2_coord = trace_pixels_exec[right_grasp_dist_idx]

                    # display the grasp points
                    plt.scatter(grasp1_coord[1], grasp1_coord[0], c='r')
                    plt.scatter(grasp2_coord[1], grasp2_coord[0], c='b')
                    plt.imshow(self.img.color._data)
                    plt.show()
                    
                elif demo_type == "crossing_relative":
                    # find index of the crossing adjacent to the demo points
                    demo = demos[0]
                    trace_pixels = demo['trace_pixels']
                    demo_trace_indices = (self.get_closest_trace_point(demo['left_coords'], trace_pixels), self.get_closest_trace_point(demo['right_coords'], trace_pixels))                    
                    crossing_pixel_indices = np.array([crossing['pixels_idx'] for crossing in demo['trace_crossings']])
                    crossing_pixel_indices_exec = np.array([crossing['pixels_idx'] for crossing in trace_crossings_exec])
                    
                    # find the GRASP points
                    
                    # find the pixel index of the crossing just before each demo point
                    demo_point_infos = []
                    for demo_point in demo_trace_indices:
                        crossing_before_idx = np.argmax(crossing_pixel_indices > demo_point)
                        crossing_before_pixel_idx = crossing_pixel_indices[crossing_before_idx]
                        crossing_after_pixel_idx = crossing_pixel_indices[crossing_before_idx + 1]
                        demo_point_fraction = (demo_point - crossing_before_pixel_idx) / (crossing_after_pixel_idx - crossing_before_pixel_idx)
                        # each demo should be a dict {'crossing_num': crossing_idx, 'normalized_dist_between_crossings': normalized_dist}
                        demo_point_infos.append({'crossing_num': crossing_before_idx, 'normalized_dist_between_crossings': demo_point_fraction})

                    grasp_points = []
                    # now find those points on the execution trace
                    for demo_point_info in demo_point_infos:
                        before_crossing_pixel_idx = crossing_pixel_indices_exec[demo_point_info['crossing_num']]
                        after_crossing_pixel_idx = crossing_pixel_indices_exec[demo_point_info['crossing_num'] + 1]
                        demo_point_info['pixel_idx'] = int(before_crossing_pixel_idx + demo_point_info['normalized_dist_between_crossings'] * (after_crossing_pixel_idx - before_crossing_pixel_idx))
                        grasp_points.append(trace_pixels_exec[demo_point_info['pixel_idx']])
                    grasp1_coord, grasp2_coord = grasp_points
                    
                    # find the place points by defining a local coordinate system at the closest point on the trace to the place points on the demo place points
                    demo_place_locations = [demo['left_place_coords'], demo['right_place_coords']]
                    demo_place_location_anchors = [self.get_closest_trace_point(demo_place_location, trace_pixels) for demo_place_location in demo_place_locations]
                    
                    place_points_exec = []
                    for demo_place_location, demo_place_location_anchor in zip(demo_place_locations, demo_place_location_anchors):
                        # find local orientation of the trace at the anchor point
                        anchor_point = trace_pixels[demo_place_location_anchor]
                        anchor_point_neighbors = trace_pixels[demo_place_location_anchor - 1:demo_place_location_anchor + 2]
                        # assume nothing leaves the image or is out of bounds
                        direction_vec = anchor_point_neighbors[-1] - anchor_point_neighbors[0]
                        direction_vec = direction_vec / np.linalg.norm(direction_vec)


                # execute the grasps
                self.g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
                l_grasp, r_grasp = self.g.double_grasp(
                    tuple(grasp1_coord[::-1]), tuple(grasp2_coord[::-1]), .0085, .0085, self.iface.L_TCP, self.iface.R_TCP, slide0=True, slide1=False)
                
                self.iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp)
                self.iface.sync()

                diff = int(time.time()) - start_time
            except Exception as e:
                # check if message says "nonresponsive"
                if "nonresponsive after sync" in str(e) or \
                    "Couldn't plan path" in str(e) or \
                    "No grasp pair" in str(e) or \
                    "Timeout" in str(e) or \
                    "No collision free grasps" in str(e) or \
                    "Not enough points traced" in str(e) or \
                    "Jump in first 4 joints" in str(e) or \
                    "Planning to pose failed" in str(e):
                    self.logger.info(f"Caught error, recovering {str(e)}")
                    failure = True
                    self.iface.y.reset()
                else:
                    raise e
                    self.logger.info("Uncaught exception, still recovering " + str(e))
                    self.iface.y.reset()
                self.iface.sync()
    
    def imitate_point_backup_full(self, pixel_demo, trace_pixels_demo, trace_crossings_demo, trace_pixels_exec,
                      trace_crossings_exec, imit_type='distance', allow_off_cable=False):
        supported_types = ['distance', 'crossing']
        assert imit_type in supported_types, f"Type {imit_type} not supported, must be one of {supported_types}"

        pixel_demo_orig = pixel_demo
        demo_anchor_idx = self.get_closest_trace_idx(pixel_demo, trace_pixels_demo)
        pixel_demo = trace_pixels_demo[demo_anchor_idx]

        anchor_point = None
        if imit_type == 'distance':
            dist_cumsum_demo = self.tkd.tracer.get_dist_cumsum_array(trace_pixels_demo)
            dist_cumsum_exec = self.tkd.tracer.get_dist_cumsum_array(trace_pixels_exec)
            # find the closest point to the demo point on the trace
            grasp_dist = self.get_trace_dist(pixel_demo, trace_pixels_demo, dist_cumsum_demo)
            # now find the closest point to the demo point on the trace for the execution
            exec_index = self.get_trace_idx_for_dist(grasp_dist, dist_cumsum_exec)
            # now get coordinates
            anchor_point = trace_pixels_exec[exec_index]
        elif imit_type == 'crossing':
            import pdb; pdb.set_trace()
            demo_trace_index = self.get_closest_trace_idx(pixel_demo, trace_pixels_demo)
            crossing_pixel_indices = [crossing['pixels_idx'] for crossing in trace_crossings_demo]
            crossing_pixel_indices_exec = [crossing['pixels_idx'] for crossing in trace_crossings_exec]

            # find the pixel index of the crossing just before each demo point
            crossing_before_idx = np.argmax(crossing_pixel_indices > demo_trace_index)
            crossing_before_pixel_idx = crossing_pixel_indices[crossing_before_idx]
            crossing_after_pixel_idx = crossing_pixel_indices[crossing_before_idx + 1]
            demo_point_fraction = (demo_trace_index - crossing_before_pixel_idx) / (crossing_after_pixel_idx - crossing_before_pixel_idx)

            before_crossing_pixel_idx = crossing_pixel_indices_exec[crossing_before_pixel_idx]
            after_crossing_pixel_idx = crossing_pixel_indices_exec[demo_point_fraction + 1]
            exec_index = int(before_crossing_pixel_idx + demo_point_fraction * (after_crossing_pixel_idx - before_crossing_pixel_idx))
            anchor_point = trace_pixels_exec[exec_index]

        projection, perpendicular = 0, 0
        perpendicular_unit_vec, direction_vec_exec = np.array([0, 0]), np.array([0, 0])
        if allow_off_cable:  # for place points
            # find local orientation of the trace at the anchor point
            anchor_point_neighbors = trace_pixels_demo[demo_anchor_idx - 1:demo_anchor_idx + 2]
            # assume nothing leaves the image or is out of bounds
            direction_vec_demo = anchor_point_neighbors[-1] - anchor_point_neighbors[0]
            direction_vec_demo = direction_vec_demo / np.linalg.norm(direction_vec_demo)
            
            displacement_vector = pixel_demo_orig - pixel_demo
            # calculate projection along direction vec and perpendicular to direction vec
            projection = np.linalg.norm(np.dot(displacement_vector, direction_vec_demo))
            perpendicular = np.linalg.norm(displacement_vector - projection * direction_vec_demo)
            perpendicular_unit_vec = (displacement_vector - projection * direction_vec_demo) / perpendicular
            
            direction_vec_exec = trace_pixels_exec[exec_index - 1] - trace_pixels_exec[exec_index + 2]
            direction_vec_exec = direction_vec_exec / np.linalg.norm(direction_vec_exec)
        return anchor_point + perpendicular * perpendicular_unit_vec + projection * direction_vec_exec


                
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
    fullPipeline = DemoPipeline(viz=True, loglevel=logLevel, initialize_iface=False)
    fullPipeline.run_pipeline()
