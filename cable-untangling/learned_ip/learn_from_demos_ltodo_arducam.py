# 3 lines of detectron imports
from codecs import IncrementalDecoder
from pickle import FALSE
from turtle import done, left
# import analysis as loop_detectron
from untangling.utils.grasp import GraspSelector, GraspException
from untangling.utils.interface_rws_arducam import Interface
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
import numpy as np
from untangling.utils.tcps import *
from untangling.utils.cable_tracing.cable_tracing.tracers.simple_uncertain_trace import trace
import time
from untangling.point_picking import *
import matplotlib.pyplot as plt
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
from scripts.full_pipeline_trunk_arducam import FullPipeline

class DemoPipeline(FullPipeline):
    def __init__(self, viz, loglevel, initialize_iface):
        FullPipeline.__init__(self, viz, loglevel, initialize_iface)
        self.demos = []

    def get_closest_trace_idx(self, pix, trace):
        distances = np.linalg.norm(np.array(trace) - np.array(pix)[None, ...], axis=1)
        return np.argmin(distances)
    
    def get_closest_trace_point(self, pix, trace):
        return trace[self.get_closest_trace_idx(pix, trace)]
    
    def get_trace_dist(self, pix, trace, cumsum):
        return cumsum[self.get_closest_trace_idx(pix, trace)]

    def get_trace_idx_for_dist(self, dist, cumsum):
        return np.argmax(cumsum > dist)
    
    '''checks if point to check is to the right of the vector, starting at the start point'''
    def check_right(self, vec, vec_start_point, point_to_check):
        check_vec = point_to_check - vec_start_point
        return np.cross(vec, check_vec) < 0

    
    def imitate_point(self, pixel_demo, trace_pixels_demo, trace_crossings_demo, trace_pixels_exec,
                      trace_crossings_exec, demo_img, imit_type='distance', allow_off_cable=False):
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
            demo_trace_index = self.get_closest_trace_idx(pixel_demo, trace_pixels_demo)
            crossing_pixel_indices = [crossing['pixels_idx'] for crossing in trace_crossings_demo]
            crossing_pixel_indices_exec = [crossing['pixels_idx'] for crossing in trace_crossings_exec]

            # find the pixel index of the crossing just before and 

            #if demo trace index is before first crossing
            if demo_trace_index < crossing_pixel_indices[0]:
                crossing_before_trace_idx = 0
                crossing_after_trace_idx = crossing_pixel_indices[0]
                exec_crossing_before_trace_idx = 0
                exec_crossing_after_trace_idx = crossing_pixel_indices_exec[0]
            
            #if demo trace index is after last crossing
            elif demo_trace_index > crossing_pixel_indices[-1]:
                crossing_before_idx = len(crossing_pixel_indices) - 1   
                crossing_before_trace_idx = crossing_pixel_indices[crossing_before_idx]
                crossing_after_trace_idx = len(trace_pixels_demo)

                exec_crossing_before_trace_idx = crossing_pixel_indices_exec[crossing_before_idx]
                #in case there are retraces and thus extra crossings and a "longer" total trace in test
                if len(crossing_pixel_indices_exec) > len(crossing_pixel_indices):
                    exec_crossing_after_trace_idx = crossing_pixel_indices_exec[crossing_before_idx + 1]           
                else:
                    exec_crossing_after_trace_idx = len(trace_pixels_exec)
            
            #demo trace index is within crossings
            else:
                #argmax finds first occurrence of True here
                crossing_after_idx = np.argmax(crossing_pixel_indices > demo_trace_index)
                crossing_before_idx = crossing_after_idx - 1
                crossing_after_trace_idx = crossing_pixel_indices[crossing_after_idx]
                crossing_before_trace_idx = crossing_pixel_indices[crossing_before_idx]
                exec_crossing_after_trace_idx = crossing_pixel_indices_exec[crossing_after_idx]
                exec_crossing_before_trace_idx = crossing_pixel_indices_exec[crossing_before_idx]

            demo_point_fraction = (demo_trace_index - crossing_before_trace_idx) / (crossing_after_trace_idx - crossing_before_trace_idx)

            exec_index = int(exec_crossing_before_trace_idx + demo_point_fraction * (exec_crossing_after_trace_idx - exec_crossing_before_trace_idx))
            anchor_point = trace_pixels_exec[exec_index]

        projection, perpendicular = 0, 0
        perpendicular_unit_vec_exec, direction_unit_vec_exec = np.array([0, 0]), np.array([0, 0])
        if allow_off_cable:  # for place points

            # find local orientation of the trace at the anchor point
            
            #find the direction vector of the trace at the anchor point
            if demo_anchor_idx == 0:
                direction_vec_demo = trace_pixels_demo[demo_anchor_idx + 2] - trace_pixels_demo[demo_anchor_idx]
                direction_check_vec_demo = direction_vec_demo
                vec_start_point_demo = trace_pixels_demo[demo_anchor_idx]
                check_point_demo = trace_pixels_demo[demo_anchor_idx + 1]
            else:
                direction_vec_demo = trace_pixels_demo[demo_anchor_idx + 1] - trace_pixels_demo[demo_anchor_idx - 1]
                direction_check_vec_demo = direction_vec_demo
                vec_start_point_demo = trace_pixels_demo[demo_anchor_idx - 1]
                check_point_demo = trace_pixels_demo[demo_anchor_idx]
                
                offset = 10
                if demo_anchor_idx > offset and len(trace_pixels_demo) > demo_anchor_idx + offset:
                    direction_check_vec_demo = trace_pixels_demo[demo_anchor_idx + offset] - trace_pixels_demo[demo_anchor_idx - offset]
                    check_point_demo = np.average(trace_pixels_demo [demo_anchor_idx - (offset-1): demo_anchor_idx + (offset-1)], axis=0)
                    vec_start_point_demo = trace_pixels_demo[demo_anchor_idx - offset]
                    plt.scatter(vec_start_point_demo[1], vec_start_point_demo[0], c = "blue", s = 10)
                    ep = vec_start_point_demo + direction_check_vec_demo
                    plt.scatter(ep[1], ep[0], c = "pink", s = 10)
                    plt.scatter(check_point_demo[1], check_point_demo[0], color="red", s = 10)

                    plt.imshow(demo_img.color._data)

                    if self.save:
                        plt.savefig(self.output_vis_dir + "demo_cable_orient_{}".format(self.demo_counter))
                    
                    if self.viz:
                        plt.show()
            
            direction_unit_vec_demo = direction_vec_demo / np.linalg.norm(direction_vec_demo)
            
            displacement_vector = pixel_demo_orig - pixel_demo
            # calculate projection along direction vec and perpendicular to direction vec
            projection = np.linalg.norm(np.dot(displacement_vector, direction_unit_vec_demo))
            perpendicular = np.linalg.norm(displacement_vector - projection * direction_unit_vec_demo)
            perpendicular_unit_vec = (displacement_vector - projection * direction_unit_vec_demo) / perpendicular
            
            #calculate rotation matrix between direction unit vec and perpendicular unit vec
            xa = direction_unit_vec_demo[0]
            ya = direction_unit_vec_demo[1]
            xb = perpendicular_unit_vec[0]
            yb = perpendicular_unit_vec[1]
            rotation_matrix = np.array([[xa*xb + ya*yb, xb*ya - xa*yb], [xa*yb - xb*ya, ya*yb + xa*xb]])
            if exec_index == 0:
                direction_vec_exec = trace_pixels_exec[exec_index + 2] - trace_pixels_exec[exec_index]
                direction_check_vec_exec = direction_vec_exec
                vec_start_point = trace_pixels_exec[exec_index]
                check_point = trace_pixels_exec[exec_index + 2]
            else:
                direction_vec_exec = trace_pixels_exec[exec_index + 1] - trace_pixels_exec[exec_index - 1]
                direction_check_vec_exec = direction_vec_exec
                vec_start_point = trace_pixels_exec[exec_index - 1]
                check_point = trace_pixels_exec[exec_index]

                if exec_index > offset and len(trace_pixels_exec) > exec_index + offset:
                    vec_start_point = trace_pixels_exec[exec_index - offset]
                    direction_check_vec_exec = trace_pixels_exec[exec_index + offset] - trace_pixels_exec[exec_index - offset]
                    check_point = np.average(trace_pixels_exec[exec_index - (offset-1): exec_index + (offset-1)], axis=0)
                    plt.scatter(vec_start_point[1], vec_start_point[0], c = "blue", s=10)
                    ep = vec_start_point + direction_check_vec_exec
                    plt.scatter(ep[1], ep[0], c = "pink", s=10 )
                    plt.scatter(check_point[1], check_point[0], color="red", s=10)
                    plt.imshow(self.img.color._data)

                    if self.save:
                        plt.savefig(self.output_vis_dir + "exec_cable_orient_{}".format(self.demo_counter))
                    
                    if self.viz:
                        plt.show()
            
            direction_unit_vec_exec = direction_vec_exec / np.linalg.norm(direction_vec_exec)
            perpendicular_unit_vec_exec = np.dot(rotation_matrix, direction_unit_vec_exec)
        
            # checks if the test and exec direction vectors are on same side of trace
            # if self.check_right(direction_check_vec_exec, vec_start_point, check_point) != self.check_right(direction_check_vec_demo, vec_start_point_demo, check_point_demo):
            #     print("REVERSED!")
            #     perpendicular_unit_vec_exec = -1 * perpendicular_unit_vec_exec
        return anchor_point + perpendicular * perpendicular_unit_vec_exec + projection * direction_unit_vec_exec

    
    # for now assumes just one endpoint
    def get_trace_pixels(self, img, endpoint):
        traces = []
        crossings = []

        if len(self.endpoints) != 2:
            import pdb; pdb.set_trace()
        assert len(self.endpoints) == 2, "Must have detected two endpoints"

        # categorize left and right endpoints
        if(self.endpoints[0][1] < self.endpoints[1][1]):
            left_endpoint = self.endpoints[0]
            right_endpoint = self.endpoints[1]
        else:
            left_endpoint = self.endpoints[1]
            right_endpoint = self.endpoints[0]

        # choose the correct endpoint
        if endpoint == "left":
            endpoint = left_endpoint
        else:
            endpoint = right_endpoint

        starting_pixels, analytic_trace_problem = self.get_trace_from_endpoint(endpoint)
        starting_pixels = np.array(starting_pixels)
        self.tkd._set_data(self.img.color._data, starting_pixels)
        perception_result = self.tkd.perception_pipeline(endpoints=self.endpoints, viz=False, vis_dir=self.output_vis_dir) #do_perturbations=True
        traces.append(self.tkd.pixels)
        crossings.append(self.tkd.detector.crossings)

        if len(traces) > 0:
            return traces[0], crossings[0] 
        

    # not using this currently - add perturb in later
    def get_trace_pixels_with_perturb(self, img, endpoint="both"):
        print("started perception")
        traces = []
        crossings = []
        trace_uncertain = True
        uncertain_endpoint = None

        if len(self.endpoints) == 1:
            endpoints_to_trace_from = self.endpoints
        else:
            if endpoint == "both":
                endpoints_to_trace_from = self.endpoints
            else:
                #checking if x value is less 
                if(self.endpoints[0][1] < self.endpoints[1][1]):
                    left_endpoint = self.endpoints[0]
                    right_endpoint = self.endpoints[1]
                else:
                    left_endpoint = self.endpoints[1]
                    right_endpoint = self.endpoints[0]
                print("the left endpoint is ", left_endpoint)
                print("the right endpoint is ", right_endpoint)

                if endpoint == "left":
                    endpoints_to_trace_from = [left_endpoint]
                else:
                    endpoints_to_trace_from = [right_endpoint]

        #for now assuming endpoint is either left or right, not both
        for point in endpoints_to_trace_from:
            starting_pixels, analytic_trace_problem = self.get_trace_from_endpoint(point)
            starting_pixels = np.array(starting_pixels)
            while analytic_trace_problem:
                uncertain_endpoint = starting_pixels
                print(starting_pixels)
                self.g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
                self.iface.perturb_point_knot_tye(self.img, uncertain_endpoint[::-1], self.g)

                self.img = self.iface.take_image()
                self.get_endpoints()
                
                if len(self.endpoints) == 1:
                    point = self.endpoints[0]

                else:
                    if(self.endpoints[0][1] < self.endpoints[1][1]):
                        left_endpoint = self.endpoints[0]
                        right_endpoint = self.endpoints[1]
                    else:
                        left_endpoint = self.endpoints[1]
                        right_endpoint = self.endpoints[0]
                    if endpoint == "left":
                        point = left_endpoint
                    elif endpoint == "right":
                        point = right_endpoint

                starting_pixels, analytic_trace_problem = self.get_trace_from_endpoint(point)
                starting_pixels = np.array(starting_pixels)
                    

            #once there is no analytical trace problem
            self.tkd._set_data(self.img.color._data, starting_pixels)
            perception_result = self.tkd.perception_pipeline(endpoints=self.endpoints, viz=True, vis_dir=self.output_vis_dir) #do_perturbations=True
            traces.append(self.tkd.pixels)
            crossings.append(self.tkd.detector.crossings)
            
        if len(traces) > 0:
            return traces[0], crossings[0] 
    

    def run_pipeline_place_only(self):
        self.img = self.iface.take_image()
        print("Choose place points")
        place_1, _= click_points_simple(self.img)
        print(place_1)

        plt.scatter(place_1[0], place_1[1], c='r')
        plt.imshow(self.img.color._data)
        plt.show()


        g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
        place1_point = g.ij_to_point(place_1).data
        intermediate_point = place1_point + np.array([0, 0, 0.1])
        self.iface.home()
        self.iface.sync()

        print("place1", place_1)
        print("place1_point", place1_point)
        intermediate_transform = RigidTransform(
                translation=intermediate_point,
                rotation= self.iface.GRIP_DOWN_R,
                from_frame=YK.l_tcp_frame,
                to_frame="base_link",
        )
        place1_transform = RigidTransform(
                translation=place1_point,
                rotation= self.iface.GRIP_DOWN_R,
                from_frame=YK.l_tcp_frame,
                to_frame="base_link",
        )

        self.iface.go_cartesian(
        l_targets=[intermediate_transform], removejumps=[6]
        )
        self.iface.sync()

        self.iface.go_cartesian(
        l_targets=[place1_transform], removejumps=[6]
        )
        self.iface.sync()
       

    """Calculates left and right grasps given grasp points (which can potentially be None)"""
    def calculate_grasps(self, grasp_left, grasp_right, g, iface):
    
        g.gripper_pos = 0.024
        #originally 0.0085
        if grasp_left is not None and grasp_right is not None:
            l_grasp, r_grasp = g.double_grasp(tuple(grasp_left), tuple(grasp_right), 0.004, 0.001, iface.L_TCP, iface.R_TCP, slide0=True, slide1=True)
            
            #setting to be thicker bc rope 
            l_grasp.gripper_pos = 0.02
            r_grasp.gripper_pos = 0.02
                   
        elif grasp_left is not None:
            l_grasp = g.single_grasp(tuple(grasp_left), 0.004, iface.L_TCP)
            l_grasp.gripper_pos = 0.02
            r_grasp = None
                
        elif grasp_right is not None:
            r_grasp = g.single_grasp(tuple(grasp_right), 0.001, iface.R_TCP)
            r_grasp.gripper_pos = 0.02
            l_grasp = None

        return l_grasp, r_grasp

    """Calculates place point intermediate and final transform given place point"""
    def calculate_place_transforms(self, place_point, g, iface, arm):
        place_3d = g.ij_to_point(place_point).data
        #so that it does not place too deep
        place_3d = place_3d + np.array([0, 0, 0.01])
        intermediate_3d = place_3d + np.array([0, 0, .1])

        if arm == "left":
            intermediate_transform = RigidTransform(
            translation=intermediate_3d,
            # rotation=iface.GRIP_DOWN_R,
            rotation = iface.y.left.get_pose().rotation,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
            )

            place_transform = RigidTransform(
            translation=place_3d,
            # rotation=iface.GRIP_DOWN_R,
            rotation = iface.y.left.get_pose().rotation,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
            )

        if arm == "right":
            intermediate_transform = RigidTransform(
            translation=intermediate_3d,
            # rotation=iface.GRIP_DOWN_R,
            rotation = iface.y.right.get_pose().rotation,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
            )

            place_transform = RigidTransform(
            translation=place_3d,
            # rotation=iface.GRIP_DOWN_R,
            rotation = iface.y.right.get_pose().rotation,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
            )




        return intermediate_transform, place_transform

    def calculate_grasp_transform(self, grasp_point, g, iface):
        grasp_3d = g.ij_to_point(grasp_point).data
        grasp_transform = RigidTransform(
        translation=grasp_3d,
        rotation=iface.GRIP_DOWN_R,
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
        )
        return grasp_3d

    def collect_full_demo(self, name):
        self.output_vis_dir = None
        self.save = None
        while True: 
            if self.iface is not None:
                self.iface.open_grippers()
                self.iface.home()
                self.iface.sync()

            input_chars = input("Press enter when ready to take an image, or press x once all demos are collected")
            if input_chars == 'x':
                break

            self.collect_demo_action()
        np.save("knot_demos/{}.npy".format(name), self.demos)
        print("done with demo!")
        return
        


    def collect_demo_action(self):
        self.img = self.iface.take_image(); 
        self.get_endpoints()
        trace_pixels, trace_crossings = self.get_trace_pixels(self.img, "right")
        self.visualize_all_crossings(self.img, trace_crossings)

        # gather click points from user
        print("Choose pick points")
        grasp_left, grasp_right = click_points_simple(self.img)
        print(grasp_left)
        print(grasp_right)
        
        # gather place points from user
        print("Choose place points")
        place_left, place_right= click_points_simple(self.img)
        print(place_left)
        print(place_right)


        if grasp_left is not None:
            #returns y, x point
            grasp_left = self.get_closest_trace_point(grasp_left[::-1], trace_pixels)
            #flip back
            grasp_left = grasp_left[::-1]
            
        if grasp_right is not None:
            #returns y, x point
            grasp_right = self.get_closest_trace_point(grasp_right[::-1], trace_pixels)
            #flip back
            grasp_right = grasp_right[::-1]

        

        #execute pick and place
        self.g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
        grasp_left_pose, grasp_right_pose = self.calculate_grasps(grasp_left, grasp_right, self.g, self.iface)
        self.iface.grasp(grasp_left_pose, grasp_right_pose)

        self.iface.sync()

        if place_left is not None:
            intermediate_transform, place_transform = self.calculate_place_transforms(place_left, self.g, self.iface, "left")
            self.iface.go_cartesian(
            l_targets=[intermediate_transform], removejumps=[5, 6]
            )
            self.iface.sync()
            self.iface.go_cartesian(
                l_targets=[place_transform], removejumps=[5, 6]
            )
            self.iface.sync()
        
        if place_right is not None:
            intermediate_transform, place_transform = self.calculate_place_transforms(place_right, self.g, self.iface, "right")
            self.iface.go_cartesian(
            r_targets=[intermediate_transform], removejumps=[5, 6]
            )
            self.iface.sync()
            self.iface.go_cartesian(
                r_targets=[place_transform], removejumps=[5, 6]
            )
            self.iface.sync()

        #tracer needs points in [y, x] format
        if grasp_left is not None: 
            grasp_left = grasp_left[::-1]
        
        if grasp_right is not None:
            grasp_right = grasp_right[::-1]
            
        if place_left is not None:
            place_left = place_left[::-1]
        
        if place_right is not None:
            place_right = place_right[::-1]


        self.demos.append({'img': self.img,'trace_pixels': trace_pixels,
        'trace_crossings': trace_crossings, 'grasp_left': grasp_left, 'grasp_right': grasp_right, 'place_left': place_left, 'place_right': place_right})

    """visualize the grasp and place points on the image"""

    def visualize(self, img, grasp_left, grasp_right, place_left, place_right, file_name=''):
        plt.clf()
        if grasp_left is not None:
            plt.scatter(grasp_left[1], grasp_left[0], c='b')
            plt.text(grasp_left[1]+0.1, grasp_left[0]+0.1, 'L', fontsize=12)

        if grasp_right is not None:
            plt.scatter(grasp_right[1], grasp_right[0], c='b', label='R')
            plt.text(grasp_right[1]+0.1, grasp_right[0]+0.1, 'R', fontsize=12)

        if place_left is not None:
            plt.scatter(place_left[1], place_left[0], c='r', label='L')
            plt.text(place_left[1]+0.1, place_left[0]+0.1, 'L', fontsize=12)
        
        if place_right is not None:
            plt.scatter(place_right[1], place_right[0], c='r', label='R')
            plt.text(place_right[1]+0.1, place_right[0]+0.1, 'R', fontsize=12)

        plt.imshow(img.color._data)

        if self.save:
            plt.savefig(self.output_vis_dir + file_name)

        if self.viz:
            plt.show()

    def visualize_demos(self, demos):
        for i in range(len(demos)):
            self.visualize(demos[i]['img'], demos[i]['grasp_left'], demos[i]['grasp_right'], demos[i]['place_left'], demos[i]['place_right'])
            self.visualize_all_crossings(demos[i]['img'], demos[i]['trace_crossings'])
            

    def visualize_all_crossings(self, img, crossings, file_name=''):
        plt.clf()
        u_clr = 'green'
        o_clr = 'orange'
        for ctr, crossing in enumerate(crossings):
            y, x = crossing['loc']
            if crossing['ID'] == 0:
                plt.scatter(x+1, y+1, c=u_clr, alpha=0.5)
                plt.text(x+10, y+10, str(ctr), fontsize=8, color=u_clr)
            if crossing['ID'] == 1:
                plt.scatter(x-1, y-1, c=o_clr, alpha=0.5)
                plt.text(x-10, y-10, str(ctr), fontsize=8, color=o_clr)
        plt.imshow(img.color._data)

        if self.save:
            plt.savefig(self.output_vis_dir + file_name)

        if self.viz:
            plt.show()

        
            
        
    def check_imitate_point(self, demo_dict, test_img, imit_type):
        #get data points of the demo image
        
        demo_trace = demo_dict['trace_pixels']
        demo_crossings = demo_dict['trace_crossings']
        demo_img = demo_dict['img']
        
        #find the trace and crossings of the test img
        self.img = test_img
        self.get_endpoints()
        test_trace_pixels, test_trace_crossings = self.get_trace_pixels(test_img, endpoint="left")
        #for now pick points so allow_off_cable=False
        self.visualize_all_crossings(demo_img, demo_crossings)
        self.visualize_all_crossings(test_img, test_trace_crossings)

        for point in ['grasp_left', 'grasp_right', 'place_left', 'place_right']:
            if demo_dict[point] is not None:
                demo_point = demo_dict[point]
                
                if point == 'place_left' or point == 'place_right':
                    allow_off_cable = True
                    color="r"
                else:
                    allow_off_cable = False
                    color="b"

                plt.imshow(demo_img.color._data)
                plt.scatter(demo_point[1], demo_point[0], c=color)
                plt.title('Demo')
                plt.show()

                test_point = self.imitate_point(demo_point, demo_trace, demo_crossings, test_trace_pixels, test_trace_crossings, imit_type = imit_type, allow_off_cable=allow_off_cable)
                #plot the demo image and point along with the test image and point
                plt.subplot(1, 2, 1)
                plt.imshow(demo_img.color._data)
                plt.scatter(demo_point[1], demo_point[0], c=color)
                plt.title('Demo')
                plt.subplot(1, 2, 2)
                plt.imshow(test_img.color._data)
                plt.scatter(test_point[1], test_point[0], c=color)
                plt.title('Test')
                plt.show()


    def collect_images(self, start_idx):
        idx = start_idx
        while True:
            input_chars = input("Press enter when ready to take an image, or press x when done")
            if input_chars == 'x':
                break  
            img = self.iface.take_image()
            color_img = img.color._data
            plt.imshow(color_img)
            plt.show()

            # cv2.imwrite("knot_tye_imgs/"+ "oh_" + str(idx) + ".png", color_img)
            np.save( "knot_tye_imgs/"+ "oh_" + str(idx) + ".npy", img)
            idx += 1
    
    def find_best_arm(self, grasp_left, grasp_right, place_left, place_right):
        #reassign grasps and places to the best arm
        if grasp_left is not None and grasp_right is not None:
            average_grasp_place_left = grasp_left
            average_grasp_place_right = grasp_right
            #if places exists, take the average of the grasp and place
            if place_left is not None:
                average_grasp_place_left = (grasp_left + place_left)/2
            if place_right is not None:
                average_grasp_place_right = (grasp_right + place_right)/2

            #checking if x coord is to the right of the center of the image
            if average_grasp_place_left[1] > average_grasp_place_right[1]:
                grasp_left, grasp_right = grasp_right, grasp_left
                place_left, place_right = place_right, place_left
        
            
        elif grasp_left is not None:
            average_grasp_place = grasp_left
            if place_left is not None:
                average_grasp_place = (grasp_left + place_left)/2
                #checking if x coord is to the right of the center of the image
            if average_grasp_place[1] > 590:
                grasp_right, grasp_left = grasp_left, grasp_right
                place_right, place_left = place_left, place_right

        elif grasp_right is not None:
            average_grasp_place = grasp_right
            if place_right is not None:
                average_grasp_place = (grasp_right + place_right)/2
                #checking if x coord is to the right of the center of the image
            if average_grasp_place[1] < 590:
                grasp_right, grasp_left = grasp_left, grasp_right
                place_right, place_left = place_left, place_right
            

        return grasp_left, grasp_right, place_left, place_right

    def mkdir_get_path(self, save_dir):
        prev_folders = os.listdir(save_dir)
        curr_folder = str(max([int(f) for f in prev_folders]) + 1)
        folder = save_dir + "/" + curr_folder + "/"  
        os.mkdir(folder)  
        return folder 

    def exec_demo_step(self, demos, i, save_dir = None):
        if save_dir is not None:
            self.output_vis_dir = save_dir
            if not os.path.exists(self.output_vis_dir):
                os.mkdir(self.output_vis_dir)
            self.save = True
        else:
            self.save=False
        self.demo_counter = i
        if i == 0:
            pick_imit_type = 'distance'
            place_imit_type = 'distance'
        else: 
            pick_imit_type = 'crossing'
            place_imit_type = 'crossing'

        grasp_left = None
        grasp_right = None
        place_left = None
        place_right = None

        self.img = self.iface.take_image()
        self.get_endpoints()
        if self.endpoints.shape[0] == 0:
            failure = True
        trace_pixels_exec, trace_crossings_exec = self.get_trace_pixels(self.img, endpoint="right")

        self.visualize_all_crossings(demos[i]['img'], demos[i]['trace_crossings'], file_name= "demo_crossings_{}".format(i))
        self.visualize_all_crossings(self.img, trace_crossings_exec, file_name="exec_crossings_{}".format(i))

        if demos[i]['grasp_left'] is not None:
            grasp_left = self.imitate_point(demos[i]['grasp_left'], demos[i]['trace_pixels'], demos[i]['trace_crossings'], trace_pixels_exec, trace_crossings_exec, demos[i]['img'], imit_type=pick_imit_type)
            if demos[i]['place_left'] is not None:
                place_left = self.imitate_point(demos[i]['place_left'], demos[i]['trace_pixels'], demos[i]['trace_crossings'], trace_pixels_exec, trace_crossings_exec, demos[i]['img'], imit_type=place_imit_type, allow_off_cable=True)
                
        if demos[i]['grasp_right'] is not None:
            grasp_right = self.imitate_point(demos[i]['grasp_right'], demos[i]['trace_pixels'], demos[i]['trace_crossings'], trace_pixels_exec, trace_crossings_exec, demos[i]['img'], imit_type=pick_imit_type)  
            if demos[i]['place_right'] is not None:
                place_right = self.imitate_point(demos[i]['place_right'], demos[i]['trace_pixels'], demos[i]['trace_crossings'], trace_pixels_exec, trace_crossings_exec, demos[i]['img'], imit_type=place_imit_type, allow_off_cable=True)

        grasp_left, grasp_right, place_left, place_right = self.find_best_arm(grasp_left, grasp_right, place_left, place_right)        
        #visualize crossings

        self.visualize(demos[i]['img'], demos[i]['grasp_left'], demos[i]['grasp_right'], demos[i]['place_left'], demos[i]['place_right'], file_name="demos_pick_place_{}".format(i))
        self.visualize(self.img, grasp_left, grasp_right, place_left, place_right, file_name="exec_pick_place_{}".format(i))

        # calculate and execute the the grasps
        self.g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
        #switching back to x, y
        if grasp_left is not None:
            grasp_left = grasp_left[::-1]
        if grasp_right is not None:
            grasp_right = grasp_right[::-1]

        l_grasp, r_grasp = self.calculate_grasps(grasp_left, grasp_right, self.g, self.iface)
        self.iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp)

        #calculate and execute the places
        if place_left is not None:
            #switching back to x, y
            place = [int(place_left[1]), int(place_left[0])]
            intermediate_transform, place_transform = self.calculate_place_transforms(place, self.g, self.iface, "left")
            
            self.iface.go_cartesian(
            l_targets=[intermediate_transform], removejumps=[5, 6]
            )
            self.iface.sync()


            self.iface.go_cartesian(
            l_targets=[place_transform], removejumps=[5, 6]
            )
            self.iface.sync()

            #calculate and execute the places
        if place_right is not None:
            #switching back to x, y
            place = [int(place_right[1]), int(place_right[0])]
            intermediate_transform, place_transform = self.calculate_place_transforms(place, self.g, self.iface, "right")

            self.iface.go_cartesian(
            r_targets=[intermediate_transform], removejumps=[5, 6]
            )
            self.iface.sync()


            self.iface.go_cartesian(
            r_targets=[place_transform], removejumps=[5, 6]
            )
            self.iface.sync()    

        self.iface.open_grippers()
        self.iface.sync()
        self.iface.home()
        self.iface.sync()

    





    def exec_demos(self, demos, save_dir = None, two_step_loop=False) :
        # if save_dir is not None:
        #     self.output_vis_dir = self.mkdir_get_path(save_dir)
        #     self.save=True

        if save_dir is not None:
            self.output_vis_dir = save_dir
            if not os.path.exists(self.output_vis_dir):
                os.mkdir(self.output_vis_dir)
            self.save = True
        else:
            self.save=False

        self.demo_counter = 0

        print("path", self.output_vis_dir)
        for i in range(len(demos)):
            if two_step_loop:
                if i == 0 or i == 1:
                    pick_imit_type = 'distance'
                    place_imit_type = 'distance'
                else: 
                    pick_imit_type = 'crossing'
                    place_imit_type = 'crossing'

            else:
                if i == 0:
                    pick_imit_type = 'distance'
                    place_imit_type = 'distance'
                else: 
                    pick_imit_type = 'crossing'
                    place_imit_type = 'crossing'


            self.iface.open_grippers()
            self.iface.sync()
            self.iface.home()
            self.iface.sync()

            grasp_left = None
            grasp_right = None
            place_left = None
            place_right = None

            self.img = self.iface.take_image()
            self.get_endpoints()
            if self.endpoints.shape[0] == 0:
                failure = True
            trace_pixels_exec, trace_crossings_exec = self.get_trace_pixels(self.img, endpoint="right")

            self.visualize_all_crossings(demos[i]['img'], demos[i]['trace_crossings'], file_name= "demo_crossings_{}".format(i))
            self.visualize_all_crossings(self.img, trace_crossings_exec, file_name="exec_crossings_{}".format(i))

            if demos[i]['grasp_left'] is not None:
                grasp_left = self.imitate_point(demos[i]['grasp_left'], demos[i]['trace_pixels'], demos[i]['trace_crossings'], trace_pixels_exec, trace_crossings_exec, demos[i]['img'], imit_type=pick_imit_type)
                if demos[i]['place_left'] is not None:
                    place_left = self.imitate_point(demos[i]['place_left'], demos[i]['trace_pixels'], demos[i]['trace_crossings'], trace_pixels_exec, trace_crossings_exec, demos[i]['img'], imit_type=place_imit_type, allow_off_cable=True)
                    
            if demos[i]['grasp_right'] is not None:
                grasp_right = self.imitate_point(demos[i]['grasp_right'], demos[i]['trace_pixels'], demos[i]['trace_crossings'], trace_pixels_exec, trace_crossings_exec, demos[i]['img'], imit_type=pick_imit_type)  
                if demos[i]['place_right'] is not None:
                    place_right = self.imitate_point(demos[i]['place_right'], demos[i]['trace_pixels'], demos[i]['trace_crossings'], trace_pixels_exec, trace_crossings_exec, demos[i]['img'], imit_type=place_imit_type, allow_off_cable=True)

            grasp_left, grasp_right, place_left, place_right = self.find_best_arm(grasp_left, grasp_right, place_left, place_right)        
            #visualize crossings

            self.visualize(demos[i]['img'], demos[i]['grasp_left'], demos[i]['grasp_right'], demos[i]['place_left'], demos[i]['place_right'], file_name="demos_pick_place_{}".format(i))
            self.visualize(self.img, grasp_left, grasp_right, place_left, place_right, file_name="exec_pick_place_{}".format(i))

            # calculate and execute the the grasps
            self.g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
            #switching back to x, y
            if grasp_left is not None:
                grasp_left = grasp_left[::-1]
            if grasp_right is not None:
                grasp_right = grasp_right[::-1]

            l_grasp, r_grasp = self.calculate_grasps(grasp_left, grasp_right, self.g, self.iface)
            self.iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp)

            #calculate and execute the places
            if place_left is not None:
                #switching back to x, y
                place = [int(place_left[1]), int(place_left[0])]
                intermediate_transform, place_transform = self.calculate_place_transforms(place, self.g, self.iface, "left")
                
                self.iface.go_cartesian(
                l_targets=[intermediate_transform], removejumps=[5, 6]
                )
                self.iface.sync()


                self.iface.go_cartesian(
                l_targets=[place_transform], removejumps=[5, 6]
                )
                self.iface.sync()

                #calculate and execute the places
            if place_right is not None:
                #switching back to x, y
                place = [int(place_right[1]), int(place_right[0])]
                intermediate_transform, place_transform = self.calculate_place_transforms(place, self.g, self.iface, "right")

                self.iface.go_cartesian(
                r_targets=[intermediate_transform], removejumps=[5, 6]
                )
                self.iface.sync()


                self.iface.go_cartesian(
                r_targets=[place_transform], removejumps=[5, 6]
                )
                self.iface.sync()    

            self.demo_counter += 1
        self.iface.open_grippers()
        self.iface.sync()
        self.iface.home()
        self.iface.sync()

    


    # def run_pipeline(self):
    #     start_time = int(time.time())
    #     self.logs_file.write(f'Start time: {start_time}\n')
    #     count = 0
    #     diff = int(time.time()) - start_time


    #     while not done or diff < self.time_limit:
    #         try:
    #             time.sleep(1)
    #             # collect demos
    #             print("COLLECTING DEMOS")

    #             demos = []
    #             i = 0
    #             while True: 
    #                 if self.iface is not None:
    #                     self.iface.open_grippers()
    #                     self.iface.home()
    #                     self.iface.sync()

    #                 input_chars = input("Press enter when ready to take an image, or press x once all demos are collected")
    #                 if input_chars == 'x':
    #                     break

    #                 self.collect_demo()
    #                 i += 1

    #             # now process demos for execution on robot
    #             input("Press enter when ready to execute policy learned from demos")
    #             # self.img = self.take_and_save_image(path='/home/justin/yumi/cable-untangling/scripts/oct_11_more_overhand/colorP_7.npy')
    #             # self.img = self.take_and_save_image()
                


    #             diff = int(time.time()) - start_time
    #         except Exception as e:
    #             # check if message says "nonresponsive"
    #             if "nonresponsive after sync" in str(e) or \
    #                 "Couldn't plan path" in str(e) or \
    #                 "No grasp pair" in str(e) or \
    #                 "Timeout" in str(e) or \
    #                 "No collision free grasps" in str(e) or \
    #                 "Not enough points traced" in str(e) or \
    #                 "Jump in first 4 joints" in str(e) or \
    #                 "Planning to pose failed" in str(e):
    #                 self.logger.info(f"Caught error, recovering {str(e)}")
    #                 failure = True
    #                 self.iface.y.reset()
    #             else:
    #                 raise e
    #                 self.logger.info("Uncaught exception, still recovering " + str(e))collect_f
    #                 self.iface.y.reset()
    #             self.iface.sync()
                
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

    fullPipeline = DemoPipeline(viz=False, loglevel=logLevel, initialize_iface=True)
    # demo_dict = np.load("knot_demos/oh2.npy", allow_pickle=True)[2]
    

    # test_img = np.load("knot_tye_imgs/oh_11.npy", allow_pickle=True).item()
    # fullPipeline.check_imitate_point(demo_dict, test_img, imit_type="distance")
    fullPipeline.collect_full_demo("Sep8_2_step_loop_center")

    # fullPipeline.collect_images(0)

    # demos = np.load("knot_demos/Sep8_2_step_loop.npy", allow_pickle=True)
    # fullPipeline.exec_demos(demos, "./test_demos/", two_step_loop=True)


    #useful code
      # l_grasp, r_grasp = self.g.double_grasp(
                #     tuple(grasp1[::-1]), tuple(grasp2[::-1]), .0085, .0085, self.iface.L_TCP, self.iface.R_TCP, slide0=True, slide1=False)
                # self.iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp)

    #debugging
    # print("self.iface.cam", self.iface.cam)
    # print("cam intsinsics", self.iface.cam.intrinsics.fx, self.iface.cam.intrinsics.fy, self.iface.cam.intrinsics.cx, self.iface.cam.intrinsics.cy)
    # print("proj mat", self.iface.cam.intrinsics.K)
    # print("frame", self.iface.cam.intrinsics.frame)
    # print("T_PHOXI_BASE", self.iface.T_PHOXI_BASE)
    # print("points_3d", self.g.points_3d)
  
