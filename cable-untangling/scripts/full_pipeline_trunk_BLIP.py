# 3 lines of detectron imports
from codecs import IncrementalDecoder
from pickle import FALSE
from turtle import done, left
import sys
sys.path.append('/home/mallika/triton4-lip/') 

from detectron2_repo import analysis as loop_detectron
from untangling.utils.grasp import GraspSelector, GraspException
from untangling.utils.interface_rws_BLIP import Interface
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
import numpy as np
from untangling.utils.tcps import *

from untangling.utils.cable_tracing.cable_tracing.tracers.simple_uncertain_trace import trace
# from untangling.shake import shake
import time
from untangling.point_picking import *
import matplotlib.pyplot as plt
# from fcxvision.kp_wrapper import KeypointNetwork
# from untangling.spool import init_endpoint_orientation,execute_spool
# from untangling.keypoint_untangling import closest_valid_point_to_pt
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

# NEED TO BRING THESE BACK LATER
from untangling.tracer_knot_detect.tracer_knot_detection import TracerKnotDetector
from untangling.tracer_knot_detect.tracer import TraceEnd

torch.cuda.empty_cache()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

class FullPipeline():
    def __init__(self, viz, loglevel, topdown_grasp=False, initialize_iface=True):
        self.time_limit = 15 * 60
        self.SPEED = (0.4, 6 * np.pi) #(0.6, 6 * np.pi)
        self.iface = None
        if initialize_iface:
            self.iface = Interface(
                "1703005",
                ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
                speed=self.SPEED,
            )

        self.total_obs = 0
        self.viz = viz
        self.alg_output_fig = 'alg_output.png'
        self.img_count = 0
        self.img_folder = 'full_pipeline_fig/test_detector'

        check_path(self.img_folder)

        
        self.date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory = f'logs_{self.date_time}'
        parent_dir = f'./logs/full_rollouts/'
        path = os.path.join(parent_dir, directory) 
        check_path(path) 
        print("Directory '%s' created" %directory) 
        logs_file_name = f'./logs/full_rollouts/logs_{self.date_time}/debug_info_log.txt'
        print(os.path.dirname(logs_file_name))
        check_path(os.path.dirname(logs_file_name))

        self.image_save_path = f'./logs/full_rollouts/logs_{self.date_time}/images'
        self.trace_save_path = f'./logs/full_rollouts/logs_{self.date_time}/traces'

        check_path(self.image_save_path)
        check_path(self.trace_save_path)

        logging.root.setLevel(loglevel)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=logs_file_name, level=loglevel, format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%H:%M:%S')
        self.logger = logging.getLogger("Untangling")

        # self.logs_file = open(logs_file_name, 'w')
        self.topdown_grasps = topdown_grasp
        self.tkd = TracerKnotDetector()
        self.action_count = 0
        self.g = None
        self.img = None
        self.endpoints = None
        self.grasppoint = None
        self.start_time = 0

        self.img_dims = np.array([772, 1032])
        left_arm_mask_poly = (np.array([(0, 174), (0, 36), (24, 24), (48, 18), (96, 24), (148, 48), (186, 75), (231, 148), (236, 186)])*4.03125).astype(np.int32)
        right_arm_mask_poly = (np.array([(30, 176), (35, 150), (70, 77), (96, 52), (160, 26), (255, 22), (255, 186)])*4.03125).astype(np.int32)
        left_arm_reachability_mask = self.draw_mask(left_arm_mask_poly)
        right_arm_reachability_mask = self.draw_mask(right_arm_mask_poly)
        self.overall_reachability_mask = (left_arm_reachability_mask + right_arm_reachability_mask) > 0
        self.overall_reachability_mask[:, 900:] = 0.0
        self.overall_reachability_mask[:, :200] = 0.0

    def draw_mask(self, polygon_coords, blur=None):
        image = np.zeros(shape=self.img_dims, dtype=np.float32)
        cv2.drawContours(image, [polygon_coords], 0, (1.0), -1)    
        if blur is not None:        
            image = cv2.GaussianBlur(image, (blur, blur), 0)
        return image

    def take_and_save_image(self, path=None, first=False):
        self.total_obs += 1
        if path is None:
            img = self.iface.take_image()
        else:
            rgb_image = np.load(path).astype(np.float32)# / 255.0
            # create fake depth channel and append
            depth_image = np.ones((rgb_image.shape[0], rgb_image.shape[1], 1))
            rgbd_image = np.concatenate((rgb_image, depth_image), axis=2)
            img = RgbdImage(rgbd_image)
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # if first:
        #     np.save(osp.join(self.start_images, cur_time + '.npy'), img._data)
        np.save(osp.join(self.image_save_path, cur_time + '.npy'), img._data)
        # img._data = np.load('/home/justin/yumi/cable-untangling/scripts/live_rollout_RSS2023/2023-01-27_16-50-39.npy')
        # plt.imshow(img.color._d
        # plt.show()
        diff = int(time.time()) - self.start_time
        self.print_and_log(f'TIME since beginning: {diff}\n')
        return img

    def show_img(self):
        if self.viz:
            plt.show()
        else:
            plt.savefig(self.alg_output_fig)
            plt.savefig(self.img_folder + "_" + str(self.img_count))
            self.img_count += 1


    def print_and_log(self, *args):
        self.logger.info(' '.join(map(str, args)))
        self.logs_file.write(' '.join(map(str, args)) + '\n')
        self.logs_file.flush()

    def closest_valid_point(self, color, depth, yx):
        valid_y, valid_x = np.nonzero((color[:, :, 0] > 100) * (depth[:, :, 0] > 0))
        pts = np.vstack((valid_y, valid_x)).T
        return pts[np.argmin(np.linalg.norm(pts - np.array(yx)[None, :], axis=-1))]

    def get_endpoints(self, img=None):
        # model not used, already specified in loop_detectron
        if img is not None:
            self.img = img
        else:
            self.img = self.iface.take_image()
        endpoint_boxes, out_viz = loop_detectron.predict(self.img.color._data, thresh=0.99, endpoints=True)
        plt.clf()
        plt.imshow(out_viz)
        plt.title("Endpoints detected")
        self.show_img()

        endpoints = []
        for box in endpoint_boxes:
            xmin, ymin, xmax, ymax = box
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2
            new_yx = self.closest_valid_point(self.img.color._data, self.img.depth._data, np.array([y,x]))
            endpoints.append([new_yx, new_yx])   
        endpoints = np.array(endpoints).astype(np.int32)
        # (n,2,2) (num endpoints, (head point, neck point), (x,y))
        endpoints = np.array([e for e in endpoints if self.img.depth._data[e[0][0], e[0][1]] > 0])
        self.logger.debug("Found {} true endpoints after filtering for depth".format(len(endpoints)))

        plt.clf()
        plt.title("Endpoint detections visualization")
        plt.imshow(self.img.color._data)
        if len(endpoints) > 0:
            plt.scatter(endpoints[:, 0, 1], endpoints[:, 0, 0])
        self.show_img()

        endpoints = endpoints.astype(np.int32).reshape(-1, 2, 2)
        # endpoint1 = endpoints[0][0]
        # endpoint2 = endpoints[1][0]
        self.endpoints = endpoints[:, 0, :] #np.array([endpoint1, endpoint2])

    # can click 1 grasp point
    def get_single_grasp_from_click(self, take_image=True):
        if take_image:
            self.img = self.iface.take_image()
        grasppoint= click_points_simple(self.img) 
        if grasppoint is not None:
            self.grasppoint = grasppoint
            print(self.grasppoint)

    # can click 1 or 2 endpoints
    def get_endpoints_from_clicks(self, take_image=True):
        if take_image:
            self.img = self.iface.take_image()
        endpoint_1, endpoint_2= click_points_simple(self.img) 
        if endpoint_1 is not None and endpoint_2 is not None:
            self.endpoints = np.array([endpoint_1[::-1], endpoint_2[::-1]])
        elif endpoint_1 is not None:
            self.endpoints = np.array([endpoint_1[::-1]])
        else:
            self.endpoints = np.array([endpoint_2[::-1]])

    def get_grasp_points_from_endpoint(self, endpoint):
        # trace from endpoint on image
        img = np.concatenate((self.img.color._data, self.img.depth._data), axis=-1)
        img[-130:, ...] = 0
        thresh_img = np.where(img[:,:,:3] > 100, 255, 0)

        full_img = np.concatenate((thresh_img, img[:,:,3:]), axis=-1)
        path, _ = trace(full_img, endpoint, None, exact_path_len=6, viz=self.viz)
        if path is None:
            return tuple(np.array([endpoint[0], endpoint[1]]).astype(np.int32)), True
        corrected_pick_point = self.closest_valid_point(self.img.color._data, self.img.depth._data, path[-3]) # takes in x,y and outputs x,y
        return tuple(corrected_pick_point.astype(np.int32)), False

    def get_trace_from_endpoint(self, point):
        # trace from endpoint on image
        img = np.concatenate((self.img.color._data, self.img.depth._data), axis=-1)
        img[-130:, ...] = 0
        thresh_img = np.where(img[:,:,:3] > 100, 255, 0)

        full_img = np.concatenate((thresh_img, img[:,:,3:]), axis=-1)
        # path, finished_paths = trace(full_img, point, None, exact_path_len=4, viz=self.viz)
        path, finished_paths = trace(full_img, point, None, exact_path_len=3, viz=self.viz)
        agree_point = tuple(np.array([point[0], point[1]]).astype(np.int32))
        if path is None:
            if len(finished_paths) > 0:
                min_len_path = float('inf')
                for fp in finished_paths:
                    if len(fp) < min_len_path:
                        min_len_path = len(fp)
                for i in range(min_len_path):
                    values = np.array([list(fp[i]) for fp in finished_paths])
                    x = values[:, 0]
                    y = values[:, 1]
                    x_diff = np.max(x)-np.min(x)
                    y_diff = np.max(y)-np.min(y)
                    if x_diff < 5 and y_diff < 5:
                        agree_point = values[0]
            return agree_point, True
        return path, False

    def find_clear_area(self, image, side=None):
        image = image.copy()
        image[650:,:] = image.max()
        vis_image = image.copy()
        # side reflects the region you're placing in and the opposite gripper is used
        if side == 'left':
            image[:, 500:] = image.max()
            right_arm_reachability = self.iface.get_right_reachability_mask(image)
            image[right_arm_reachability == 0] = image.max()
        elif side == 'right':
            image[:, :550] = image.max()
            left_arm_reachability = self.iface.get_left_reachability_mask(image)
            image[left_arm_reachability == 0] = image.max()
        
        # find farthest black pixel from any white pixel using cv2.distanceTransform
        # https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html
        
        img = np.where(image > 100, 0, 255)
        img = img.astype(np.uint8)
        dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        # find argmax of distance transform
        argmax = np.unravel_index(np.argmax(dist), dist.shape)

        # plt.clf()
        # plt.imshow(vis_image)
        # plt.title("Place point")
        # plt.scatter(*argmax)
        # plt.show()

        return argmax

    def run_endpoint_separation_incremental(self):
        '''
        Pull apart 0.3m, and while detectron detects no knots, keep pulling apart by specified increments 
        until knot detected or ceiling=0.6m reached. 
        
        Tilt not implemented because regrasping should take care of it.
        '''
        MAX_DISTANCE = 0.5
        g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE,grabbing_endpoint=True)

        endpt1, on_endpt1 = self.get_grasp_points_from_endpoint(self.endpoints[0])
        endpt2, on_endpt2 = self.get_grasp_points_from_endpoint(self.endpoints[1])
        endpt1 = endpt1[::-1]
        endpt2 = endpt2[::-1]

        plt.clf()
        plt.scatter(*endpt1)
        plt.scatter(*endpt2)
        plt.imshow(self.img.color._data * self.overall_reachability_mask[:, :, None])
        plt.show()
        # self.show_img()
        # endpoints_tuples = endpoints
        self.iface.home()
        self.iface.sync()
        
        # change by m - making double grasp different
        # l_grasp,r_grasp=g.double_grasp(endpt1,endpt2,.0085,.0085,self.iface.L_TCP,self.iface.R_TCP,slide0=on_endpt1, slide1=on_endpt2) #0.005 each
        l_grasp, r_grasp = g.double_grasp(endpt1,endpt2, 0.01, 0.01, self.iface.L_TCP, self.iface.R_TCP, slide0=True, slide1=True)
        #TODO: REMOVE THIS FROM INFO ONCE DEBUGGED
        logging.info(f"l_grasp: {l_grasp.pose.translation}")
        logging.info(f"r_grasp: {r_grasp.pose.translation}")
        l_grasp.pregrasp_dist, r_grasp.pregrasp_dist = 0.05, 0.05
        self.iface.open_grippers()
        #self.iface.sync()
        self.iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp,topdown=self.topdown_grasps,reid_grasp=True)
        self.iface.set_speed((.4,5))

        def l_to_r(config):
                return np.array([-1, 1, -1, 1, -1, 1, 1]) * config

        def get_clear_area(img, side='left'):
            try:
                g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE,grabbing_endpoint=True)
                clear_point = self.find_clear_area(img, side=side)
                # plt.imshow(self.img._data[..., :3].astype(np.uint8))
                # plt.scatter(*clear_point[::-1])
                # plt.show()
                clear_point = g.ij_to_point(clear_point[::-1]).data
                clear_point[2] = 0.07 # CHANGE HERE: 0.1 --> 0.07
            except Exception as e:
                raise e
                clear_point = None
            return clear_point

        def return_to_center():
            self.iface.home()
            self.iface.sync()
            self.img = self.take_and_save_image()
            color_img = self.img.color._data
            color_img = np.array(color_img, dtype=np.uint8)[:, :, 0]

            default_right_pose = [0.35, 0.15, 0.13] # CHANGE HERE: 0.1 --> 0.13
            clear_point_right_arm = get_clear_area(color_img, side='left')
            clear_point_right_arm = default_right_pose if clear_point_right_arm is None else clear_point_right_arm
            if np.linalg.norm(np.array(default_right_pose) - np.array(clear_point_right_arm)) > 0.2:
                clear_point_right_arm = default_right_pose
            rp = RigidTransform(
                translation=clear_point_right_arm,
                rotation=self.iface.GRIP_DOWN_R,  # 0.1 0.2
                from_frame=YK.r_tcp_frame,
                to_frame="base_link",
            )

            self.iface.home()
            self.iface.sync()
            try:
                self.iface.go_cartesian(
                    r_targets=[rp]
                )
            except:
                self.logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.iface.go_pose_plan(r_target=rp)
            self.iface.sync()
            self.iface.release_cable("right")
            self.iface.sync()
            self.iface.go_config_plan(r_q=self.iface.R_HOME_STATE)
            self.iface.sync()
            time.sleep(1)

            self.img = self.take_and_save_image()
            color_img = self.img.color._data
            default_left_pose = [0.35, -0.15, 0.13] # CHANGE HERE: 0.1 --> 0.13
            color_img = self.img.color._data
            color_img = np.array(color_img, dtype=np.uint8)[:, :, 0]
            clear_point_left_arm = get_clear_area(color_img, side='right')
            clear_point_left_arm = default_left_pose if clear_point_left_arm is None else clear_point_left_arm
            if np.linalg.norm(np.array(default_left_pose) - np.array(clear_point_left_arm)) > 0.2:
                clear_point_left_arm = default_left_pose

            lp = RigidTransform(
                translation=clear_point_left_arm,
                rotation=self.iface.GRIP_DOWN_R,
                from_frame=YK.l_tcp_frame,
                to_frame="base_link",
            )

            try:
                self.iface.go_cartesian(
                    l_targets=[lp]
                )
            except:
                self.logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.iface.go_pose_plan(l_target=lp)
            self.iface.sync()
            self.iface.release_cable("left")
            self.iface.sync()
            self.iface.go_config_plan(l_q=self.iface.L_HOME_STATE)
            self.iface.sync()
        self.iface.pull_apart(0.6, slide_left=False, slide_right=False, return_to_center=False)
        self.iface.sync()
        
        # lift up
        cur_left_pos = self.iface.y.left.get_joints()
        cur_right_pos = self.iface.y.right.get_joints()
        OUT_POS_2_L = np.array([-0.80388363, -0.77492968, -1.30770091, -0.73786254,  2.85080849, 0.79577677,  2.01228079])
        OUT_POS_2_R = l_to_r(OUT_POS_2_L)
        OUT_POS_2_R[6] = cur_right_pos[6]
        self.iface.go_configs(l_q=[OUT_POS_2_L], r_q=[OUT_POS_2_R])
        self.iface.sync()
        # time.sleep(5)
        return_to_center()
        self.iface.sync()
        self.iface.open_grippers()
        self.iface.home()
        self.iface.sync()

    
    def pull_apart(self):
        '''
        Pull apart 0.3m, and while detectron detects no knots, keep pulling apart by specified increments 
        until knot detected or ceiling=0.6m reached. 
        
        Tilt not implemented because regrasping should take care of it.
        '''
        self.get_endpoints()
        MAX_DISTANCE = 0.5
        g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE,grabbing_endpoint=True)

        endpt1, on_endpt1 = self.get_grasp_points_from_endpoint(self.endpoints[0])
        endpt2, on_endpt2 = self.get_grasp_points_from_endpoint(self.endpoints[1])
        endpt1 = endpt1[::-1]
        endpt2 = endpt2[::-1]

        plt.clf()
        plt.scatter(*endpt1)
        plt.scatter(*endpt2)
        plt.imshow(self.img.color._data * self.overall_reachability_mask[:, :, None])
        plt.show()
        # self.show_img()
        # endpoints_tuples = endpoints
        self.iface.home()
        self.iface.sync()
        
        # change by m - making double grasp different
        # l_grasp,r_grasp=g.double_grasp(endpt1,endpt2,.0085,.0085,self.iface.L_TCP,self.iface.R_TCP,slide0=on_endpt1, slide1=on_endpt2) #0.005 each
        l_grasp, r_grasp = g.double_grasp(endpt1,endpt2, 0.01, 0.01, self.iface.L_TCP, self.iface.R_TCP, slide0=True, slide1=True)
        #TODO: REMOVE THIS FROM INFO ONCE DEBUGGED
        logging.info(f"l_grasp: {l_grasp.pose.translation}")
        logging.info(f"r_grasp: {r_grasp.pose.translation}")
        l_grasp.pregrasp_dist, r_grasp.pregrasp_dist = 0.05, 0.05
        self.iface.open_grippers()
        #self.iface.sync()
        self.iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp,topdown=self.topdown_grasps,reid_grasp=True)
        self.iface.set_speed((.4,5))

        def l_to_r(config):
                return np.array([-1, 1, -1, 1, -1, 1, 1]) * config

        def get_clear_area(img, side='left'):
            try:
                g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE,grabbing_endpoint=True)
                clear_point = self.find_clear_area(img, side=side)
                # plt.imshow(self.img._data[..., :3].astype(np.uint8))
                # plt.scatter(*clear_point[::-1])
                # plt.show()
                clear_point = g.ij_to_point(clear_point[::-1]).data
                clear_point[2] = 0.07 # CHANGE HERE: 0.1 --> 0.07
            except Exception as e:
                raise e
                clear_point = None
            return clear_point

        def return_to_center():
            self.iface.home()
            self.iface.sync()
            self.img = self.take_and_save_image()
            color_img = self.img.color._data
            color_img = np.array(color_img, dtype=np.uint8)[:, :, 0]

            default_right_pose = [0.35, -0.15, 0.13] # CHANGE HERE: 0.1 --> 0.13
            clear_point_right_arm = get_clear_area(color_img, side='right')
            clear_point_right_arm = default_right_pose if clear_point_right_arm is None else clear_point_right_arm
            if np.linalg.norm(np.array(default_right_pose) - np.array(clear_point_right_arm)) > 0.2:
                clear_point_right_arm = default_right_pose
            rp = RigidTransform(
                translation=clear_point_right_arm,
                rotation=self.iface.GRIP_DOWN_R,  # 0.1 0.2
                from_frame=YK.r_tcp_frame,
                to_frame="base_link",
            )

            self.iface.home()
            self.iface.sync()
            try:
                self.iface.go_cartesian(
                    r_targets=[rp]
                )
            except:
                self.logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.iface.go_pose_plan(r_target=rp)
            self.iface.sync()
            self.iface.release_cable("right")
            self.iface.sync()
            self.iface.go_config_plan(r_q=self.iface.R_HOME_STATE)
            self.iface.sync()
            time.sleep(1)

            self.img = self.take_and_save_image()
            color_img = self.img.color._data
            default_left_pose = [0.35, 0.15, 0.13] # CHANGE HERE: 0.1 --> 0.13
            color_img = self.img.color._data
            color_img = np.array(color_img, dtype=np.uint8)[:, :, 0]
            clear_point_left_arm = get_clear_area(color_img, side='left')
            clear_point_left_arm = default_left_pose if clear_point_left_arm is None else clear_point_left_arm
            if np.linalg.norm(np.array(default_left_pose) - np.array(clear_point_left_arm)) > 0.2:
                clear_point_left_arm = default_left_pose

            lp = RigidTransform(
                translation=clear_point_left_arm,
                rotation=self.iface.GRIP_DOWN_R,
                from_frame=YK.l_tcp_frame,
                to_frame="base_link",
            )

            try:
                self.iface.go_cartesian(
                    l_targets=[lp]
                )
            except:
                self.logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.iface.go_pose_plan(l_target=lp)
            self.iface.sync()
            self.iface.release_cable("left")
            self.iface.sync()
            self.iface.go_config_plan(l_q=self.iface.L_HOME_STATE)
            self.iface.sync()
        self.iface.pull_apart(0.6, slide_left=False, slide_right=False, return_to_center=False)
        self.iface.sync()
        
        # lift up
        cur_left_pos = self.iface.y.left.get_joints()
        cur_right_pos = self.iface.y.right.get_joints()
        OUT_POS_2_L = np.array([-0.80388363, -0.77492968, -1.30770091, -0.73786254,  2.85080849, 0.79577677,  2.01228079])
        OUT_POS_2_R = l_to_r(OUT_POS_2_L)
        OUT_POS_2_R[6] = cur_right_pos[6]
        self.iface.go_configs(l_q=[OUT_POS_2_L], r_q=[OUT_POS_2_R])
        self.iface.sync()
        # time.sleep(5)
        return_to_center()
        self.iface.sync()
        self.iface.open_grippers()
        self.iface.home()
        self.iface.sync()

    def disambiguate_endpoints(self, focus_area=None, perturb=False):
        """Pulls an endpoint in if 2 aren't visible, otherwise does a Reidemeister move.
        
        Returns: True if termination, False otherwise.
        """
        self.logger.debug(f"endpoints: {self.endpoints}")
        
        color_img = self.img.color._data
        plt.clf()
        plt.imshow(color_img)
        plt.title("Original endpoint before revealing")
        plt.scatter(self.endpoints[:, 1], self.endpoints[:, 0])
        self.show_img()

        self.action_count += 1

        if len(self.endpoints) >= 2:
            self.logger.info(f'Two endpoints detected, so running endpoint separation.')
            try: 
                self.run_endpoint_separation_incremental()
            except Exception as e:
                self.logger.info(e)
                # if reidemeister fails, do an action similar to endpoint reveal
                self.logger.info("CAUGHT REIDEMEISTER ISSUE, falling back on reveal endpoint.")
                result = False
                self.iface.y.reset()
                self.iface.sync()
                self.iface.open_grippers()
                self.iface.home()
                self.iface.sync()
                self.iface.open_arms()
                self.iface.sync()
                self.img = self.take_and_save_image()
                self.get_endpoints()
                
                if focus_area is None:
                    self.iface.reveal_endpoint(self.img, closest_to_pos=self.endpoints[0][::-1])
                else:
                    self.iface.reveal_endpoint(self.img, closest_to_pos=focus_area[::-1])
                self.iface.sync()
            return 

        delta_multiplier = 1.0
        attempts = 0
        while (len(self.endpoints) < 2 and attempts < 3) or (focus_area is not None):
            self.g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
            if focus_area is not None:
                self.logger.info("Doing a disambiguate action on the focus area to bring into frame")
            else:
                self.logger.info('Fewer than two endpoints visible, so revealing endpoints.')


            if perturb:
                try:
                    self.iface.perturb_point(self.img, focus_area[::-1], self.g)
                except:
                    self.iface.y.reset()
                    self.iface.sync()
                    self.iface.open_grippers()
                    self.iface.home()
                    self.iface.sync()
                    self.iface.open_arms()
                    self.iface.sync()
                    self.img = self.take_and_save_image()
                    self.get_endpoints()
                    #focus area is knot exactly valid here
                    self.iface.reveal_endpoint(self.img, closest_to_pos=focus_area[::-1], delta_multiplier=1/2*delta_multiplier)
            else:
                # self.iface.sync()
                # self.iface.open_grippers()
                # self.iface.home()
                self.iface.sync()
                self.iface.open_arms()
                self.iface.sync()
                self.img = self.take_and_save_image()
                self.get_endpoints()
                try:
                    if focus_area is not None:
                        self.iface.reveal_endpoint(self.img, closest_to_pos=focus_area[::-1], delta_multiplier=delta_multiplier)
                    else:
                        self.iface.reveal_endpoint(self.img, delta_multiplier=delta_multiplier)
                except:
                    pass
            delta_multiplier += 0.2
            attempts += 1
            self.iface.sync()

            if focus_area is not None:
                return

    def perception(self):
        '''
        Uses the trace knot detector to determine the knot from either endpoint with highest confidence and gets the 
        cage & pinch for that point. Also determines if we are done untangling, which is when knots are not detected from
        either endpoint.
        Return: cage point, pinch point, pull apart distance, where (if at all) trace left image, and whether we are done
        '''
        print("started perception")
        outputs = []
        trace_uncertain = True
        uncertain_endpoint = None
        for point in self.endpoints:
            starting_pixels, trace_problem = self.get_trace_from_endpoint(point)
            starting_pixels = np.array(starting_pixels)
            if not trace_problem:
                trace_uncertain = False
                color_img = self.img.color._data
                self.tkd._set_data(color_img, starting_pixels)
                perception_result = self.tkd.perception_pipeline(endpoints=self.endpoints, viz=True)
                outputs.append(perception_result)
            else:
                uncertain_endpoint = starting_pixels

        if trace_uncertain and len(self.endpoints) > 0:
            self.disambiguate_endpoints(focus_area=uncertain_endpoint, perturb=True) #IP move to move endpoint
            return -1 #do perception again

        # keep track of retrace knots vs regular knots
        still_knots = []
        retrace_knots = []
        for output in outputs:
            # if len(outputs) <= 1 and output['done_untangling']:
            #     return output
            if output['knot_confidence'] is not None: #check if this change is ok, used to be done_untangling
                if not output['trace_incomplete']:
                    still_knots.append(output)
                elif output['trace_end'] == TraceEnd.RETRACE:
                    retrace_knots.append(output)
                    
        # do a partial if only knot remaining is knot has trace incompelte. Edit pull apart to keep x constant.
        # if no regular knots left, then return retrace knot

        if len(still_knots) > 0: #there is a knot
            confidence = -float('inf')
            selected = None
            for output in still_knots:
                if output['knot_confidence'] > confidence and output['cage'] is not None and output['pinch'] is not None:
                    confidence = output['knot_confidence']
                    selected = output
            cage = self.closest_valid_point(self.img.color._data, self.img.depth._data, selected['cage'])
            pinch = self.closest_valid_point(self.img.color._data, self.img.depth._data, selected['pinch'])
            selected['cage'] = cage
            selected['pinch'] = pinch
            return selected
            # output contains the following keys: cage, pinch, knot_confidence, pull_apart_dist, done_untangling, trace_end, reveal_point
        else:
            if len(retrace_knots) > 0:
                confidence = -float('inf')
                selected = None
                for output in retrace_knots:
                    if output['knot_confidence'] > confidence and output['cage'] is not None and output['pinch'] is not None:
                        confidence = output['knot_confidence']
                        selected = output
                cage = self.closest_valid_point(self.img.color._data, self.img.depth._data, selected['cage'])
                pinch = self.closest_valid_point(self.img.color._data, self.img.depth._data, selected['pinch'])
                selected['cage'] = cage
                selected['pinch'] = pinch
                return selected
            else:
                # what to do if no knots
                print("outputs: ", outputs)
                for output in outputs:
                    if output['trace_end'] == TraceEnd.EDGE:
                        print("Realized trace left image")
                        output['reveal_point'] = self.closest_valid_point(self.img.color._data, self.img.depth._data, output['reveal_point'])
                        return output
                    if output['done_untangling']:
                        return output
                if len(outputs) <= 0:
                    return -1
                selected = outputs[0]
                cage = self.closest_valid_point(self.img.color._data, self.img.depth._data, selected['cage']) if selected['cage'] is not None else None
                pinch = self.closest_valid_point(self.img.color._data, self.img.depth._data, selected['pinch']) if selected['pinch'] is not None else None
                selected['cage'] = cage
                selected['pinch'] = pinch
                return selected

    def perform_cage_pinch(self, cage, pinch, pull_apart_dist):
        self.print_and_log(self.action_count, int(time.time()), "Doing a cage-pinch dilation move...")

        plt.clf()
        plt.imshow(self.img.color._data)
        plt.title("Cage-Pinch In-Place Pull Apart points")
        plt.scatter([cage[1]], [cage[0]], color='red', label='cage')
        plt.scatter([pinch[1]], [pinch[0]], color='blue', label='pinch')
        plt.legend()
        self.show_img()

        self.g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
        try:
            l_grasp, r_grasp = self.g.double_grasp(
                tuple(cage[::-1]), tuple(pinch[::-1]), .0085, .0085, self.iface.L_TCP, self.iface.R_TCP, slide0=True, slide1=False)
        except:
            print("Couldn't plan a double grasp")
            self.iface.reveal_endpoint(self.img, closest_to_pos=((np.array(cage) + np.array(pinch))/2)[::-1])
            self.action_count += 1
            return

        l_grasp.pregrasp_dist = 0.05
        r_grasp.pregrasp_dist = 0.05

        s1, s2 = l_grasp.slide, r_grasp.slide

        logging.info("Homing arms.")
        self.iface.home()
        self.iface.sync()
        logging.info(f"Left arm slide: {s1}, Right arm slide: {s2}")

        self.iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp, topdown=self.topdown_grasps, bend_elbow=False)

        try:
            if pull_apart_dist <= 0.05:
                self.iface.partial_pull_apart(pull_apart_dist, slide_left=s1, slide_right=s2, layout_nicely=False) #, return_to_center=True)
            else:
                self.iface.partial_pull_apart(pull_apart_dist, slide_left=s1, slide_right=s2, layout_nicely=True)
            self.iface.sync()
        except:
            if pull_apart_dist <= 0.05:
                pass
            else:
                self.iface.y.reset()
                logging.info("FAILED PULL APART ONCE, trying again")
                #bring arms in first before pulling again
                self.iface.go_delta(l_trans=[0, -0.05, 0], r_trans=[0, 0.05, 0])
                self.iface.sync()
                self.iface.shake_R('right', rot=[0, 0, 0], num_shakes=4, trans=[0, 0, .05])
                self.iface.sync()
                self.iface.shake_R('left', rot=[0, 0, 0], num_shakes=4, trans=[0, 0, .05])
                self.iface.sync()
                self.iface.partial_pull_apart(pull_apart_dist + 0.02, slide_left=s1, slide_right=s2, layout_nicely=True) #, return_to_center=True)
                self.iface.sync()
        self.iface.open_arms()
        self.action_count += 1

    def run_pipeline(self):
        self.start_time = int(time.time())
        self.print_and_log(f'START TIME: {self.start_time}\n')
        done = False
        failure = False
        first = True
        count = 0
        diff = int(time.time()) - self.start_time
        while not done and diff < self.time_limit:
            try:
                self.iface.open_grippers()
                self.iface.open_arms()
                time.sleep(1)
                self.img = self.take_and_save_image(first=first)

                plt.clf()
                plt.imshow(self.img.color._data)
                plt.title("Initial Image")
                plt.savefig(self.img_folder + "_" + "init.png")

                # self.get_endpoints(self.img)
                self.get_endpoints()

                if self.endpoints.shape[0] == 0:
                    failure = True

                if failure:
                    self.print_and_log(self.action_count, int(time.time()), "Disambiguation move due to previous failure or no endpoints detected...")
                    failure = False
                    self.disambiguate_endpoints()
                    continue

                output = self.perception()
                if output == -1:
                    continue

                pinch, cage = output['pinch'], output['cage']
                done_temp = output['done_untangling']
                reveal_point = output['reveal_point']
                pull_apart_dist = output['pull_apart_dist']

                print("done: ", done_temp)

                if done_temp and not first:
                    count += 1
                    if count >= 1:
                        done = True
                        continue
                first = False
                # else:
                #     if count > 0:
                #         count -= 1
                #         done = False

                print("reveal_point: ", reveal_point)

                if cage is not None and pinch is not None:
                    if output['trace_incomplete']:
                        print("doing partial cage pinch bc trace incomplete")
                        pull_apart_dist = 0.045
                    self.perform_cage_pinch(cage, pinch, pull_apart_dist)
                else:
                    self.disambiguate_endpoints(reveal_point) #doesn't matter if its none or not, will do some action


                diff = int(time.time()) - self.start_time
            except Exception as e:
                failure = True
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
                    self.iface.y.reset()
                else:
                    # raise e
                    self.logger.info("Uncaught exception, still recovering " + str(e))
                    self.iface.y.reset()
                self.iface.sync()
        
        if (time.time() > self.start_time + self.time_limit):
            self.print_and_log(f"Timed out at time {int(time.time())} after duration {int(time.time()) - self.start_time}.")
        else:
            self.print_and_log(f"Done at time {int(time.time())} after duration {int(time.time()) - self.start_time}.")
        # close logs file
        self.logs_file.close()
        exit()

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
    fullPipeline = FullPipeline(viz=False, loglevel=logLevel)
    fullPipeline.run_pipeline()
