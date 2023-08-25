# 3 lines of detectron imports
from pickle import FALSE
from turtle import done, left
import analysis as loop_detectron
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
import os
import datetime
import logging 
import argparse
import matplotlib.pyplot as plt
import shutil
from untangling.tracer_knot_detect.tracer_knot_detection import TracerKnotDetector
from untangling.tracer_knot_detect.tracer import TraceEnd

plt.set_loglevel('info')
torch.cuda.empty_cache()

class PerceptionExp():
    def __init__(self, results_folder, og_config_folders):
        self.idx = 0
        self.results_folder = results_folder
        self.og_config_folders = og_config_folders
        if os.path.exists(self.results_folder):
            shutil.rmtree(self.results_folder)
        os.mkdir(self.results_folder)
        # os.mkdir(os.path.join(self.results_folder, 'sgtm2'))
        # os.mkdir(os.path.join(self.results_folder, 'superman'))
        # os.mkdir(os.path.join(self.results_folder, 'superman_analytic_trace'))
        experiment_types = ['sgtm2', 'superman', 'superman_analytic_trace', 'superman_wo_cc']
        for folder in experiment_types:
            exp_folder = os.path.join(self.results_folder, folder)
            os.mkdir(exp_folder)
            for config in og_config_folders:
                os.mkdir(os.path.join(exp_folder, config))
        self.config = 0
            
        self.tkd = TracerKnotDetector()
        
    def sgtm2(self, img):
        color_img = (img[:,:,:3]).astype(np.uint8)
        _, viz = loop_detectron.predict(color_img, thresh=0.99)
        cv2.imwrite(os.path.join(self.results_folder, 'sgtm2', self.og_config_folders[self.config], str(self.idx) + '.png'), viz)
    
    def get_endpoints(self, img):
        # model not used, already specified in loop_detectron
        color_img = (img[:,:,:3]).astype(np.uint8)
        depth_img = img[:,:,3:]
        endpoint_boxes, out_viz = loop_detectron.predict(color_img, thresh=0.99, endpoints=True)
        endpoints = []
        for box in endpoint_boxes:
            xmin, ymin, xmax, ymax = box
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2
            new_yx = self.closest_valid_point(color_img, depth_img, np.array([y,x]))
            endpoints.append([new_yx, new_yx])   
        endpoints = np.array(endpoints).astype(np.int32)
        endpoints = np.array([e for e in endpoints if depth_img[e[0][0], e[0][1]] > 0])
        endpoints = endpoints.astype(np.int32).reshape(-1, 2, 2)
        return endpoints[:, 0, :]
    
    def get_trace_from_endpoint(self, img, point):
        color_img = img[:,:,:3]
        depth_img = img[:,:,3:]
        # trace from endpoint on image
        img = np.concatenate((color_img, depth_img), axis=-1)
        img[-130:, ...] = 0
        thresh_img = np.where(img[:,:,:3] > 100, 255, 0)

        full_img = np.concatenate((thresh_img, img[:,:,3:]), axis=-1)
        path, finished_paths = trace(full_img, point, None, exact_path_len=4, viz=False)
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
    
    def closest_valid_point(self, color, depth, yx):
        valid_y, valid_x = np.nonzero((color[:, :, 0] > 100) * (depth[:, :, 0] > 0))
        pts = np.vstack((valid_y, valid_x)).T
        return pts[np.argmin(np.linalg.norm(pts - np.array(yx)[None, :], axis=-1))]
    
    def superman(self, img):
        color_img = (img[:,:,:3]).astype(np.uint8)
        depth_img = img[:,:,3:]
        endpoints = self.get_endpoints(img)
        knot_confidence = -float('inf')
        higher = -1
        for i, point in enumerate(endpoints):
            starting_pixels, trace_problem = self.get_trace_from_endpoint(img, point)
            starting_pixels = np.array(starting_pixels)
            if not trace_problem:
                self.tkd._set_data(color_img, starting_pixels)
                self.tkd.trace_and_detect_knot(endpoints=endpoints)
                if self.tkd.knot:
                    conf = self.tkd._get_knot_confidence()
                    if conf > knot_confidence:
                        higher = i
                    
        starting_pixels, trace_problem = self.get_trace_from_endpoint(img, endpoints[higher])
        starting_pixels = np.array(starting_pixels)
        if not trace_problem:
            self.tkd._set_data(color_img, starting_pixels)
            self.tkd.trace_and_detect_knot(endpoints=endpoints)          
            trace_save_path = os.path.join(self.results_folder, 'superman', self.og_config_folders[self.config], str(self.idx) + '_trace' + str(i) + '_.png')
            knot_save_path = os.path.join(self.results_folder, 'superman', self.og_config_folders[self.config], str(self.idx) + '_knot' + str(i) + '_.png')
            self.tkd._visualize_full(path=trace_save_path)
            if self.tkd.knot:
                self.tkd._visualize_knot(knot_save_path)
                    
    def superman_wo_cc(self, img):
        color_img = (img[:,:,:3]).astype(np.uint8)
        depth_img = img[:,:,3:]
        endpoints = self.get_endpoints(img)
        knot_confidence = -float('inf')
        higher = -1
        for i, point in enumerate(endpoints):
            starting_pixels, trace_problem = self.get_trace_from_endpoint(img, point)
            starting_pixels = np.array(starting_pixels)
            if not trace_problem:
                self.tkd._set_data(color_img, starting_pixels)
                self.tkd.trace_and_detect_knot(endpoints=endpoints, cancel=False)
                if self.tkd.knot:
                    conf = self.tkd._get_knot_confidence()
                    if conf > knot_confidence:
                        higher = i
                    
        starting_pixels, trace_problem = self.get_trace_from_endpoint(img, endpoints[higher])
        starting_pixels = np.array(starting_pixels)
        if not trace_problem:
            self.tkd._set_data(color_img, starting_pixels)
            self.tkd.trace_and_detect_knot(endpoints=endpoints, cancel=False)
            trace_save_path = os.path.join(self.results_folder, 'superman_wo_cc', self.og_config_folders[self.config], str(self.idx) + '_trace' + str(higher) + '_.png')
            knot_save_path = os.path.join(self.results_folder, 'superman_wo_cc', self.og_config_folders[self.config], str(self.idx) + '_knot' + str(higher) + '_.png')
            self.tkd._visualize_full(path=trace_save_path)
            if self.tkd.knot:
                self.tkd._visualize_knot(knot_save_path)
        
    def superman_analytic_trace(self, img):
        self.tkd = TracerKnotDetector(analytic=True)
        color_img = (img[:,:,:3]).astype(np.uint8)
        depth_img = img[:,:,3:]
        endpoints = self.get_endpoints(img)
        knot_confidence = -float('inf')
        higher = -1
        for i, point in enumerate(endpoints):
            starting_pixels, trace_problem = self.get_trace_from_endpoint(img, point)
            starting_pixels = np.array(starting_pixels)
            if not trace_problem:
                self.tkd._set_data(color_img, starting_pixels)
                self.tkd.trace_and_detect_knot(endpoints=endpoints)
                if self.tkd.knot:
                    conf = self.tkd._get_knot_confidence()
                    if conf > knot_confidence:
                        higher = i

        starting_pixels, trace_problem = self.get_trace_from_endpoint(img, endpoints[higher])
        starting_pixels = np.array(starting_pixels)
        if not trace_problem:
            self.tkd._set_data(color_img, starting_pixels)
            self.tkd.trace_and_detect_knot(endpoints=endpoints)          
            trace_save_path = os.path.join(self.results_folder, 'superman_analytic_trace', self.og_config_folders[self.config], str(self.idx) + '_trace' + str(i) + '_.png')
            knot_save_path = os.path.join(self.results_folder, 'superman_analytic_trace', self.og_config_folders[self.config], str(self.idx) + '_knot' + str(i) + '_.png')
            self.tkd._visualize_full(path=trace_save_path)
            if self.tkd.knot:
                self.tkd._visualize_knot(knot_save_path)
    
    def run_exp(self, img, config=0):
        print("IMG:", self.idx)
        self.config= config
        # self.sgtm2(img)
        # self.superman(img)
        # self.superman_wo_cc(img)
        self.superman_analytic_trace(img)
        self.idx += 1
        
    def reset_idx(self):
        self.idx = 0
    
if __name__ == "__main__":
    curr_path = '/home/justin/yumi/cable-untangling/scripts/'
    og_config_folders = ['config1', 'config2', 'config3']
    # og_config_folders = ['config2']
    config_folders = [os.path.join(curr_path, folder) for folder in og_config_folders]
    perception_exp = PerceptionExp(os.path.join(curr_path, 'perception_results3'), og_config_folders)

    avg_times = []
    for j,folder in enumerate(config_folders): 
        perception_exp.reset_idx()
        curr_config_times = []
        for i in range(len(os.listdir(folder))//2):
            color_path = os.path.join(folder, 'color_' + str(i) + '.npy')
            depth_path = os.path.join(folder, 'depth_' + str(i) + '.npy')
            color_img = np.load(color_path)
            depth_img = np.load(depth_path)
            img = np.concatenate((color_img, depth_img), axis=2)
            start_time = time.time()
            perception_exp.run_exp(img, config=j)
            curr_config_times.append(time.time() - start_time)
        avg_times.append(np.mean(curr_config_times))
    print("times for each config", avg_times)