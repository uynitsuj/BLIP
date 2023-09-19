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
import pickle as pkl
import shutil
plt.set_loglevel('info')
# plt.style.use('seaborn-darkgrid')
from untangling.tracer_knot_detect.tracer_knot_detection import TracerKnotDetector
from untangling.tracer_knot_detect.tracer import TraceEnd
from annot_spline import KeypointsAnnotator
from phoxipy.phoxi_sensor import PhoXiSensor

class MultiCableTracer():
    def __init__(self, folder_name, analytic=False):
        SPEED = (0.6, 2 * np.pi) #(0.6, 6 * np.pi)
        self.iface = Interface(
            "1703005",
            ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
            ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
            speed=SPEED,
        )
        self.iface.close_grippers()
        self.iface.sync()
        self.iface.open_arms()
        self.iface.sync()
        self.total_obs = 0
        self.image_save_path = '/home/justin/yumi/cable-untangling/scripts/' + folder_name
        if not os.path.exists(self.image_save_path):
            os.mkdir(self.image_save_path)
            self.file_idx = 0
        else:
            self.file_idx = len(os.listdir(self.image_save_path))//2 +1
        self.idx = 0
        self.annotator = KeypointsAnnotator()
        self.viz = False
        self.alg_output_fig = 'alg_output.png'
        self.tkd = TracerKnotDetector(analytic=analytic)
        # self.adapter_locs = np.array([[630, 490], [633, 561], [637, 624]])
        # self.adapter_locs = np.array([[621,530], [635, 589], [640, 658]])
        self.adapter_locs = np.array([[626, 544], [630, 608], [635, 672]])

        self.T_PHOXI_BASE = RigidTransform.load(
        "/home/justin/yumi/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")
        self.cam = PhoXiSensor("1703005")
        self.cam.start()

    def get_trace_from_endpoint(self, point, img):
        # trace from endpoint on image
        img = np.concatenate((img[:,:, :3], img[:, :, 3:]), axis=-1)
        img[-130:, ...] = 0
        thresh_img = np.where(img[:,:,:3] > 100, 255, 0)

        full_img = np.concatenate((thresh_img, img[:,:,3:]), axis=-1)
        self.viz = True
        path, finished_paths = trace(full_img, point, None, exact_path_len=4, viz=self.viz)
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

    def set_adapter_locs(self):
        self.adapter_locs = []
        img = self.cam.read()
        for i in range(3):
            print("Adapter " + str(i) + ":")
            intr= PhoXiSensor.create_intr(img.width, img.height)
            pt,_ = click_points(img, self.T_PHOXI_BASE, intr)
            pt = pt[::-1]
            print("point: ", pt)
            self.adapter_locs.append(pt)

    def trace(self):
        '''
        Uses the trace knot detector to determine the knot from either endpoint with highest confidence and gets the 
        cage & pinch for that point. Also determines if we are done untangling, which is when knots are not detected from
        either endpoint.
        Return: cage point, pinch point, pull apart distance, where (if at all) trace left image, and whether we are done
        '''
        imgs = []
        endpoints = []
        paths = []
        pt = None
        folder = "folder_" + str(self.file_idx)
        folder_path = os.path.join(self.image_save_path, folder)
        while os.path.exists(folder_path):
            self.file_idx += 1
            folder = "folder_" + str(self.file_idx)
            folder_path = os.path.join(self.image_save_path, folder)
        print("file number: ", self.file_idx)
        os.mkdir(folder_path)
        for i in range(1):
            trace_problem = True
            while trace_problem:
                img = self.cam.read()
                intr= PhoXiSensor.create_intr(img.width, img.height)
                if pt == None:
                    pt,_ = click_points(img, self.T_PHOXI_BASE, intr)
                    pt = pt[::-1]
                    # other_endpoint,_ = click_points(img, self.T_PHOXI_BASE, intr)
                    # other_endpoint = np.array([other_endpoint[::-1]])
                starting_pixels, trace_problem = self.get_trace_from_endpoint(pt, img.data)
                starting_pixels = np.array(starting_pixels)
                if trace_problem:
                    print("trace issue")
                    manual_img = np.array(img.data[:,:,:3], dtype=np.uint8)
                    print("manual img: ", manual_img)
                    _, starting_pixels = self.annotator.run(manual_img)
                    starting_pixels = np.squeeze(np.array(starting_pixels))
                    starting_pixels = starting_pixels[:, ::-1]
                    print("starting pixels: ", starting_pixels)
                color_img = img.color._data
                self.tkd._set_data(color_img, starting_pixels)
                self.tkd.trace_and_detect_knot(trace_only=True, endpoints=self.adapter_locs)
                dist_to_adapters = np.linalg.norm(self.adapter_locs - np.expand_dims(self.tkd.pixels[-1], axis=0), axis=-1)
                # if any([dist < 20 for dist in dist_to_adapters]):
                path_save = os.path.join(folder_path, str(self.idx) + '.png')
                original_path_save = os.path.join(folder_path, str(self.idx) + '_input.png')
                self.tkd._visualize_full(path_save)
                cv2.imwrite(original_path_save, color_img)
                self.idx += 1
                imgs.append(img.data)
                endpoints.append(pt)
                # endpoints.append(other_endpoint[0])
                paths.append(self.tkd.pixels)
        dictionary = {'imgs': np.array(imgs), 'endpoints': np.array(endpoints), 'paths': np.array(paths), 'adapter_locs': np.array(self.adapter_locs)}
        file_path = os.path.join(self.image_save_path, "data_" + str(self.file_idx) + ".pkl")
        pkl.dump(dictionary, open(file_path, "wb"))
        self.file_idx += 1

    def trace_w_input(self, data):
        '''
        Uses the trace knot detector to determine the knot from either endpoint with highest confidence and gets the 
        cage & pinch for that point. Also determines if we are done untangling, which is when knots are not detected from
        either endpoint.
        Return: cage point, pinch point, pull apart distance, where (if at all) trace left image, and whether we are done
        '''
        imgs = data['imgs']
        endpoints = data['endpoints']
        paths = data['paths']
        adapter_locs = data['adapter_locs']
        # if self.file_idx >= 18:
        #     adapter_locs = [loc[::-1] for loc in adapter_locs]
        # print(adapter_locs)
        pt = None
        folder = "folder_" + str(self.file_idx)
        print("file number: ", self.file_idx)
        folder_path = os.path.join(self.image_save_path, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)
        path_save = os.path.join(folder_path, str(self.idx) + '_og.png')
        cv2.imwrite(path_save, imgs[0][:, :, :3])

        for i in range(3):
            trace_problem = True
            while trace_problem:
                img = imgs[i]
                pt = endpoints[i]
                starting_pixels, trace_problem = self.get_trace_from_endpoint(pt, img)
                starting_pixels = np.array(starting_pixels)
                if trace_problem:
                    print("trace issue")
                    continue
                color_img = img[:, :, :3]
                self.tkd._set_data(color_img, starting_pixels)
                self.tkd.trace_and_detect_knot(trace_only=True, endpoints=adapter_locs)
                dist_to_adapters = np.linalg.norm(adapter_locs - np.expand_dims(self.tkd.pixels[-1], axis=0), axis=-1)
                if any([dist < 20 for dist in dist_to_adapters]):
                    path_save = os.path.join(folder_path, str(self.idx) + '_done.png')
                    self.tkd._visualize_full(path_save)
                    self.idx += 1
                else:
                    path_save = os.path.join(folder_path, str(self.idx) + '.png')
                    self.tkd._visualize_full(path_save)
                    self.idx += 1
        self.file_idx += 1

    def trace_on_saved_img_pick_points(self, data):
        imgs = []
        endpoints = []
        paths = []
        pt = None
        adapter_locs = data['adapter_locs']
        folder = "folder_" + str(self.file_idx)
        print("file number: ", self.file_idx)
        folder_path = os.path.join(self.image_save_path, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)

        count = 0
        for j in range(3):
            trace_problem = True
            pt = None
            for i in range(1):
                trace_problem = True
                while trace_problem:
                    img = data['imgs'][i]
                    img_no_depth_scaled = img[:,:,:3]/255
                    if pt == None:
                        pt = self.click_points_2d_img(img_no_depth_scaled)
                        pt = pt[::-1]
                        print("got to pt")
                    starting_pixels, trace_problem = self.get_trace_from_endpoint(pt, img)
                    starting_pixels = np.array(starting_pixels)
                    if trace_problem:
                        print("trace issue")
                        continue
                    print("starting pixels", starting_pixels)
                    color_img = img[:,:,:3]
                    self.tkd._set_data(color_img, starting_pixels)
                    self.tkd.trace_and_detect_knot(trace_only=True, endpoints=adapter_locs)
                   
                    paths.append(self.tkd.pixels)
                
                    dist_to_adapters = np.linalg.norm(self.adapter_locs - np.expand_dims(self.tkd.pixels[-1], axis=0), axis=-1)
                    # if any([dist < 20 for dist in dist_to_adapters]):
                    #     print("SAVING WRONG")
                    #     path_save = "/home/justin/yumi/cable-untangling/scripts/multi_cable_traces/" + "trace_fail_" + str(i) + ".png"
                    #     self.tkd._visualize_full(path_save)
                    print("SAVING")
                    count += 1
                    path_save = "/home/justin/yumi/cable-untangling/scripts/multi_cable_traces/tier1/" + "trace_" + str(count) + ".png"
                    self.tkd._visualize_single_color_paths(paths, path_save)
                    self.idx += 1
                    imgs.append(img.data)
                    endpoints.append(pt)

            
        dictionary = {'imgs': np.array(imgs), 'endpoints': np.array(endpoints), 'paths': np.array(paths), 'adapter_locs': np.array(self.adapter_locs)}
        file_path = os.path.join(self.image_save_path, "data_" + str(self.file_idx) + ".pkl")
        pkl.dump(dictionary, open(file_path, "wb"))
        self.file_idx += 1

    
    def click_points_2d_img(self, img):
        # fig, (ax1,ax2) = plt.subplots(1,2)
        # ax1.imshow(img.color.data)
        # ax2.imshow(img.depth.data)
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(img)
  
        left_coords,right_coords = None,None
        def onclick(event):
            xind,yind = int(event.xdata),int(event.ydata)
            coords=(xind,yind)

            nonlocal left_coords,right_coords
            if(event.button==1):
                print("in here")
                left_coords=coords

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        return left_coords



if __name__ == '__main__':
    folder_name = 'demo'
    # save_folder_name = 'tracer_analytic_exp_RSS2023_r1'
    multi_tracer = MultiCableTracer(folder_name, analytic=False)
    while True:
        # multi_tracer.set_adapter_locs()
        multi_tracer.trace()
        user = input("next?y/n")
    # multi_tracer.set_adapter_locs()
    # for _ in range(27 - multi_tracer.file_idx):
    #     multi_tracer.trace()
    #     user = input("next?y/n")
    # load from tier 1, data_9.pkl
    # folder = '/home/justin/yumi/cable-untangling/scripts/' + folder_name
    # # for i in range(27):
    # #     pickle_file = os.path.join(folder, 'data_' + str(i) + '.pkl')
    # #     data = pkl.load(open(pickle_file, "rb"))
    # #     multi_tracer.trace_w_input(data)
    # folder = '/home/justin/yumi/cable-untangling/scripts/' + 'tracer_exp_RSS2023_tier1'
    # pickle_file = os.path.join(folder, 'data_' + str(0) + '.pkl')
    # data = pkl.load(open(pickle_file, "rb"))

    # multi_tracer.trace_on_saved_img_pick_points(data)
 



        
