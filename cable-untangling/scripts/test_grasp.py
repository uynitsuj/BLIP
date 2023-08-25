from untangling.utils.grasp import GraspSelector, GraspException
from untangling.utils.interface_rws import Interface
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
import numpy as np
from untangling.utils.tcps import *
import time
from untangling.point_picking import *
import matplotlib.pyplot as plt
from untangling.keypoint_untangling import closest_valid_point_to_pt
import torch
import cv2
import threading
from queue import Queue
import signal
import os
import os.path as osp
import datetime
# for the last test:
from phoxipy.phoxi_sensor import PhoXiSensor
from datetime import datetime

from full_pipeline import run_endpoint_separation_incremental
#from fcvision.kp_wrapper import KeypointNetwork

torch.cuda.empty_cache()

def l_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(
        translation=trans,
        rotation=rot,
        from_frame=YK.l_tcp_frame,
        to_frame=YK.base_frame,
    )

def r_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(
        translation=trans,
        rotation=rot,
        from_frame=YK.r_tcp_frame,
        to_frame=YK.base_frame,
    )


def print_and_log(file, *args):
    print(*args)
    file.write(' '.join(map(str, args)) + '\n')
    file.flush()

def get_handler(q):
    # Handles signals, e.g. sigint to close camera streams.
    def signal_handler(sig, frame):
        global t2, _FINISHED
        _FINISHED = True
        t2.join()
        # q.put('stop')
        # while q.qsize() > 0:
        #     time.sleep(0.1) # some time for camera streams to close. TODO: make cleaner
        sys.exit(0)
    return signal_handler


def take_video(vfname):
    # Create a VideoCapture object
    print(vfname)
    cap = cv2.VideoCapture(14)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")
        return

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # print("here1")

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(vfname,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    start_time = time.time()

    while(not _FINISHED):
        ret, frame = cap.read()

        if ret == True:
            # Write the frame into the file 'output.avi'
            out.write(frame)

        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()


def run_pipeline():
    for i in range(10): print("TEST")
    global t2, _FINISHED
    vid_queue = Queue()
    start_time = int(time.time())
    # signal.signal(signal.SIGINT, get_handler(vid_queue))
#    t2.start()
#    signal.signal(signal.SIGQUIT, get_handler(vid_queue))

    # format as date time
    # date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SPEED = (0.6, 2 * np.pi) #(0.6, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    
    iface.open_grippers()
    while True:
        try:
            for i in range(10): print("entered")
            iface.open_grippers()
            iface.open_arms()
            # 1. try to grab an endpoint, if not possible, fallback to a shake action
            time.sleep(1)
            img = iface.take_image()
            for i in range(10): print("entered2")                                                   
            # 5. compute grasps based on HULK keypoints
            g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
            left_coords, right_coords = click_points(img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
            left_coords = closest_valid_point_to_pt(img.depth._data, left_coords, color=img.color._data)
            right_coords = closest_valid_point_to_pt(img.depth._data, right_coords, color=img.color._data)

            # grasp, arm = g.single_grasp_closer_arm(
            #     left_coords, .003, iface.tcp_dict)
            # iface.grasp_single(arm, grasp)
            for i in range(10): print("entered3")                                                   

            grasp_l, grasp_r = g.double_grasp(left_coords, right_coords, 0.014, 0.014, l_tcp=iface.L_TCP, r_tcp=iface.R_TCP)
            iface.grasp(l_grasp=grasp_l, r_grasp=grasp_r, topdown=False, bend_elbow=False)
            iface.sync()
            for i in range(10): print("entered4")                                                   

            iface.go_delta(l_trans=[0, 0, .1], r_trans=[0, 0, .1])
            # 7. execute grasps and CG+CG pull apart motion
            iface.sync()
            iface.open_grippers()
            iface.open_arms()
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
                print("Caught error, recovering", str(e))
            else:
                print("Uncaught exception, still recovering " + str(e))
                # print traceback of exception
                # raise e # re-raise
            failure = True
            iface.y.reset()

def test_partial_pull_apart():
    SPEED = (0.6, 2 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    iface.open_grippers()
    iface.open_arms()
    time.sleep(3)
    iface.sync()
    iface.close_grippers()
    iface.sync()
    img = iface.take_image()
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    left_coords, right_coords = click_points(img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
    if left_coords is not None:
        # print_and_log(logs_file, action_count, int(time.time()), "Doing an untangling move...")
        l_grasp, r_grasp = g.double_grasp(
            left_coords, right_coords, .007, .007, iface.L_TCP, iface.R_TCP)
        iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp)
        # iface.go_delta(l_trans=[0, 0, .05], r_trans=[0, 0, .05])
        # iface.go_delta(l_trans=[0, 0, -.03],
        #             r_trans=[0, 0, -.03], reltool=True)
        iface.sync()
        iface.partial_pull_apart(1,tilt=True)
        iface.sync()

def test_endpoint_separation():
    
    SPEED = (0.6, 2 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    iface.open_grippers()
    iface.sync()
    iface.open_arms()
    iface.sync()
    iface.close_grippers()
    iface.sync()
    img = iface.take_image()
    # keypoint_model = KeypointNetwork(
    #     "~/brijenfc-vision/models/endpoint_fcn.ckpt", params={'task': 'cable_endpoints'})
    # iface.open_grippers()
    # iface.open_arms()
    # time.sleep(1)
    # img = iface.take_image()
    endpoints = np.array(click_points(img, iface.T_PHOXI_BASE, iface.cam.intrinsics))
     #pseudo_endpoints = get_endpoints(keypoint_model, img)
    endpoints = [(endpoint[::-1], endpoint) for endpoint in endpoints]

    #endpoints_tuples = [tuple(l[0].tolist())[:s:-1] for l in endpoints]
    # 
    # (
    # ((402, 255)
    # (402, 255)),  
    # (608, 163)
    # (608, 163)
    # )
    # n * 2 * 2
    # 
    #if len(endpoints) == 2 and not just_slid: # should I really take out this condition?
    print(endpoints)
    run_endpoint_separation_incremental(endpoints, img, iface)
    #run_endpoint_separation_experimental(endpoints, img, iface)
    
def test_rotation():
    SPEED = (0.6, 2 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    iface.open_grippers()
    iface.open_arms()
    time.sleep(3)
    # iface.sync()
    # iface.close_grippers()
    iface.sync()
    img = iface.take_image()
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    left_coords, right_coords = click_points(img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
    if left_coords is not None:
        # print_and_log(logs_file, action_count, int(time.time()), "Doing an untangling move...")
        l_grasp, r_grasp = g.double_grasp(
            left_coords, right_coords, .007, .007, iface.L_TCP, iface.R_TCP)
        iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp)
        # iface.go_delta(l_trans=[0, 0, .05], r_trans=[0, 0, .05])
        # iface.go_delta(l_trans=[0, 0, -.03],
        #             r_trans=[0, 0, -.03], reltool=True)
        iface.sync()
        iface.rotate_to_visible(np.pi/2)
        iface.sync()
    iface.open_grippers()
    
# def test_incremental_endpoint_separation():
#     SPEED = (0.6, 2 * np.pi)
#     iface = Interface(
#         "1703005",
#         ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
#         ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
#         speed=SPEED,
#     )
#     iface.open_grippers()
#     iface.open_arms()
#     iface.sync()
#     img = iface.take_image()
#     endpoints = click_points(img, iface.T_PHOXI_BASE, iface.cam.intrinsics) 
#     print(endpoints)
#     run_endpoint_separation_incremental(endpoints, img, iface,thresh=0.99, increment = 0.1)
        

def test_pull_apart():
    SPEED = (0.6, 2 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    iface.open_arms()
    iface.open_grippers()
    time.sleep(3)
    iface.sync()
    iface.close_grippers()
    iface.sync()
    img = iface.take_image()
    iface.sync()
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    iface.partial_pull_apart(0.4, g, 0.5, 0.5)
    iface.sync()
    
    
def test_detectron_camera():
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
    )
    # depth_dir = "temp_collection/depth_images/"
    # color_dir = "temp_collection/color_images/"
    # output_dir = "temp_collection/detectron_output/"
    i = 15 # CHANGE THIS
    # while True:
    cam_only = True
    for _ in range(100):
        start = datetime.now()
        img = iface.take_image(cam_only)
        cam_only = not cam_only
        #import pdb;pdb.set_trace()
        end = datetime.now()
        print(end - start)
        plt.imshow(img.color._data)
        plt.show()    

if __name__ == "__main__":
    # test_incremental_endpoint_separation()
    run_pipeline()
