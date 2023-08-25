import os
from socket import timeout
import subprocess
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
from autolab_core import Box, ColorImage, DepthImage, RgbdImage, RigidTransform,Point
from fcvision.kp_wrapper import KeypointNetwork
from phoxipy.phoxi_sensor import PhoXiSensor

from untangling.spool import execute_spool, init_endpoint_orientation
from untangling.utils.cable_tracer import CableTracer
from untangling.utils.circle_BFS import trace
from untangling.utils.grasp import GraspException, GraspSelector
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import cv2
import open3d
import torch
from untangling.endpoint_classifier.endpoint_classifier import EndpointClassifier
from fcvision.kp_wrapper import KeypointNetwork
from fcvision.vision_utils import closest_nonzero_pt
from untangling.utils.workspace import left_of_workspace, WORKSPACE_CENTER
import logging
# from scripts.collect_images import move_to_slide_pos

slide_pos = os.path.dirname(os.path.abspath(__file__)) + "/../scripts"
sys.path.insert(0,slide_pos)
from collect_images import move_to_slide_pos

class NetworkStopCond:
    def __init__(self,model_path,vis=True):
        self.model=EndpointClassifier(pretrained=False, channels=2, num_classes=2, img_height=200, img_width=200, dropout=False).cpu()
        self.model.load_state_dict(torch.load(model_path))
        self.ret_dict={0:"endpoint", 1:"not_endpoint"}
        self.vis=vis
        self.T_CAM_BASE = RigidTransform.load("/home/justin/yumi/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")
        self.T_BASE_CAM = self.T_CAM_BASE.inverse()
        self.conf_threshold=.95
        
    def prepare_image(self,img:RgbdImage,right_pos:RigidTransform):
        img_color = img.color._data
        depth_og = img.depth.data
        p = Point(right_pos.translation,frame=YK.base_frame)
        
        intr = PhoXiSensor.create_intr(img_color.shape[1], img_color.shape[0])
        right_grip_pixel_coord = intr.project(self.T_BASE_CAM*p)
        right_x = right_grip_pixel_coord[0]
        right_y = right_grip_pixel_coord[1]
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        depth_og = np.squeeze(depth_og)
        thresh = 0.7
        gray[depth_og>thresh]=0
        gray[depth_og==0]=0
        img = gray
        depth_og[depth_og>thresh] = 0
        depth = depth_og

        img[int(4*img.shape[0]/5):, :] = 0
        depth[int(4*depth.shape[0]/5):, :] = 0
        img_crop = img[right_y-220:right_y-20, right_x-80: right_x+120]/255.
        depth_crop = depth[right_y-220:right_y-20, right_x-80: right_x+120]
        depth_crop = np.resize(depth_crop, (depth_crop.shape[1], depth_crop.shape[0]))
        combined = np.stack((img_crop, depth_crop)).astype(np.float32) 
        return combined

    def query(self,img:RgbdImage,right_gripper_pos:RigidTransform):
        cropped_img = self.prepare_image(img,right_gripper_pos)
        with torch.no_grad():
            result = self.model(torch.from_numpy(cropped_img).unsqueeze(0).cuda()).cpu().squeeze()
        category = int(torch.argmax(result))
        confidence=result[category]
        logger.debug("Network result:",self.ret_dict[category],'confidence:',confidence)
        if self.vis:
            _,axs=plt.subplots(1,2)
            axs[0].set_title("sliding condition image")
            axs[0].imshow(cropped_img[0,:,:])
            axs[1].imshow(cropped_img[1,:,:])
            plt.show()
        if self.ret_dict[category] == "endpoint" and confidence>self.conf_threshold:
            logger.debug("Detected endpoint!")
        return self.ret_dict[category] == "endpoint" and confidence>self.conf_threshold

def click_points(img, CAM_T, intr):
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img.color.data)
    ax2.imshow(img.depth.data)
    points_3d = intr.deproject(img.depth)
    left_coords,right_coords = None,None
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
        nonlocal left_coords,right_coords
        point=CAM_T*points_3d[lin_ind]
        logger.debug("Clicked point in world coords: ",point)
        if(point.z>.5):
            logger.debug("Clicked point with no depth info!")
            return
        if(event.button==1):
            left_coords=coords
        elif(event.button==3):
            right_coords=coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords,right_coords

class FCNNetworkStopCond:
    def __init__(self,vis=False):
        self.model = KeypointNetwork("/home/jkerr/brijenfc-vision/models/endpoint_slide_fcn.ckpt", {"task": "cable_slide"}) # old, uses rgbd
        # self.model = KeypointNetwork("/home/jkerr/brijenfc-vision/models/endpoint-overhead-fcn.ckpt", {"task": "cable_slide"}) # new, just grayscale
        self.ret_dict={0:"endpoint", 1:"not_endpoint"}
        self.vis = vis
        self.T_CAM_BASE = RigidTransform.load("/home/justin/yumi/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")
        self.T_BASE_CAM = self.T_CAM_BASE.inverse()
        self.conf_threshold = 0.5
        
    def prepare_image(self,img:RgbdImage,right_pos:RigidTransform):
        img_color = img.color._data
        depth_og = img.depth.data
        p = Point(right_pos.translation,frame=YK.base_frame)
        
        intr = PhoXiSensor.create_intr(img_color.shape[1], img_color.shape[0])
        right_grip_pixel_coord = intr.project(self.T_BASE_CAM*p)
        right_x = right_grip_pixel_coord[0]
        right_y = right_grip_pixel_coord[1]
        logger.debug("RIGHT Y AND X", (right_y, right_x))
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        depth_og = np.squeeze(depth_og)
        thresh = 0.7
        gray[depth_og>thresh]=0
        img = gray
        depth_og[depth_og>thresh] = 0
        depth = depth_og

        img[int(4*img.shape[0]/5):, :] = 0
        depth[int(4*depth.shape[0]/5):, :] = 0
        img_crop = img[right_y-220:right_y-20, right_x-80: right_x+120]/255.
        depth_crop = depth[right_y-220:right_y-20, right_x-80: right_x+120]
        depth_crop = np.resize(depth_crop, (depth_crop.shape[1], depth_crop.shape[0]))
        combined = np.stack((img_crop, img_crop, img_crop)).astype(np.float32) 
        return combined

    def query(self,img:RgbdImage,right_gripper_pos:RigidTransform,which_arm:str='right'):
        cropped_img = self.prepare_image(img,right_gripper_pos)
        # if arm is left, flip image left to right
        # if which_arm=='left':
        #     cropped_img = cropped_img[::-1].copy()
        result = self.model(cropped_img, mode="class", prep=False)
        confidence = result.mean()
        category = confidence < self.conf_threshold
        logger.debug("Network result:",self.ret_dict[category],'confidence:',confidence)
        if self.vis:
            _,axs=plt.subplots(1,2)
            axs[0].set_title("sliding condition image")
            axs[0].imshow(cropped_img[0,:,:], cmap='gray')
            # axs[1].imshow(cropped_img[1,:,:])
            axs[1].imshow(result[0], cmap='gray')
            plt.show()
        if self.ret_dict[category] == "endpoint" and confidence>self.conf_threshold:
            input("Detected endpoint! press enter to continue")
        return self.ret_dict[category] == "endpoint" and confidence>self.conf_threshold

    def __call__(self,img:RgbdImage,right_gripper_pos:RigidTransform):
        return self.query(img, right_gripper_pos)

def get_num_connected_components(img):
    img = img._data.copy()
    img[img>0]=1
    img = img.astype(np.uint8)
    n_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    # filter out components with less than 10 pixels
    filtered_components = []
    for i in range(1, n_components):
        if stats[i, cv2.CC_STAT_AREA] > 8:
            filtered_components.append(i)

    return len(filtered_components)

def knot_in_hand(img, right_gripper_pos, which_arm='right', vis=True):
    T_CAM_BASE = RigidTransform.load(
        "/home/justin/yumi/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")
    intr = PhoXiSensor.create_intr(img.width, img.height)
    points_3d = T_CAM_BASE*intr.deproject(img.depth)
    if vis:
        o3dpc = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector((points_3d.data).T))
        open3d.visualization.draw_geometries([o3dpc])

    buffer_lo1 = np.array([0.025, -0.05, -0.07]) # prev 0.03q
    buffer_hi1 = np.array([0.06, 0.05, 0.1])

    buffer_lo2 = np.array([-.04, 0.005 if which_arm=='right' else -0.05,-.07])
    buffer_hi2 = np.array([.025,  0.05 if which_arm=='right' else -0.005,   0.1]) # prev 0.03
    box1 = Box(right_gripper_pos.translation + buffer_lo1,
              right_gripper_pos.translation + buffer_hi1, frame=points_3d.frame)
    box2 = Box(right_gripper_pos.translation + buffer_lo2,
              right_gripper_pos.translation + buffer_hi2, frame=points_3d.frame)
    crop_pointcloud1, _ = points_3d.box_mask(box1)
    crop_pointcloud2, _ = points_3d.box_mask(box2)
    if vis:
        o3dpc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector((crop_pointcloud1.data).T))
        open3d.visualization.draw_geometries([o3dpc])
        o3dpc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector((crop_pointcloud2.data).T))
        open3d.visualization.draw_geometries([o3dpc])

    tot_points = crop_pointcloud2.num_points+crop_pointcloud1.num_points
    logger.debug(crop_pointcloud1.num_points,crop_pointcloud2.num_points,tot_points)

    # slicing code to see if we have isolated segments of cable
    # cur_buffer_lo1 = np.array([0.03, -0.1, -0.07]) # prev 0.03
    # cur_buffer_hi1 = np.array([0.06, 0.035, 0.0])
    # box1 = Box(right_gripper_pos.translation + cur_buffer_lo1,
    #           right_gripper_pos.translation + cur_buffer_hi1, frame=points_3d.frame)
    # cropped, _ = points_3d.box_mask(box1)
    # o3dpc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector((cropped.data).T))
    # open3d.visualization.draw_geometries([o3dpc])

    for i in range(0, 6):
        if which_arm=='left':
            cur_buffer_lo1 = np.array([0.03, -0.1, -0.01 - 0.01*i]) # prev 0.03
            cur_buffer_hi1 = np.array([0.06, 0.035, 0.0 - 0.01*i])
        else:
            cur_buffer_lo1 = np.array([0.03, -0.035, -0.01 - 0.01*i]) # prev 0.03
            cur_buffer_hi1 = np.array([0.06, 0.1, 0.0 - 0.01*i])
        
        box1 = Box(right_gripper_pos.translation + cur_buffer_lo1,
              right_gripper_pos.translation + cur_buffer_hi1, frame=points_3d.frame)
        cropped, _ = points_3d.box_mask(box1)

        slice_img = intr.project_to_image(T_CAM_BASE.inverse() * cropped)

        connected_components = get_num_connected_components(slice_img)
        logger.debug("Number of connected components:",connected_components)
        if vis or connected_components >= 2 and vis:
            img_disp = img.copy()
            # img_disp._data[:, :, 2] = (slice_img._data.squeeze() > 0)
            plt.imshow(slice_img._data)
            plt.title("image for slice {}".format(i))
            plt.show()
        if connected_components >= 2 and tot_points > 1000:
            return True

    # gripper points is 577, 541
    return tot_points > 2000 #200 # 1700 before # 600 before #tot_points=0 often cause fall positives on knots bc the cable gets pushed to the side

def endpoint_in_hand(stopCond, img, right_gripper_pos: RigidTransform, which_arm='right'):
    T_CAM_BASE = RigidTransform.load(
        "/home/justin/yumi/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")
    intr = PhoXiSensor.create_intr(img.width, img.height)
    endpoints, _ = stopCond(img,vis=True)
    if len(endpoints)==0:return False
    for e in endpoints:
        e=e[0]
        logger.debug("endpoint",e)
        nearest_point = closest_nonzero_pt(img.depth._data.squeeze(),e)
        depth = img.depth._data[nearest_point[0],nearest_point[1]]
        point_3d = T_CAM_BASE*intr.deproject_pixel(depth,Point(np.array((nearest_point[1],nearest_point[0])),frame=intr.frame))
        logger.debug(f"endpoint: {e}, depth: {depth}, point3d: {point_3d._data.squeeze()}, gripper: {right_gripper_pos.translation}")
        xlo,xhi=0,.1
        ylo,yhi=-.05,.05
        zlo,zhi=-.05,.1
        p3d=Point(point_3d._data.squeeze()-right_gripper_pos.translation,frame=point_3d.frame)
        if p3d.x>xlo and p3d.x<xhi and p3d.y>ylo and p3d.y<yhi and p3d.z>zlo and p3d.z<zhi:
            return True
    return False


def slide():
    # stop_cond_net = NetworkStopCond("/home/justin/yumi/cable-untangling/untangling/endpoint_classifier/model_endpoint_classifier.pth",vis=True)
    #stop_cond_net = FCNNetworkStopCond(vis=True)
    stop_cond_net = KeypointNetwork(
        "~/brijenfc-vision/models/combined_endpoint_fcn.ckpt", params={'task': 'cable_endpoints'})
    def l_p(trans, rot=Interface.GRIP_DOWN_R):
        return RigidTransform(translation=trans, rotation=rot, from_frame=YK.l_tcp_frame, to_frame=YK.base_frame)

    def r_p(trans, rot=Interface.GRIP_DOWN_R):
        return RigidTransform(translation=trans, rotation=rot, from_frame=YK.r_tcp_frame, to_frame=YK.base_frame)
    SPEED = (.6, 4*np.pi)  # 0.5, 6*np.pi
    iface = Interface("1703005", ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                      ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
    iface.open_grippers()
    iface.home()
    iface.open_arms()
    time.sleep(1)
    img = iface.take_image()
    manual = True
    if not manual:
        keypoint_model = KeypointNetwork(
            "~/brijenfc-vision/models/endpoint_fcn.ckpt")
        endpoints = keypoint_model(img)
        logger.debug(endpoints)

        endpoint_grab_coords = endpoints[0][::-1]
        depth_img = img.depth._data  # [H, W, 1]
        color_img = img.color._data.copy()  # [H, W, 3]
        rgbd_img = np.concatenate((color_img, depth_img), axis=-1)
        traced_pts, old_pt_recent_pt = trace(
            rgbd_img, endpoint_grab_coords[::-1], True, 3, 0.0010)
        old_pt, recent_pt = old_pt_recent_pt
        old_pt = old_pt[:2][::-1]
        recent_pt = recent_pt[0:2][::-1]
        logger.debug(old_pt, recent_pt)
        logger.debug(f"Found {len(traced_pts)} points in BFS")

        for ret_pt in traced_pts:
            color_img[ret_pt[0], ret_pt[1], :] = np.array([255, 0, 0])

        # display image with tracing
        plt.imshow(color_img)
        plt.show()
        slide_grab_coords = (int(old_pt[0]), int(old_pt[1]))
        logger.debug(slide_grab_coords)

        plt.imshow(color_img)
        plt.scatter(*recent_pt)
        plt.scatter(*slide_grab_coords)
        plt.show()

        slide_grab_dir = ((old_pt - recent_pt) /
                          np.linalg.norm(old_pt - recent_pt))
    else:
        # np.save(f"/home/justin/yumi/cable-untangling/rollouts/sliding_depth/{time.time()}.npy", img.depth._data)
        slide_grab_coords, direction_coord=click_points(
            img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
        while slide_grab_coords is None or direction_coord is None:
            logger.debug("Must click two points")
            slide_grab_coords, direction_coord = click_points(
                img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
        coord_dir = (direction_coord[0]-slide_grab_coords[0],
                     direction_coord[1]-slide_grab_coords[1])
        slide_grab_dir = np.array((-coord_dir[1], -coord_dir[0])).astype(float)
        slide_grab_dir /= np.linalg.norm(slide_grab_dir)
    iface.home()

    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    grasp, which_arm = g.single_grasp_closer_arm(slide_grab_coords, 0.003, iface.tcp_dict)
    # grasp = g.single_grasp(slide_grab_coords, 0.003, iface.R_TCP)
    logger.debug('using arm for sliding', which_arm)
    iface.grasp_single(which_arm, grasp)
    iface.close_gripper(which_arm)
    iface.sync()

    T = iface.y.right.get_pose() if which_arm == 'right' else iface.y.left.get_pose()
    dotprod = T.matrix[:2, 1].dot(slide_grab_dir)
    logger.debug(
        f"gripper y: {T.matrix[:2,1]} slide dir: {slide_grab_dir}, dotprod {dotprod}")
    if T.translation[2] < .05:
        iface.go_delta_single(which_arm, [0, 0, .05], reltool=False)
    # this function is passed into slide so it knows when to stop
    def done_fn(iface_nonlocal, which_arm):
        img = iface_nonlocal.take_image()
        gripper_pos = iface_nonlocal.y.right.get_pose() if which_arm == 'right' else iface_nonlocal.y.left.get_pose()
        found_knot = knot_in_hand(img, gripper_pos, which_arm, vis=True)
        found_endpoint = endpoint_in_hand(stop_cond_net, img, gripper_pos, which_arm)
        logger.debug("knot,endpoint: ",found_knot,found_endpoint)
        if found_endpoint:
            return False, found_endpoint
        elif found_knot:
            return found_knot, False
        return False, False
    # slide until done
    knot,endpoint=iface.slide(sliding_arm=which_arm, distance=3, done_fn=done_fn, slide_direction=dotprod,
                do_initialization_motion=True, end_in_pull_arm=False)
    if knot:
        iface.flatten_new(which_arm=which_arm)
    if endpoint:
        init_endpoint_orientation(iface,which_arm)
        execute_spool(iface)

def test_slide_motion():
    from spool import init_endpoint_orientation
    SPEED = (.3, 4*np.pi)
    iface = Interface("1703005", ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                      ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
    for i in range(10):
        iface.open_grippers()
        iface.home()
        iface.sync()
        iface.close_grippers()
        dotprod = np.random.uniform(-1,1)
        # this function is passed into slide so it knows when to stop
        def done_fn(iface_nonlocal):
            return True,False
        # slide until done
        iface.slide(sliding_arm='right', distance=3, done_fn=done_fn, slide_direction=dotprod,
                    do_initialization_motion=True, end_in_pull_arm=False)
        iface.flatten_new()
        # init_endpoint_orientation(iface)

def test_separate_endpoints():
    from spool import init_endpoint_orientation
    SPEED = (.3, 4*np.pi)
    iface = Interface("1703005", ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                      ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
    iface.open_grippers()
    iface.home()
    iface.sync()
    iface.close_grippers()
    # this function is passed into slide so it knows when to stop
    img = iface.take_image()
    manual = True

    pt1, pt2 = click_points(
            img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
    while pt1 is None or pt2 is None:
        logger.debug("Must click two points")
        pt1, pt2 = click_points(
            img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
    
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    try:
        #move grippers
        l_grasp,r_grasp=g.double_grasp(pt1,pt2,.005,.005,iface.L_TCP,iface.R_TCP) #0.015 each
        iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp)
        iface.set_speed((.4,5))
        #####
    except GraspException:
        logger.debug("Ungraspable double location picked, retrying")
        grasp_failure = True
    
    iface.sync()
    iface.go_delta(l_trans=[0,0,.05],r_trans=[0,0,.05])
    iface.sync()
    iface.go_delta(l_trans=[0,0,-.03],r_trans=[0,0,-.03],reltool=True)
    iface.sync()
    iface.pull_apart(0.6)
    iface.sync()
    iface.open_grippers()
    iface.sync()
    iface.home()

def test_knot_in_hand():
    SPEED = (0.4, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    init = False
    if init:
        iface.open_arms()
        iface.close_grippers()
        iface.sync()
        move_to_slide_pos(iface,'left')
        iface.slide_grippers()
    while True:
        input(f"Press enter to take image")
        img = iface.take_image()
        r_pos = iface.y.left.get_pose()
        found_knot = knot_in_hand(img, r_pos, "left", vis=False)


if __name__ == '__main__':
    # test_separate_endpoints()
    # slide()
    # test_slide_motion()
    test_knot_in_hand()
