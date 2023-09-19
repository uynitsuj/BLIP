from untangling.utils.interface_rws import Interface
import numpy as np
from untangling.utils.grasp import GraspSelector
from fcvision.vision_utils import get_shake_point, get_highest_depth_pt_within_radius, get_cable_mask
from autolab_core import RigidTransform
from yumiplanning.yumi_kinematics import YuMiKinematics as YK
import matplotlib.pyplot as plt
from untangling.utils.workspace import WORKSPACE_CENTER, left_of_workspace
from untangling.point_picking import *
import random
import cv2
import time
import logging

logger = logging.getLogger("Untangling")

def np_to_int_tuple(np_array):
    return tuple(np_array.astype(int).tolist())


def l_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(translation=trans, rotation=rot, from_frame=YK.l_tcp_frame, to_frame=YK.base_frame)


def r_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(translation=trans, rotation=rot, from_frame=YK.r_tcp_frame, to_frame=YK.base_frame)


def reveal_endpoint(img, endpoints=[]):
    # blackout center of image
        img_cp = img.copy()
        img_cp = img_cp.color.data
        img_cp[600:] = 0
        img_blked = img_cp.copy()

        left_arm_mask_poly = (np.array([(0, 174), (0, 36), (24, 24), (48, 18), (96, 24), (148, 48), (186, 75), (231, 148), (236, 186)])*4.03125).astype(np.int32)
        right_arm_mask_poly = (np.array([(30, 176), (35, 150), (70, 77), (96, 52), (160, 26), (255, 22), (255, 186)])*4.03125).astype(np.int32)

        def draw_mask(polygon_coords, shape, blur=None):
            image = np.zeros(shape=shape, dtype=np.float32)
            cv2.drawContours(image, [polygon_coords], 0, (1.0), -1)    
            if blur is not None:        
                image = cv2.GaussianBlur(image, (blur, blur), 0)    
            return image

        left_arm_reachability_mask = (draw_mask(left_arm_mask_poly, img_cp.shape[:2])).astype(np.uint8)
        right_arm_reachability_mask = (draw_mask(right_arm_mask_poly, img_cp.shape[:2])).astype(np.uint8)

        combined_mask = (left_arm_reachability_mask + right_arm_reachability_mask) > 0.0
        combined_mask = combined_mask.astype(np.uint8)

        kernel = np.ones((20, 20), np.uint8)
        mask_large = cv2.erode(combined_mask, kernel)

        kernel2 = np.ones((40, 40), np.uint8)
        mask_small = cv2.erode(mask_large, kernel2)

        new_mask = mask_large - mask_small
        new_mask[110:, 170:200] = 1.0
        new_mask[470:500, 170:] = 1.0
        new_mask[:, :170] = 0.0
        new_mask[600:, :] = 0.0
        new_mask = np.expand_dims(new_mask, axis=-1)
        new_mask = np.concatenate((new_mask, new_mask, new_mask), axis=-1)

        img_cp[new_mask <=0.0] = 0 # = (np.logical_and(img_cp, mask_small)).astype(np.uint8) * 255

        depth = img.depth.data
        img_cp[depth < 0.1] = 0

        img_cp_mask = (img_cp[:,:,0] > 100).astype(np.uint8)
        edge_mask = cv2.dilate(img_cp_mask, kernel)
        _, _, _, centroids = cv2.connectedComponentsWithStats(edge_mask, 8, cv2.CV_32S)
        centroids = (centroids[1:][::-1]).astype(np.uint32)

        # get non-dark pixels along border
        if len(centroids) == 0:
            img_blked[depth < 0.1] = 0
            centroids = np.argwhere(img_blked[:, :, 0] > 150)

        idx = random.randint(0, len(centroids)-1)
        grasp_point = centroids[idx][::-1]

        white_pixs = np.argwhere(img_cp[:, :, 0] > 150)
        distances = np.linalg.norm(white_pixs - grasp_point, axis=-1)
        grasp_point = white_pixs[np.argmin(distances)][::-1]

        plt.title("Endpoint reveal move point.")
        plt.imshow(img_cp)
        plt.scatter(grasp_point[0], grasp_point[1])
        plt.savefig('alg_output.png')

        arm_to_use = 'left' if left_of_workspace((grasp_point[1], grasp_point[0])) else 'right'
        logger.debug("arm to use: " + arm_to_use)
        try:
            if arm_to_use == 'left':
                iface.go_delta(l_trans=[0, 0, 0.05])
            else:
                iface.go_delta(r_trans=[0, 0, 0.05])
            grasp, _ = g.single_grasp_closer_arm(grasp_point, 0.008, {"left": iface.L_TCP, "right": iface.R_TCP})
        except Exception:
            logger.debug("Single grasp failed, falling back to top down grasp")
            grasp = g.top_down_grasp(tuple(grasp_point), .008, iface.L_TCP if arm_to_use == 'left' else iface.R_TCP)
        iface.grasp(l_grasp=grasp, reid_grasp=True) if arm_to_use == 'left' else iface.grasp(r_grasp=grasp)
        
        try:
            if arm_to_use == 'left':
                iface.go_delta(l_trans=[0, 0, 0.05])
            else:
                iface.go_delta(r_trans=[0, 0, 0.05])
        except:
            logger.debug("Sad. Cannot go delta 0.05 up.")
        iface.sync()

        if arm_to_use == 'right':
            rp1 = RigidTransform(
                translation=[(grasp.pose.translation[0] - 0.35) * 1/6 + 0.35, grasp.pose.translation[1] * 1/4, 0.05],
                rotation=iface.GRIP_DOWN_R, #RigidTransform.x_axis_rotation(-.4)@
                from_frame=YK.r_tcp_frame,
                to_frame="base_link"
            )
            try:
                iface.go_cartesian(
                    r_targets=[rp1],
                    nwiggles=(None, None),
                    rot=(None, None), removejumps=[6]
                )
            except:
                logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                iface.go_pose_plan(r_target=rp1)
            iface.open_grippers() 
            time.sleep(iface.GRIP_SLEEP_TIME)
            iface.sync()       
            iface.go_configs(r_q=[iface.R_HOME_STATE])
            iface.sync()
        else:
            lp1 = RigidTransform(
                translation=[(grasp.pose.translation[0] - 0.35) * 1/6 + 0.35, grasp.pose.translation[1] * 1/4, 0.05],
                rotation=iface.GRIP_DOWN_R, #RigidTransform.x_axis_rotation(-.4)@
                from_frame=YK.l_tcp_frame,
                to_frame="base_link"
            )
            try:
                iface.go_cartesian(
                    l_targets=[lp1],
                    nwiggles=(None, None),
                    rot=(None, None), removejumps=[6]
                )
            except:
                logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                iface.go_pose_plan(l_target=lp1)
            iface.open_grippers()
            time.sleep(iface.GRIP_SLEEP_TIME)
            iface.sync()
            iface.go_configs(l_q=iface.L_HOME_STATE)
            iface.sync()
        iface.sync()
        iface.open_arms()
        iface.sync()

def debug_grasp():
    while True:
        img = iface.take_image()
        grasp_point, _ = click_points(img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
        arm_to_use = 'left' if left_of_workspace((grasp_point[1], grasp_point[0])) else 'right'
        g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
        try:
            grasp, _ = g.single_grasp_closer_arm(grasp_point, 0.006, {"left": iface.L_TCP, "right": iface.R_TCP})
        except Exception:
            print("Single grasp failed, falling back to top down grasp")
            grasp = g.top_down_grasp(tuple(grasp_point), .006, iface.L_TCP if arm_to_use == 'left' else iface.R_TCP)
        iface.grasp(l_grasp=grasp) if arm_to_use == 'left' else iface.grasp(r_grasp=grasp)
        iface.sync()


if __name__ == '__main__':
    from untangling.utils.tcps import *
    SPEED = (.5, 6*np.pi)
    iface = Interface("1703005", ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                      ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
    iface.open_grippers()
    iface.open_arms()
    # debug_grasp()
    img = iface.take_image()
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    endpoint, _ = click_points(img, iface.T_PHOXI_BASE, iface.cam.intrinsics)
    endpoints = np.array([endpoint]) 
    print("endpoints: ", endpoints)
    reveal_endpoint(img=img, endpoints=endpoints)
