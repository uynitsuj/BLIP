from untangling.utils.interface_rws import Interface
import numpy as np
from untangling.utils.grasp import GraspSelector
from fcvision.vision_utils import get_shake_point, get_highest_depth_pt_within_radius, get_cable_mask
from autolab_core import RigidTransform
from yumiplanning.yumi_kinematics import YuMiKinematics as YK
import matplotlib.pyplot as plt
from untangling.utils.workspace import WORKSPACE_CENTER, left_of_workspace
import logging

def np_to_int_tuple(np_array):
    return tuple(np_array.astype(int).tolist())


def l_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(translation=trans, rotation=rot, from_frame=YK.l_tcp_frame, to_frame=YK.base_frame)


def r_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(translation=trans, rotation=rot, from_frame=YK.r_tcp_frame, to_frame=YK.base_frame)


def recenter(iface: Interface, img):
    logger.debug("Recentering...")
    shake_point = get_shake_point(img._data, random=False)[::-1]
    logger.debug("Shake point: ", shake_point)
    vector_delta = np.array(shake_point) - WORKSPACE_CENTER
    if (np.linalg.norm(vector_delta) < 0.1):
        return

    points = np.nonzero(get_cable_mask(img._data))
    points = np.array(points).T

    # find farthest point in direction of vector_delta
    dotted = np.dot(points - WORKSPACE_CENTER, vector_delta)
    farthest_point = np.array(shake_point)  # points[np.argmin(dotted)]

    # plt.imshow(img._data[:, :, :3].astype(np.uint8))
    # plt.scatter(farthest_point[1], farthest_point[0], c='r')
    # plt.scatter(shake_point[1], shake_point[0], c='g')
    # plt.scatter(workspace_center[1], workspace_center[0], c='b')
    # plt.show()

    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    # logger.debug(farthest_point.shape, farthest_point)
    farthest_point_grasp = (int(farthest_point[1]), int(farthest_point[0]))
    target_pos = g.ij_to_point(np_to_int_tuple(
        farthest_point - WORKSPACE_CENTER))

    # TODO: make this use left_of_workspace
    which_arm = 'left' if vector_delta[1] < 0 else 'right'
    if which_arm == 'left':
        graspL = g.single_grasp(farthest_point_grasp, .003, iface.L_TCP)
        iface.grasp(l_grasp=graspL)
        iface.sync()
        cur_pose = iface.y.left.get_pose().copy()
        cur_pose.translation[2] = 0.05
        iface.go_pose(which_arm, cur_pose, linear=False)
        iface.sync()
        iface.go_pose(which_arm, l_p(
            [target_pos.x, target_pos.y, 0.05]), linear=False)
    else:
        graspR = g.single_grasp(farthest_point_grasp, .003, iface.R_TCP)
        iface.grasp(r_grasp=graspR)
        iface.sync()
        cur_pose = iface.y.right.get_pose().copy()
        cur_pose.translation[2] = 0.05
        iface.go_pose(which_arm, cur_pose, linear=False)
        iface.sync()
        iface.go_pose(which_arm, r_p(
            [target_pos.x, target_pos.y, 0.05]), linear=False)
    iface.go_delta_single(which_arm, [0, 0, 0.05], linear=False)
    iface.sync()

    iface.open_grippers()
    iface.open_arms()
    return


def shake(iface: Interface, img, endpoints=[], com=False):
    # shake action
    point_highest = None
    endpoint_shake = True
    if not com and len(endpoints) > 0:
        endpt_of_interest = endpoints[np.random.randint(0, endpoints.shape[0])]
        logger.debug("Trying to shake on an endpoint...", endpt_of_interest[0])
        point = endpt_of_interest[0]
        point_highest = get_highest_depth_pt_within_radius(
            img._data[:, :, 3], (point[0], point[1]),radius=10)
        if img.data[point_highest[0], point_highest[1], 3] == 0:
            logger.debug("But invalid depth")
            point_highest = None
    if point_highest is None:
        com = True
        endpoint_shake = False
        logger.debug("Due to no valid endpoints, shaking near the center...")
        point = get_shake_point(img._data, random=True)
        point_highest = get_highest_depth_pt_within_radius(
            img._data[:, :, 3], (point[1], point[0]))
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE,grabbing_endpoint=endpoint_shake)
    
    use_left_arm = left_of_workspace(point_highest)
    which_arm = 'left' if use_left_arm else 'right'
    point_highest = (int(point_highest[1]), int(point_highest[0]))
    try:
        grasp = g.single_grasp(point_highest, .003 + 0.004*int(endpoint_shake),
                               iface.L_TCP if use_left_arm else iface.R_TCP)
    except Exception:
        logger.debug("Single grasp failed, falling back to top down grasp")
        grasp = g.top_down_grasp(
            point_highest, .003 + 0.004*int(endpoint_shake), iface.L_TCP if use_left_arm else iface.R_TCP)
    # lift up arm slightly
    try:
        iface.go_delta_single(which_arm, [0, 0, .10], linear=False)
        iface.sync()
    except Exception:
        pass
    iface.grasp(l_grasp=grasp) if use_left_arm else iface.grasp(r_grasp=grasp)
    iface.sync()
    out_dist=.35#distance in y to shake towards arm
    drop_dist=.2
    if use_left_arm:
        iface.go_pose_plan(l_target=l_p([.4, out_dist, .7]))
    else:
        iface.go_pose_plan(r_target=r_p([.4, -out_dist, .7]))
    iface.sync()
    iface.shake_left_J([2], num_shakes=3, ji=[5], speed=(
        5, 2000)) if use_left_arm else iface.shake_right_J([2], num_shakes=3, ji=[5], speed=(5, 2000))
    iface.sync()
    com_val = -1 #if com else 1
    if use_left_arm:
        iface.go_pose_plan(l_target=l_p([.4,0,.15]), table_z=0.06)
        iface.go_pose_plan(l_target=l_p([.4,com_val*drop_dist,.10]), table_z=0.06) # used to be negative drop dist and vice versa for bottom case
    else:
        iface.go_pose_plan(r_target=r_p([.4,0,.15]), table_z=0.06)
        iface.go_pose_plan(r_target=r_p([.4,com_val*-drop_dist,.10]), table_z=0.06)
    iface.sync()
    iface.open_grippers()
    iface.open_arms()
    img = iface.take_image()


if __name__ == '__main__':
    from untangling.utils.tcps import *
    SPEED = (.5, 6*np.pi)
    iface = Interface("1703005", ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                      ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
    iface.open_grippers()
    iface.open_arms()
    img = iface.take_image()
    shake(iface=iface, img=img)
