# from msvcrt import SEM_FAILCRITICALERRORS
# create functions for return_to_center actions and gripper slide / cage actions in pull apart and pull up
from fileinput import close
from turtle import down, left, right
from yumirws.yumi import YuMi
from autolab_core import RigidTransform, Point
import numpy as np
from yumiplanning.yumi_kinematics import YuMiKinematics as YK
from yumiplanning.yumi_planner import *
from phoxipy.phoxi_sensor import PhoXiSensor
from untangling.utils.grasp import Grasp, GraspSelector
import time
from scipy import interpolate as inter
from untangling.utils.tcps import ABB_WHITE
from untangling.utils.workspace import WORKSPACE_CENTER, left_of_workspace
# commenting for now
# from in_hand_manipulation.zed import ZedImageCapture
import math
import matplotlib.pyplot as plt
import logging
import random 
import cv2
import os
import traceback
logger = logging.getLogger("Untangling")
"""
This class invokes yumirws, phoxipy, yumiplanning to give a user-friendly interface for
sending motion commands, getting depth camera data, and etc

Consider it an abstract wrapper for the yumi+phoxi which enables some project-specific
capabilities without polluting the original code
(ie keep the yumipy,yumiplanning repo's pure to their intent)
"""

def getH(R, T, which_arm):
    tip_frame = YK.l_tcp_frame if which_arm == "left" else YK.r_tcp_frame
    return RigidTransform(R, T, tip_frame, YK.base_frame)


def remove_twist(traj, joint=-1):
    """removes the jump in the last joint by freezing it before the jump"""
    lastval = traj[0, joint]
    jumpi = None
    for i in range(1, len(traj)):
        if abs(traj[i, joint] - traj[i - 1, joint]) > 0.4:
            jumpi = i
            break
        lastval = traj[i, joint]
    if jumpi is not None:
        traj[jumpi:, joint] = lastval
    return traj


class Interface:
    # orientation where the gripper is facing downwards
    GRIP_DOWN_R = np.diag([1, -1, -1])
    GRIP_UP_R = np.diag([1, 1, 1])
    GRIP_SIDEWAYS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    GRIP_SIDEWAYS_L = RigidTransform.x_axis_rotation(-np.pi / 2) @ GRIP_DOWN_R
    GRIP_TILT_R = RigidTransform.x_axis_rotation(np.pi / 4) @ RigidTransform.z_axis_rotation(np.pi / 2) @ GRIP_DOWN_R
    GRIP_TILT_L = RigidTransform.x_axis_rotation(-np.pi / 4) @ RigidTransform.z_axis_rotation(np.pi / 2) @ GRIP_DOWN_R
    GRIP_TILT_R_Y = RigidTransform.y_axis_rotation(-np.pi / 8) @ RigidTransform.z_axis_rotation(np.pi / 2) @ GRIP_DOWN_R
    GRIP_TILT_L_Y = RigidTransform.y_axis_rotation(np.pi / 8) @ RigidTransform.z_axis_rotation(np.pi / 2) @ GRIP_DOWN_R
    GRIP_SLIDE_DIST = 0.002 #24 #5  # meters
    SLIDE_SLEEP_TIME = 0.1
    GRIP_SLEEP_TIME = 0.7
    L_HOME_STATE = np.array([
        -0.5810662,
        -1.34913424,
        0.73567095,
        0.55716616,
        1.56402364,
        1.25265177,
        2.84548536,
    ]
    )
    R_HOME_STATE = np.array(
        [
            0.64224786,
            -1.34920282,
            -0.82859683,
            0.52531042,
            -1.64836569,
            1.20916355,
            -2.83024169,
        ]
    )

    L_CENTER_STATE = np.array([-1.1572419 , -0.81430498,  0.62794815,  0.4810117 ,  1.54553438,
        1.66530219, -0.92134641])

    R_CENTER_STATE = None #initialize in init

    L_PIC_STATE = [
        -0.74470653,
        -1.17249329,
        0.97104672,
        -0.41459336,
        1.25081939,
        1.18064083,
        -2.62890405,
    ]
    R_PIC_STATE = np.array(
        [
            0.69660052,
            -1.04367319,
            -1.15089772,
            -0.46609577,
            -1.01159388,
            1.07832464,
            2.54754901,
        ]
    )

    def __init__(
        self,
        phoxi_name="1703005",
        l_tcp=ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        r_tcp=ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=(0.2, 2 * np.pi),
        zed=False,
    ):
        # set up the yumi robot
        self.L_TCP = l_tcp # left hand
        self.R_TCP = r_tcp # right hand
        self.R_CENTER_STATE = self.l_to_r(self.L_CENTER_STATE)
        self.tcp_dict = {'left': self.L_TCP, 'right': self.R_TCP}
        self.speed = speed
        self.y = YuMi(l_tcp=self.L_TCP, r_tcp=self.R_TCP)
        self.yk = YK()
        self.yk.set_tcp(self.L_TCP, self.R_TCP)
        self.default_speed = speed
        self.set_speed(speed)
        # set up the phoxi
        self.T_PHOXI_BASE = RigidTransform.load(
            "/home/mallika/triton4-lip/phoxipy/tools/phoxi_to_world_bww.tf"
        ).as_frames(from_frame="phoxi", to_frame="base_link")
        # commented out by m for now
        # self.T_ZED_BASE = RigidTransform.load(
        #     "/home/justin/yumi/cable-untangling/in_hand_manipulation/T_zed_world.tf"
        # ).as_frames(from_frame="zed", to_frame="base_link")
        self.zed = zed
        if zed:
            self.cam = ZedImageCapture()
            # TODO: do we need to start this as well?

        else:
            self.cam = PhoXiSensor("1703005")
            # self.cam.start()
            # # m change
            # # img = self.cam.read(cam_only=False)
            #img = self.cam.read()
            #print(img.width, img.height)
            #print(type(img.width), type(img.height))
            height, width = 772, 1032
            self.cam.intrinsics = self.cam.create_intr(width, height)
            #self.img_width = img.width
            #print(img.width)
            #self.img_height = img.height
            #print(img.height)
        self.planner = Planner()

        self.before_after_poses_dir = '/home/mallika/triton4-lip/cable-untangling/untangling/utils/debug_motions/before_after_poses'
        self.successful_motion_dir = '/home/mallika/triton4-lip/cable-untangling/untangling/utils/debug_motions/successful_motions'

    def get_frame(self, which_arm):
        return self.yk.l_tcp_frame if which_arm == "left" else self.yk.r_tcp_frame

    def take_image(self, cam_only=False, depth=False):
        # import pdb; pdb.set_trace()
        if self.zed:
            if depth:
                img_left, img_right, depth_data = self.cam.capture_image(depth=depth)
            else:
                img_left, img_right = self.cam.capture_image(depth=depth)
            np.save('left.npy', img_left)
            np.save('right.npy', img_right)                
            img_left = np.load('left.npy')
            img_right = np.load('right.npy')
            if depth:
                np.save('depth.npy', depth_data)
                depth_data = np.load('depth.npy')
                return img_left, img_right, depth_data
            return img_left, img_right
        else:
            self.cam.start()
            # m change
            # img = self.cam.read(cam_only=False)
            img = self.cam.read()
            # self.cam.intrinsics = self.cam.create_intr(img.width, img.height)
            # self.img_width = img.width
            # self.img_height = img.height
            return img
            # # m change
            # # return self.cam.read(cam_only)
            # return self.cam.read()

    def set_speed(self, speed):
        """
        set tcp move speed. format is tuple of (m/s,deg/s)
        """
        self.speed = speed

    def home(self):
        l_pos = self.y.left.get_pose()
        r_pos = self.y.right.get_pose()
        if l_pos.translation[2] < .08 or r_pos.translation[2] < .08:
            self.set_speed((.2, np.pi))
            self.go_delta([0, 0, 0.15 - l_pos.translation[2]],
                          [0, 0, 0.15 - r_pos.translation[2]])
            
            self.sync()
        self.set_speed((.4, 2*np.pi))

        try:
            self.go_config_plan(self.L_HOME_STATE,
                                self.R_HOME_STATE, table_z=.08)
        except PlanningException:
            self.go_configs(self.L_HOME_STATE, self.R_HOME_STATE)
        self.set_speed(self.default_speed)
        
    def home_near(self):
        l_pos = self.y.left.get_pose()
        r_pos = self.y.right.get_pose()
        if l_pos.translation[2] < .08 or r_pos.translation[2] < .08:
            self.set_speed((.2, np.pi))
            self.go_delta([0, 0, 0.15 - l_pos.translation[2]],
                          [0, 0, 0.15 - r_pos.translation[2]])
            
            self.sync()
        self.set_speed((.4, 2*np.pi))
        try:
            self.go_config_plan(self.L_HOME_STATE,
                                self.R_HOME_STATE, table_z=.08)
        except PlanningException:
            self.go_configs([self.L_HOME_STATE], [self.R_HOME_STATE])
        self.set_speed(self.default_speed)

    def open_arms(self):
        self.sync()
        l_pos = self.y.left.get_pose()
        r_pos = self.y.right.get_pose()
        if l_pos.translation[2] < .08 or r_pos.translation[2] < .08:
            self.set_speed((.2, np.pi))
            self.go_delta([0, 0, 0.15 - l_pos.translation[2]],
                          [0, 0, 0.15 - r_pos.translation[2]])
            
            self.sync()
        if self.y.left.get_pose().translation[1] < 0.3 and \
           self.y.right.get_pose().translation[1] > -0.3:
            self.home()
            self.sync()
        try:
            self.go_config_plan(self.L_PIC_STATE, self.R_PIC_STATE,table_z=.07)
        except PlanningException:
            self.go_configs([self.L_PIC_STATE], [self.R_PIC_STATE])
        self.sync()

    def __del__(self):
        print("closing")
        # print("camera closing!")
        # if self.zed:
        #     self.cam.close()
        # else:
        #     self.cam.stop()

    def open_grippers(self):
        self.y.right.open_gripper()
        self.y.left.open_gripper()
        # self.y.right.open_gripper()

    def open_gripper(self, which_arm):
        '''
        FLAG: FIX!
        '''
        # arm = self.y.right
        # arm = self.y.left if which_arm == "left" else self.y.right
        # arm.open_gripper()
        if which_arm == "left":
            self.y.left.open_gripper()
        else:
            self.y.right.open_gripper()


    def close_gripper(self, which_arm):
        '''
        FLAG: FIX!
        '''
        # arm = self.y.right
        arm = self.y.left if which_arm == "left" else self.y.right
        arm.close_gripper()

    def close_grippers(self):
        self.y.left.close_gripper()
        self.y.right.close_gripper()

    def slide_grippers(self):
        self.y.left.move_gripper(self.GRIP_SLIDE_DIST)
        self.sync()
        self.y.right.move_gripper(self.GRIP_SLIDE_DIST)
        self.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)


    def slide_gripper(self, which_arm):
        '''
        FLAG: FIX!
        '''
        if which_arm == "left":
            self.y.left.move_gripper(self.GRIP_SLIDE_DIST)
            self.y.left.sync()
        else:
            self.y.right.move_gripper(self.GRIP_SLIDE_DIST)
            self.y.right.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)

    def move_gripper(self, which_arm,dist):
        '''
        FLAG: FIX!
        '''
        if which_arm == "left":
            self.y.left.move_gripper(dist)
            self.y.left.sync()
        else:
            self.y.right.move_gripper(dist)
            self.y.right.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)

    def log_to_file(self, data_dict):
        cur_time = time.time()
        np_file_path = os.path.join(self.before_after_poses_dir, str(cur_time) + '.npy')
        data_dict.update({
            'l_joint_angles': self.y.left.get_joints(),
            'r_joint_angles': self.y.right.get_joints(),
            'stack_trace': traceback.format_stack()
        })
        logger.debug(data_dict)
        np.save(np_file_path, data_dict)
        return np_file_path

    def go_cartesian(
        self,
        l_targets=[],
        r_targets=[],
        fine=False,
        nwiggles=(None, None),
        rot=(None, None),
        removejumps=[],
        wiggle_joint=5,N=30, mode='Speed',
        n_tries=15
    ):
        log_path = self.log_to_file({
            'l_targets': l_targets,
            'r_targets': r_targets,
            'type': 'go_cartesian',
            'fine': fine,
            'nwiggles': nwiggles,
            'rot': rot,
            'removejumps': removejumps,
        })
        #self.sync()
        l_cur_q = self.y.left.get_joints()
        r_cur_q = self.y.right.get_joints()
        l_cur_p, r_cur_p = self.yk.fk(qleft=l_cur_q, qright=r_cur_q)
        lpts = [l_cur_p] + l_targets
        rpts = [r_cur_p] + r_targets

        # compute the actual path (THESE ARE IN URDF ORDER (see urdf_order_2_yumi for details))
        earliest_jump, earliest_jump_path = -1, None
        for i in range(n_tries):
            try:
                lpath, rpath, earliest_jump_ret = self.yk.interpolate_cartesian_waypoints(
                    l_waypoints=lpts, l_qinit=l_cur_q, r_qinit=r_cur_q, r_waypoints=rpts, N=N,mode=mode,
                    random=(i>0)
                )
                earliest_jump = max(earliest_jump, earliest_jump_ret)
                earliest_jump_path = (lpath, rpath)
            except Exception as e:
                logger.warning(e)
                logger.warning("Retrying planning...")
            if earliest_jump == 10:
                break
        if earliest_jump == -1:
            raise Exception("Unable to plan jump-free cartesian path despite numerous attempts.")
        lpath, rpath = earliest_jump_path

        lpath = np.array(lpath)
        rpath = np.array(rpath)
        for j in removejumps:
            if len(l_targets)>0:
                lpath = remove_twist(lpath, j)
            if len(r_targets)>0:
                rpath = remove_twist(rpath, j)
        if nwiggles[0] is not None:
            lpath = self.wiggle(lpath, rot[0], nwiggles[0], wiggle_joint)
        if nwiggles[1] is not None:
            rpath = self.wiggle(rpath, rot[1], nwiggles[1], wiggle_joint)
        self.go_configs(l_q=lpath, r_q=rpath, together=True)
        # this "fine" motion is to account for the fact that our ik is slightly (~.5cm) wrong becase
        # the urdf file is wrong
        if fine:
            if len(l_targets) > 0:
                self.y.left.goto_pose(l_targets[0], speed=self.speed)
            if len(r_targets) > 0:
                self.y.right.goto_pose(r_targets[0], speed=self.speed)
        np.save(log_path.replace(self.before_after_poses_dir, self.successful_motion_dir), [])

    def go_pose_plan_single(self,which_arm,pose,table_z=PLAN_TABLE_Z,mode='Manipulation1'):
        if which_arm=='left':
            self.go_pose_plan(l_target=pose,table_z=table_z,mode=mode)
        else:
            self.go_pose_plan(r_target=pose,table_z=table_z,mode=mode)

    def go_pose_plan(
        self, l_target=None, r_target=None, fine=False, table_z=PLAN_TABLE_Z,
                mode='Manipulation1'):
        self.sync()
        log_path = self.log_to_file({
            'l_targets': l_target,
            'r_targets': r_target,
            'type': 'go_pose_plan',
            'fine': fine,
            'table_z': table_z,
        })
        l_cur = self.y.left.get_joints()
        r_cur = self.y.right.get_joints()

        res = self.planner.plan_to_pose(
            l_cur, r_cur, self.yk, l_target, r_target, table_z=table_z,mode=mode
        )
        if res is None:
            raise PlanningException("Planning to pose failed")
        l_path, r_path = res

        self.go_configs(l_path, r_path, True)
        if fine:
            if l_target is not None:
                self.y.left.goto_pose(l_target, speed=self.speed)
            if r_target is not None:
                self.y.right.goto_pose(r_target, speed=self.speed)
        np.save(log_path.replace(self.before_after_poses_dir, self.successful_motion_dir), [])

    def go_config_plan(self, l_q=None, r_q=None, table_z=PLAN_TABLE_Z):
        self.sync()
        l_cur = self.y.left.get_joints()
        r_cur = self.y.right.get_joints()
        l_path, r_path = self.planner.plan(
            l_cur, r_cur, l_q, r_q, table_z=table_z)
        self.go_configs(l_path, r_path, True)

    def go_pose(self, which_arm, pose, linear=True):
        arm = self.y.left if which_arm == "left" else self.y.right
        arm.goto_pose(pose, speed=self.speed, linear=linear)

    def go_delta(self, l_trans=None, r_trans=None, reltool=False, linear=True, preserve_rot=False):
        # meant for small motions
        self.sync()
        l_delta, r_delta = None, None
        if l_trans is not None:
            l_cur = self.y.left.get_pose()
            if reltool:
                l_delta = RigidTransform(
                    translation=l_trans,
                    from_frame=l_cur.from_frame,
                    to_frame=l_cur.from_frame,
                )
                l_new = l_cur * l_delta
            else:
                l_delta = RigidTransform(
                    translation=l_trans,
                    from_frame=l_cur.to_frame,
                    to_frame=l_cur.to_frame,
                )
                l_new = l_delta * l_cur
        if r_trans is not None:
            r_cur = self.y.right.get_pose()
            if reltool:
                r_delta = RigidTransform(
                    translation=r_trans,
                    from_frame=r_cur.from_frame,
                    to_frame=r_cur.from_frame,
                )
                r_new = r_cur * r_delta
            else:
                r_delta = RigidTransform(
                    translation=r_trans,
                    from_frame=r_cur.to_frame,
                    to_frame=r_cur.to_frame,
                )
                r_new = r_delta * r_cur
        if preserve_rot:
            l_new.rotation = l_cur.rotation
            r_new.rotation = r_cur.rotation
        if l_delta is not None:
            self.y.left.goto_pose(l_new, speed=self.speed, linear=linear)
        if r_delta is not None:
            self.y.right.goto_pose(r_new, speed=self.speed, linear=linear)

    def go_delta_single(self, which_arm, trans, reltool=False, linear=True):
        if which_arm == "left":
            self.go_delta(l_trans=trans, reltool=reltool)
        elif which_arm == "right":
            self.go_delta(r_trans=trans, reltool=reltool)

    def l_to_r(self, config):
        return np.array([-1, 1, -1, 1, -1, 1, 1]) * config
    
    def pull_apart(self, dist, slide_left=True,slide_right=True, return_to_center=False, tilt=False):
        self.sync()
        # initialize grip types according to need
        if slide_left:
            # logger.info("LEFT SLIDE")
            logger.debug("slide left")
            self.slide_gripper("left")
        else:
            # logger.info("LEFT CLOSE")
            self.close_gripper("left")
        if slide_right:
            # logger.info("RIGHT SLIDE")
            logger.debug("slide right")
            self.slide_gripper("right")
        else:
            # logger.info("RIGHT CLOSE")
            self.close_gripper("right")
        self.sync()

        try:
            self.go_delta(l_trans=[0,0,0.05], r_trans=[0,0,0.05])
        except:
            logger.debug("Go delta failed.")
    
        l_R = self.y.left.get_pose().rotation
        if l_R[:, 1].dot(np.array((1, 0, 0))) > 0:  # if y axis points towards pos x
            l_tar_R = RigidTransform.z_axis_rotation(np.pi/2)@self.GRIP_DOWN_R
        else:
            l_tar_R = RigidTransform.z_axis_rotation(-np.pi/2)@self.GRIP_DOWN_R
        lp = RigidTransform(
            translation=[0.4, dist, 0.2],
            rotation=l_tar_R,  # 0.45 (also below)
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        
        r_R = self.y.right.get_pose().rotation
        if r_R[:, 1].dot(np.array((1, 0, 0))) > 0:  # if y axis points towards pos x
            r_tar_R = RigidTransform.z_axis_rotation(np.pi/2)@self.GRIP_DOWN_R
        else:
            r_tar_R = RigidTransform.z_axis_rotation(-np.pi/2)@self.GRIP_DOWN_R
        rp = RigidTransform(
            translation=[0.4, -dist, 0.2],
            rotation=r_tar_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

        self.sync()
        # def l_to_r(config):
        #     return np.array([-1, 1, -1, 1, -1, 1, 1]) * config

        # cur_left_pos = self.y.left.get_joints()
        # cur_right_pos = self.y.right.get_joints()
        # CENTER_L = np.array([-1.41362668, -1.20112134,  1.07746664,  0.39039282,  1.47596742, 1.22066055, cur_left_pos[6]])
        # CENTER_R = l_to_r(CENTER_L)
        # CENTER_R[6] = cur_right_pos[6]

        # self.go_configs(l_q=[CENTER_L], r_q=[CENTER_R])
        # self.sync()
        
        try:
            self.go_cartesian(
                l_targets=[lp],
                r_targets=[rp],
                nwiggles=(10, 10),
                rot=(0.3, 0.3),
                removejumps=[5, 6, 4],
            )
        except:
            logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
            self.set_speed((.2,np.pi*2/3))
            self.go_pose_plan(lp, rp)
            self.set_speed(self.default_speed)
            
        if tilt:
            self.open_grippers()
            self.sync()
            lp.rotation = self.GRIP_TILT_L
            rp.rotation = self.GRIP_TILT_R
            self.go_cartesian(
                l_targets=[lp],
                r_targets=[rp],
                nwiggles=(1, 1),
                rot=(0.3, 0.3),
                removejumps=[5, 6, 4]
            )
            self.sync()
        
        if return_to_center:
            # now move closer to center again
            lp = RigidTransform(
                translation=[0.5, 0.1, 0.15],
                rotation=self.GRIP_DOWN_R,
                from_frame=YK.l_tcp_frame,
                to_frame="base_link",
            )
            rp = RigidTransform(
                translation=[0.5, -0.1, 0.15],
                rotation=self.GRIP_DOWN_R,  # 0.1 0.2
                from_frame=YK.r_tcp_frame,
                to_frame="base_link",
            )
            try:
                self.go_cartesian(
                    l_targets=[lp],
                    r_targets=[rp],
                    nwiggles=(2, 2),
                    rot=(0.3, 0.3),
                    removej6jump=True,
                )
            except:
                logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.go_pose_plan(lp, rp)
    
    def partial_pull_apart(self, dist, slide_left = True,slide_right=True,tilt=False, layout_nicely=False):
        '''
        Similar to pull_apart, but with a low z and keeps track of current coords, not centered. 
        '''
        logger.debug(f"Partial pull apart with distance {dist}")
        MAX_DIST = 0.57
        # back off a little
        self.go_delta(l_trans=[0,0.05,0.05], r_trans=[0,-0.05,0.05])
        
        self.sync()
        # initialize grip types according to need
        left_point = self.y.left.get_pose().translation[1] 
        right_point = self.y.right.get_pose().translation[1]

        average_x = np.mean([self.y.left.get_pose().translation[0], self.y.right.get_pose().translation[0]])
        # l_init = RigidTransform(
        #     translation=[0, left_point, 0.1],
        #     from_frame=YK.l_tcp_frame,
        #     to_frame="base_link",
        # )
        # r_init = RigidTransform(
        #     translation=[0, right_point, 0.1],
        #     from_frame=YK.l_tcp_frame,
        #     to_frame="base_link",
        # )
        # self.go_cartesian(l_targets=l_init, r_targets=r_init)
        if slide_left:
            # logger.info("LEFT SLIDE")
            self.slide_gripper("left")
        else:
            # logger.info("LEFT CLOSE")
            self.close_gripper("left")
        if slide_right:
            # logger.info("RIGHT SLIDE")
            self.slide_gripper("right")
        else:
            # logger.info("RIGHT CLOSE")
            self.close_gripper("right")

        self.sync()

        left_exceed = max(dist+left_point - MAX_DIST, 0) # over 0.6
        right_exceed = -min(right_point-dist + MAX_DIST, 0) # less than -0.6 if over

        # logger.debug(left_exceed, right_exceed)
        # logger.debug(dist+left_point-left_exceed+right_exceed, right_point-dist+right_exceed-left_exceed)
        l_R = self.y.left.get_pose().rotation
        if l_R[:, 1].dot(np.array((1, 0, 0))) >0:  # if y axis points towards pos x
            l_tar_R = RigidTransform.z_axis_rotation(np.pi/2)@self.GRIP_DOWN_R
        else:
            l_tar_R = RigidTransform.z_axis_rotation(-np.pi/2)@self.GRIP_DOWN_R


        lp = RigidTransform(
            translation=[average_x if not layout_nicely else 0.25, min(dist+left_point-left_exceed+right_exceed, MAX_DIST), 0.15],
            rotation=l_tar_R,  # 0.45 (also below)
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        
        r_R = self.y.right.get_pose().rotation
        if r_R[:, 1].dot(np.array((1, 0, 0))) > 0:  # if y axis points towards pos x
            r_tar_R = RigidTransform.z_axis_rotation(np.pi/2)@self.GRIP_DOWN_R
        else:
            r_tar_R = RigidTransform.z_axis_rotation(-np.pi/2)@self.GRIP_DOWN_R
        rp = RigidTransform(
            translation=[average_x if not layout_nicely else 0.25, max(right_point-dist+right_exceed-left_exceed, -MAX_DIST), 0.15],
            rotation=r_tar_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        
        try:
            wiggle_rot = (0.6, 0.6)
            if dist < 0.2:
                wiggle_rot = (0, 0)
            # self.set_speed((0.2, np.pi*2/3))
            self.go_cartesian(
                l_targets=[lp],
                r_targets=[rp],
                nwiggles=(12, 12), #reduce nwiggles
                rot=wiggle_rot,
                removejumps=[5, 6, 4],
            )
            self.set_speed(self.default_speed)
        except:
            logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
            self.set_speed((.2,np.pi*2/3))
            self.go_pose_plan(lp, rp)
            self.set_speed(self.default_speed)
        self.sync()

        if tilt:
            #self.open_grippers()
            # self.sync()
            # lp.rotation = self.GRIP_SIDEWAYS_L
            # rp.rotation = self.GRIP_SIDEWAYS
            # self.go_cartesian(
            #     l_targets=[lp],
            #     r_targets=[rp],
            #     nwiggles=(1, 1),
            #     rot=(0.3, 0.3),
            #     removejumps=[5, 6, 4]
            # )
            def l_to_r(config):
                return np.array([-1, 1, -1, 1, -1, 1, 1]) * config
            OUT_POS_2_L = np.array([-0.97958402, -1.25103124,  0.89924877, -0.64772485,  2.23664466, 0.86638389, -1.29129557])
            OUT_POS_2_R = l_to_r(OUT_POS_2_L)
            self.go_configs(l_q=[OUT_POS_2_L], r_q=[OUT_POS_2_R])    
            self.sync()
        
        if layout_nicely:
            success=False
            target_x = 0.45
            while (not success and target_x > 0.3):
                try:
                    self.sync()
                    lp.translation[0] = target_x
                    rp.translation[0] = target_x
                    lp.translation[1] *= 0.3
                    rp.translation[1] *= 0.3
                    self.go_cartesian(
                        l_targets=[lp],
                        r_targets=[rp],
                        nwiggles=(1, 1),
                        rot=(0.3, 0.3),
                        removejumps=[5, 6, 4],
                        n_tries=5
                    )
                    self.sync()
                    success = True
                except:
                    target_x -= 0.05
            if dist > 0.05:
                self.release_cable("right")
                self.release_cable("left")

        self.sync()
        self.open_grippers()
        self.sync()
        logger.debug(self.y.left.get_pose().translation[1] - self.y.right.get_pose().translation[1])

    def _draw_mask(self, polygon_coords, shape, blur=None):
        image = np.zeros(shape=shape, dtype=np.float32)
        cv2.drawContours(image, [polygon_coords], 0, (1.0), -1)    
        if blur is not None:        
            image = cv2.GaussianBlur(image, (blur, blur), 0)    
        return image

    def get_left_reachability_mask(self, img):
        left_arm_mask_poly = (np.array([(0, 174), (0, 26), (24, 22), (48, 18), (96, 24), (148, 48), (186, 75), (231, 148), (236, 186)])*4.03125).astype(np.int32)
        left_arm_mask_poly -= np.array([[0, 28]])
        left_arm_reachability_mask = (self._draw_mask(left_arm_mask_poly, img.shape[:2])).astype(np.uint8)
        return left_arm_reachability_mask
    
    def get_right_reachability_mask(self, img):
        right_arm_mask_poly = (np.array([(30, 176), (35, 150), (70, 77), (96, 46), (160, 22), (255, 18), (255, 186)])*4.03125).astype(np.int32)
        right_arm_mask_poly -= np.array([[0, 28]])
        right_arm_reachability_mask = (self._draw_mask(right_arm_mask_poly, img.shape[:2])).astype(np.uint8)
        return right_arm_reachability_mask

    def reveal_endpoint(self, img, closest_to_pos=None, delta_multiplier=1.0):
        g = GraspSelector(img, self.cam.intrinsics, self.T_PHOXI_BASE)
        # blackout center of image
        img_cp = img.copy()
        img_cp = img_cp.color.data
        img_cp[650:] = 0
        img_blked = img_cp.copy()

        # left_arm_mask_poly = (np.array([(0, 174), (0, 36), (24, 24), (48, 18), (96, 24), (148, 48), (186, 75), (231, 148), (236, 186)])*4.03125).astype(np.int32)
        # right_arm_mask_poly = (np.array([(30, 176), (35, 150), (70, 77), (96, 52), (160, 26), (255, 22), (255, 186)])*4.03125).astype(np.int32)

        left_arm_reachability_mask = self.get_left_reachability_mask(img_cp)
        right_arm_reachability_mask = self.get_right_reachability_mask(img_cp)
        combined_mask = (left_arm_reachability_mask + right_arm_reachability_mask) > 0.0
        combined_mask = combined_mask.astype(np.uint8)

        kernel = np.ones((20, 20), np.uint8)
        mask_large = cv2.erode(combined_mask, kernel)

        kernel2 = np.ones((40, 40), np.uint8)
        mask_small = cv2.erode(mask_large, kernel2)

        new_mask = mask_large - mask_small
        new_mask[110:, 170:200] = 1.0
        new_mask[600:630, 170:] = 1.0
        new_mask[:, :170] = 0.0
        new_mask[630:, :] = 0.0
        new_mask = np.expand_dims(new_mask, axis=-1)
        new_mask = np.concatenate((new_mask, new_mask, new_mask), axis=-1)
        # import pdb; pdb.set_trace()
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

        if closest_to_pos is None:
            idx = random.randint(0, len(centroids)-1)
        else:
            idx = np.argmin(np.linalg.norm(np.array(closest_to_pos)[None, :] - centroids, axis=-1))
        grasp_point = centroids[idx][::-1]

        white_pixs = np.argwhere(img_blked[:, :, 0] > 150)
        distances = np.linalg.norm(white_pixs - grasp_point, axis=-1)
        grasp_point = white_pixs[np.argmin(distances)][::-1]

        plt.clf()
        plt.title("Endpoint reveal move point.")
        plt.imshow(img_cp)
        plt.scatter(grasp_point[0], grasp_point[1])
        plt.savefig('alg_output.png')

        arm_to_use = 'left' if left_of_workspace((grasp_point[1], grasp_point[0])) else 'right'
        logger.debug("arm to use: " + arm_to_use)
        self.home()
        self.sync()
        try:
            # if arm_to_use == 'left':
            #     self.go_delta(l_trans=[0, 0, 0.05])
            # else:
            #     self.go_delta(r_trans=[0, 0, 0.05])
            grasp, _ = g.single_grasp_closer_arm(grasp_point, 0.008, {"left": self.L_TCP, "right": self.R_TCP})
        except Exception:
            logger.debug("Single grasp failed, falling back to top down grasp")
            grasp = g.top_down_grasp(tuple(grasp_point), .008, self.L_TCP if arm_to_use == 'left' else self.R_TCP)
        #will sometimes go to a loc where there is no rope
        logging.info(f"Going for grasp pose {grasp.pose.translation}")

        grasp.pregrasp_dist = 0.05
        grasp.slide = True
        self.grasp(l_grasp=grasp, reid_grasp=True) if arm_to_use == 'left' else self.grasp(r_grasp=grasp, reid_grasp=True)
        
        try:
            if arm_to_use == 'left':
                self.go_delta(l_trans=[0, 0, 0.06])
            else:
                self.go_delta(r_trans=[0, 0, 0.06])
        except:
            logger.debug("Sad. Cannot go delta 0.06 up.")
        self.sync()

        action_magnitude = 0.15 * delta_multiplier
        action_dir = np.array([0.45, 0., 0.]) - grasp.pose.translation
        action_dir /= np.linalg.norm(action_dir)
        goal_pos = grasp.pose.translation + action_dir*action_magnitude
        if arm_to_use == 'right':
            rp1 = RigidTransform(
                translation=[goal_pos[0], goal_pos[1], 0.1],
                rotation=grasp.pose.rotation,
                from_frame=YK.r_tcp_frame,
                to_frame="base_link"
            )
            try:
                self.go_cartesian(
                    r_targets=[rp1],
                    nwiggles=(None, None),
                    rot=(None, None), removejumps=[6]
                )
            except:
                logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.go_pose_plan(r_target=rp1)
            self.open_grippers()
            time.sleep(self.GRIP_SLEEP_TIME)
            self.sync()
        else:
            lp1 = RigidTransform(
                translation=[goal_pos[0], goal_pos[1], 0.1],
                rotation=grasp.pose.rotation, #RigidTransform.x_axis_rotation(-.4)@
                from_frame=YK.l_tcp_frame,
                to_frame="base_link"
            )
            try:
                self.go_cartesian(
                    l_targets=[lp1],
                    nwiggles=(None, None),
                    rot=(None, None), removejumps=[6]
                )
            except:
                logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.go_pose_plan(l_target=lp1)
            self.open_grippers()
            time.sleep(self.GRIP_SLEEP_TIME)
            self.sync()

        self.release_cable(arm_to_use)
        time.sleep(self.GRIP_SLEEP_TIME)
        self.home()
        self.sync()
        self.open_arms()
        self.sync()

    def release_cable(self, arm_to_use):
        self.open_gripper(arm_to_use)
        self.sync()
        try:
            if arm_to_use == 'left':
                self.go_delta(l_trans=[0, 0, 0.05])
            else:
                self.go_delta(r_trans=[0, 0, 0.05])
        except:
            logger.debug("Sad. Cannot go delta 0.05 up.")
        self.sync()
        self.shake_right_J([.4], 4) if arm_to_use == 'right' else self.shake_left_J([.4], 4)
        self.sync()

    
    def perturb_point_knot_tye(self, img, point, g, delta_multiplier=1.0):
        # blackout center of image
        img_cp = img.copy()
        img_cp = img_cp.color.data
        img_cp[600:] = 0
        img_blked = img_cp.copy()

        grasp_point = point

        plt.clf()
        plt.title("Perturb point.")
        plt.imshow(img_cp)
        plt.scatter(grasp_point[0], grasp_point[1])
        # plt.show()
        plt.savefig('alg_output.png')

        arm_to_use = 'left' if left_of_workspace((grasp_point[1], grasp_point[0])) else 'right'
        logger.debug("arm to use: " + arm_to_use)
        self.home()
        self.sync()
        try:
            # if arm_to_use == 'left':
            #     self.go_delta(l_trans=[0, 0, 0.05])
            # else:
            #     self.go_delta(r_trans=[0, 0, 0.05])
            grasp, _ = g.single_grasp_closer_arm(grasp_point, 0.008, {"left": self.L_TCP, "right": self.R_TCP})
        except Exception:
            logger.debug("Single grasp failed, falling back to top down grasp")
            grasp = g.top_down_grasp(tuple(grasp_point), .008, self.L_TCP if arm_to_use == 'left' else self.R_TCP)
        logging.info(f"Going for grasp pose {grasp.pose.translation}")
        grasp.pregrasp_dist = 0.05
        grasp.gripper_pos = 0.012
        grasp.slide = True
        #changed reid_grasp to False
        self.grasp(l_grasp=grasp, reid_grasp=False) if arm_to_use == 'left' else self.grasp(r_grasp=grasp, reid_grasp=False)
        
        try:
            if arm_to_use == 'left':
                self.go_delta(l_trans=[0, 0, 0.05]*delta_multiplier)
            else:
                self.go_delta(r_trans=[0, 0, 0.05]*delta_multiplier)
        except:
            logger.debug("Sad. Cannot go delta 0.05 up.")
        self.sync()
        
        delta1 = np.random.uniform(-0.1, 0.1, size=1)[0]
        delta2 =  np.random.uniform(-0.1, 0.1, size=1)[0]

        if arm_to_use == 'right':
            rp1 = RigidTransform(
                translation=[grasp.pose.translation[0] + delta1, grasp.pose.translation[1] + delta2, 0.1],
                rotation=grasp.pose.rotation, #RigidTransform.x_axis_rotation(-.4)@
                from_frame=YK.r_tcp_frame,
                to_frame="base_link"
            )
            try:
                self.go_cartesian(
                    r_targets=[rp1],
                    nwiggles=(None, None),
                    rot=(None, None), removejumps=[6]
                )
            except:
                logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.go_pose_plan(r_target=rp1)
            self.open_grippers()
            time.sleep(self.GRIP_SLEEP_TIME)
            self.go_configs(r_q=[self.R_HOME_STATE])
            self.sync()
        else:
            lp1 = RigidTransform(
                translation=[grasp.pose.translation[0] + delta1, grasp.pose.translation[1] + delta2, 0.1],
                rotation=grasp.pose.rotation, #RigidTransform.x_axis_rotation(-.4)@
                from_frame=YK.l_tcp_frame,
                to_frame="base_link"
            )
            try:
                self.go_cartesian(
                    l_targets=[lp1],
                    nwiggles=(None, None),
                    rot=(None, None), removejumps=[6]
                )
            except:
                logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.go_pose_plan(l_target=lp1)
            self.open_grippers()
            time.sleep(self.GRIP_SLEEP_TIME)
            self.go_configs(l_q=self.L_HOME_STATE)
            self.sync()

        try:
            if arm_to_use == 'left':
                self.go_delta(l_trans=[0, 0, 0.05])
            else:
                self.go_delta(r_trans=[0, 0, 0.05])
        except:
            logger.debug("Sad. Cannot go delta 0.05 up.")
        self.sync()

        self.sync()
        time.sleep(self.GRIP_SLEEP_TIME)
        self.home()
        self.sync()
        

    def perturb_point(self, img, point, g, delta_multiplier=1.0):
        # blackout center of image
        img_cp = img.copy()
        img_cp = img_cp.color.data
        img_cp[600:] = 0
        img_blked = img_cp.copy()

        grasp_point = point

        plt.clf()
        plt.title("Perturb point.")
        plt.imshow(img_cp)
        plt.scatter(grasp_point[0], grasp_point[1])
        # plt.show()
        plt.savefig('alg_output.png')

        arm_to_use = 'left' if left_of_workspace((grasp_point[1], grasp_point[0])) else 'right'
        logger.debug("arm to use: " + arm_to_use)
        self.home()
        self.sync()
        try:
            # if arm_to_use == 'left':
            #     self.go_delta(l_trans=[0, 0, 0.05])
            # else:
            #     self.go_delta(r_trans=[0, 0, 0.05])
            grasp, _ = g.single_grasp_closer_arm(grasp_point, 0.008, {"left": self.L_TCP, "right": self.R_TCP})
        except Exception:
            logger.debug("Single grasp failed, falling back to top down grasp")
            grasp = g.top_down_grasp(tuple(grasp_point), .008, self.L_TCP if arm_to_use == 'left' else self.R_TCP)
        logging.info(f"Going for grasp pose {grasp.pose.translation}")
        grasp.pregrasp_dist = 0.05
        grasp.slide = True
        self.grasp(l_grasp=grasp, reid_grasp=True) if arm_to_use == 'left' else self.grasp(r_grasp=grasp, reid_grasp=True)
        
        try:
            if arm_to_use == 'left':
                self.go_delta(l_trans=[0, 0, 0.05]*delta_multiplier)
            else:
                self.go_delta(r_trans=[0, 0, 0.05]*delta_multiplier)
        except:
            logger.debug("Sad. Cannot go delta 0.05 up.")
        self.sync()
        
        delta1 = np.random.uniform(-0.1, 0.1, size=1)[0]
        delta2 =  np.random.uniform(-0.1, 0.1, size=1)[0]

        if arm_to_use == 'right':
            rp1 = RigidTransform(
                translation=[grasp.pose.translation[0] + delta1, grasp.pose.translation[1] + delta2, 0.1],
                rotation=grasp.pose.rotation, #RigidTransform.x_axis_rotation(-.4)@
                from_frame=YK.r_tcp_frame,
                to_frame="base_link"
            )
            try:
                self.go_cartesian(
                    r_targets=[rp1],
                    nwiggles=(None, None),
                    rot=(None, None), removejumps=[6]
                )
            except:
                logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.go_pose_plan(r_target=rp1)
            self.open_grippers()
            time.sleep(self.GRIP_SLEEP_TIME)
            self.go_configs(r_q=[self.R_HOME_STATE])
            self.sync()
        else:
            lp1 = RigidTransform(
                translation=[grasp.pose.translation[0] + delta1, grasp.pose.translation[1] + delta2, 0.1],
                rotation=grasp.pose.rotation, #RigidTransform.x_axis_rotation(-.4)@
                from_frame=YK.l_tcp_frame,
                to_frame="base_link"
            )
            try:
                self.go_cartesian(
                    l_targets=[lp1],
                    nwiggles=(None, None),
                    rot=(None, None), removejumps=[6]
                )
            except:
                logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
                self.go_pose_plan(l_target=lp1)
            self.open_grippers()
            time.sleep(self.GRIP_SLEEP_TIME)
            self.go_configs(l_q=self.L_HOME_STATE)
            self.sync()

        try:
            if arm_to_use == 'left':
                self.go_delta(l_trans=[0, 0, 0.05])
            else:
                self.go_delta(r_trans=[0, 0, 0.05])
        except:
            logger.debug("Sad. Cannot go delta 0.05 up.")
        self.sync()

        self.sync()
        time.sleep(self.GRIP_SLEEP_TIME)
        self.home()
        self.sync()
        self.open_arms()
        self.sync()
        
    
    #not sure that this works
    def get_grasper_cartesian_coords(self, g):
        left_point = g.ij_to_point(left_coords).data
        right_point = g.ij_to_point(right_coords).data

    def shake_R(
        self,
        which_arm,
        rot=np.zeros(3),
        num_shakes=5,
        trans=[0, 0, 0],
        speed=(1.5, 2000),
    ):
        if which_arm == "left":
            self.shake_left_R(rot, num_shakes, trans, speed)
        elif which_arm == "right":
            self.shake_right_R(rot, num_shakes, trans, speed)
    

    def flatten(self):
        # sweep extra stuff in center of workspace out of the way
        self.open_gripper("left")
        self.sync()
        self.go_delta_single("left", [0, 0.2, 0], reltool=False)  # move left
        self.sync()
        self.close_gripper("left")
        wrist_rot = self.y.right.get_joints()[6]
        self.go_configs(l_q=[self.L_HOME_STATE], r_q=[
                        [1.04735588, -0.79394698, -0.83759745, 0.03420543, 0.19812349, 1.13718923, wrist_rot]])
        self.sync()
        # TODO make this not a delta
        self.go_delta_single("left", [0, -0.3, -0.15], reltool=False)
        self.sweep(-0.2, 0.3, 0.6, "left")
        self.sync()

        # rotation of the wrist depends on the holding angle of the rope
        right_pos = self.y.right.get_pose()
        right_y = right_pos.rotation[:, 1]
        if right_y.dot(np.array((-1, 0, 0))) > 0:
            # want y axis to point in -y of world
            right_target_rot = self.GRIP_DOWN_R
        else:
            right_target_rot = RigidTransform.z_axis_rotation(
                np.pi)@self.GRIP_DOWN_R
        # bring down onto table
        rp0 = RigidTransform(
            translation=[0.4, -0.1, 0.15],
            rotation=right_target_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link"
        )
        self.go_delta_single('left', [0, 0, .2], reltool=False, linear=False)
        self.sync()
        _, config = self.yk.ik(
            right_pose=rp0, right_qinit=self.R_HOME_STATE, solve_type='Manipulation1')
        self.go_config_plan(r_q=config, table_z=.05)

        # left grip get up from sweep position
        lp0 = RigidTransform(
            translation=[0.5, .2, 0.2],
            rotation=self.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link"
        )
        self.go_pose('left', lp0, linear=False)
        self.sync()

        # pinch left gripper near right
        right_grip_pos = self.y.right.get_pose()
        new_right_pos = RigidTransform(translation=right_grip_pos.translation,
                                       rotation=RigidTransform.x_axis_rotation(
                                           .2)@right_grip_pos.rotation,
                                       from_frame=YK.r_tcp_frame, to_frame=YK.base_frame)
        left_grip_trans = right_grip_pos.translation + [0, 0.015, -.01]
        left_grip_rot = RigidTransform.x_axis_rotation(
            -np.pi/5) @ right_grip_pos.rotation
        left_grip_pos = RigidTransform(
            translation=left_grip_trans, rotation=left_grip_rot, from_frame=YK.l_tcp_frame, to_frame=YK.base_frame)

        self.y.right.goto_pose(new_right_pos, speed=self.speed)
        self.sync()
        grasp = Grasp(left_grip_pos, pregrasp_dist=.1,
                      grip_open_dist=.1, gripper_pos=.02)
        self.grasp(l_grasp=grasp)
        self.sync()

        # get pixel coords of left gripper position
        p = Point(left_grip_pos.translation, frame=YK.base_frame)
        T_CAM_BASE = self.T_PHOXI_BASE
        T_BASE_CAM = T_CAM_BASE.inverse()
        intr = self.cam.intrinsics
        left_grip_pixel_coord = intr.project(T_BASE_CAM*p)
        # right cage grasp out
        self.slide_gripper("right")
        self.sync()
        rp1 = RigidTransform(
            translation=[0.3, -0.65, 0.15],
            rotation=RigidTransform.x_axis_rotation(-.4)@right_target_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link"
        )
        try:
            self.go_cartesian(
                r_targets=[rp1],
                nwiggles=(None, 10),
                rot=(None, 0.3)
            )
        except:
            logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
            self.go_pose_plan(r_target=rp1)
        self.sync()
        self.open_grippers()
        time.sleep(self.GRIP_SLEEP_TIME)
        self.go_delta_single('left', [0, 0, .15])
        self.home()
        self.sync()
        return (left_grip_pixel_coord[1], left_grip_pixel_coord[0])

    def other_arm(self, arm):
        if arm == "left":
            return "right"
        return "left"

    def flatten_new(self,which_arm='right',distance_slid=1):
        multiplier = 1 if which_arm=='right' else -1
        right_arm = which_arm=='right'
        self.sync()
        self.close_gripper(which_arm)
        if distance_slid<.5:
            self.open_gripper(self.other_arm(which_arm))
        else:
            self.move_gripper(self.other_arm(which_arm),.0016)
        # pull out from cable a bit
        self.set_speed((.2, np.pi))
        self.go_delta_single(self.other_arm(which_arm), [0, multiplier*.1, 0], linear=True)
        l_out = RigidTransform(
            translation=[0.3, multiplier*0.65, 0.15],
            rotation=RigidTransform.x_axis_rotation(multiplier*.4)@ self.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame if right_arm else YK.l_tcp_frame,
            to_frame="base_link"
        )
        self.sync()
        self.set_speed((.3,np.pi))
        # move far off to the left to pull the endpoint away
        self.go_pose(self.other_arm(which_arm), l_out, linear=False)
        self.open_gripper(self.other_arm(which_arm))
        self.sync()
        self.shake_left_J([.2], 1) if right_arm else self.shake_right_J([.2], 1)
        self.sync()
        self.go_delta_single(self.other_arm(which_arm), [0, -multiplier*0.1, .1], linear=False)
        self.close_gripper(self.other_arm(which_arm))
        right_pose = self.y.right.get_pose() if right_arm else self.y.left.get_pose()
        #version with draping below
        l_pos = RigidTransform(
            translation=right_pose.translation+[0, multiplier*0.02, 0],
            rotation=RigidTransform.x_axis_rotation(-multiplier*np.pi/2) @
            RigidTransform.z_axis_rotation(np.pi/2) @
            self.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame if right_arm else YK.r_tcp_frame,
            to_frame="base_link"
        )
        #move next to right
        try:
            if right_arm:
                self.go_pose_plan(l_target=l_pos.copy(), table_z=.06)
            else:
                self.go_pose_plan(r_target=l_pos.copy(), table_z=.06)
        except PlanningException:
            self.go_pose(self.other_arm(which_arm),l_pos.copy(),linear=False)
        #go down
        l_pos.translation[2] -= .12
        self.go_pose(self.other_arm(which_arm),l_pos.copy())
        self.sync()
        #go in
        l_pos.translation[1] -= multiplier*.15
        self.go_pose(self.other_arm(which_arm),l_pos.copy())
        # self.go_pose_plan_single(self.other_arm(which_arm),l_pos.copy(),mode='Distance')
        l_pos.translation += [.17,.0,0]
        l_pos.rotation = RigidTransform.x_axis_rotation(-multiplier*np.pi/4) @ RigidTransform.z_axis_rotation(multiplier*np.pi/4) @ l_pos.rotation
        # self.go_pose(self.other_arm(which_arm),l_pos.copy(),linear=False)
        self.go_pose_plan_single(self.other_arm(which_arm),l_pos.copy(),mode='Distance')
        self.sync()
        #move left
        l_pos.translation=[.6,multiplier*0.4,.4]
        l_pos.rotation=RigidTransform.z_axis_rotation(multiplier*np.pi/4)@l_pos.rotation
        if right_arm:
            self.go_pose_plan(l_target=l_pos.copy(),table_z=.06)
        else:
            self.go_pose_plan(r_target=l_pos.copy(),table_z=.06)
        self.sync()
        l_pos.rotation=self.GRIP_DOWN_R
        l_pos.translation=[.5,multiplier*.4,.2]
        if right_arm:
            left_joint_angles = [-0.75634088, -0.51258261,  1.76810061, -0.17745042,  1.04217203,  0.20622845, 0.34592705]
            self.go_configs(l_q=[left_joint_angles])
        else:
            right_joint_angles = [ 0.57006435, -0.54717644, -1.75185679, -0.24886964, -0.24470128,  0.64385228, 1.79749885]
            self.go_configs(r_q=[right_joint_angles])
        self.sync()
        self.shake_left_J([.4,.4,.25],4,ji=[5,4,3]) if right_arm else self.shake_right_J([.4,.4,.25],4,ji=[5,4,3])
        #then drag the knot on the table
        down_pose = RigidTransform(
            translation=[.4,multiplier*-.2,.07],
            rotation=RigidTransform.x_axis_rotation(-multiplier*np.pi/2)@right_pose.rotation,
            from_frame=YK.r_tcp_frame if right_arm else YK.l_tcp_frame,
            to_frame="base_link"
        )
        self.slide_gripper(which_arm)
        if which_arm=='left':
            self.go_pose_plan(l_target=down_pose,table_z=.055)
        else:
            self.go_pose_plan(r_target=down_pose,table_z=.055)
        self.sync()
        self.shake_right_J([.4], 4) if right_arm else self.shake_left_J([.4], 4)
        self.sync()
        self.open_gripper(which_arm)
        self.go_delta_single(which_arm, [0, 0, .05])
        self.sync()
        self.shake_right_J([.3], 2) if right_arm else self.shake_left_J([.3], 2)
        self.set_speed(self.default_speed)

        p = Point(down_pose.translation, frame=YK.base_frame)
        T_CAM_BASE = self.T_PHOXI_BASE
        T_BASE_CAM = T_CAM_BASE.inverse()
        intr = self.cam.intrinsics
        left_grip_pixel_coord = intr.project(T_BASE_CAM*p)
        return left_grip_pixel_coord[::-1]

    def rotate_to_visible(self,deg):
        '''
        Assumes both grippers are closed on a segment. Rotate the segment by 45 deg
        to "flatten" a vertical knot.
        Arguments: radians
        ''' 
        self.close_grippers()
        self.go_cartesian(
        rot=(deg, deg),
        removejumps=[],
        wiggle_joint=5,N=30, mode='Speed')
        
        
    def slide(
        self,
        sliding_arm,
        distance,
        done_fn,
        end_in_pull_arm=True,
        slide_direction=-1,
        do_initialization_motion=True,
        distance_chunk=0.09, slide_speed=(.25, 1.5*np.pi),
        check_endpoint_buffer=.4#this param is the length of cable at the end that we consider endpoint detections valid
    ):
        """
        Assumes that the rope is already grasped inside 'sliding_arm'
        executes a sliding motion which pulls the rope approx 'distance'
        meters through 'sliding_arm'
        if end_in_pull_arm is true, the motion ends with
        the pulling arm grabbing the cable
        if 'slide_direction' is positive, slide in the positive y direction of the sliding gripper,
        if negative slide in the negative y direction of the sliding gripper
        """
        self.sync()
        pulling_arm = "left" if sliding_arm == "right" else "right"
        # angle tilting away from the other gripper
        tilt_angle = np.deg2rad(90)
        # angle tilting away from the base to let the cable droop some
        if sliding_arm == "left":
            tilt_angle *= -1
        slide_pos = np.array((0.4, 0, 0.5))#was .45 z height
        wrist_rot = 0 if slide_direction < 0 else np.pi
        if sliding_arm == "left":
            wrist_rot = np.pi - wrist_rot
        upwards_angle = -np.deg2rad(40)#was 20
        slide_R = RigidTransform.y_axis_rotation(
            np.pi / 2 + wrist_rot + upwards_angle
        ) @ RigidTransform.x_axis_rotation(np.pi + tilt_angle)
        slide_H = getH(slide_R, slide_pos, sliding_arm)
        backoff_dist = 0.015  # distance to go away from cable
        forward_dist = 0.03  # distance to go into cable
        pull_backoff = RigidTransform(
            np.eye(3),
            [0, 0, -backoff_dist],
            self.get_frame(pulling_arm),
            self.get_frame(pulling_arm),
        )
        pull_pos = slide_pos
        pull_R = slide_R @ RigidTransform.x_axis_rotation(np.pi)
        if wrist_rot > 0:
            pull_R = pull_R @ RigidTransform.z_axis_rotation(np.pi)
        pull_H = getH(pull_R, pull_pos, pulling_arm) * pull_backoff
        # init_H is slightly above the other gripper so that it pushes other cables somewhat out of the way
        sideways_backoff = .13
        pull_init_H = (
            RigidTransform(
                translation=[0, sideways_backoff if sliding_arm=='right' else -sideways_backoff, 0.05],
                from_frame=self.yk.base_frame,
                to_frame=self.yk.base_frame,
            )
            * pull_H
        )
        multiplier = -1
        if do_initialization_motion:
            self.close_gripper(pulling_arm)
            # move the sliding arm near the pose and shake to get cable down
            self.go_pose(
                sliding_arm,
                getH(
                    RigidTransform.z_axis_rotation(-np.pi / 2 + wrist_rot + (np.pi if sliding_arm=='left' else 0))
                    @ self.GRIP_DOWN_R,
                    [0.4, 0, 0.4],
                    sliding_arm,
                ),
                linear=False,
            )
            self.sync()
            if sliding_arm=='left':
                curj=self.y.left.get_joints()
                slide_joints = self.yk.ik(left_pose=slide_H,solve_type='Manipulation1')[0]
                curj[6]=slide_joints[6]
                self.go_configs(l_q=[curj])
            else:
                curj = self.y.right.get_joints()
                slide_joints = self.yk.ik(right_pose=slide_H,solve_type='Manipulation1')[1]
                curj[6]=slide_joints[6]
                self.go_configs(r_q=[curj])
            self.sync()
            self.shake_R(sliding_arm, rot=[0, 0, 0],
                         num_shakes=4, trans=[0, 0, .1])
            self.sync()
            if sliding_arm=='left':
                self.go_config_plan(l_q=slide_joints)
            else:
                self.go_config_plan(r_q=slide_joints)
            # self.go_pose_plan(r_target=slide_H,mode='Distance') if sliding_arm=='right' else self.go_pose_plan(l_target=slide_H,mode='Distance')
            self.sync()
            # go to initial pose plus backoff distance
            self.go_pose(pulling_arm, pull_init_H, linear=False)
            self.sync()
            self.go_delta_single(pulling_arm,trans=[0,0,sideways_backoff],reltool=True)
            self.go_pose(pulling_arm, pull_H, linear=True)
            self.sync()
            self.open_gripper(pulling_arm)
            self.set_speed(slide_speed)
            time.sleep(self.GRIP_SLEEP_TIME)
            # move into the cable
            self.go_delta_single(
                pulling_arm, [0, 0, backoff_dist + forward_dist], reltool=True
            )
            multiplier = -1 if pulling_arm=='left' else 1
            self.go_delta_single(pulling_arm, [0, multiplier*0.019, 0], reltool=True)
            self.sync()
            self.close_gripper(pulling_arm)
            time.sleep(self.GRIP_SLEEP_TIME)
            self.move_gripper(sliding_arm,.001)
            # self.go_delta_single(
            #     pulling_arm, [0, multiplier*distance_chunk, 0], reltool=True)
        for i in range(math.ceil(distance / distance_chunk)):
            # at the beginning of this loop, the sliding gripper is open and pulling gripper is closed
            # pull
            distance_slid = i*distance_chunk
            self.go_delta_single(
                pulling_arm, [0, multiplier*distance_chunk, 0], reltool=True)
            self.sync()
            # sleep for the rope to settle before querying done-ness
            time.sleep(2)
            found_knot, found_endpoint = done_fn(self, sliding_arm)
            if distance - distance_slid>check_endpoint_buffer:
                if found_endpoint:
                    logger.debug(f"Found endpoint but haven't slid far enough yet, ignoring. Distance to go: {distance - distance_slid}")
                    found_endpoint=False
            # if found_endpoint and found_knot:
            #     raise RuntimeError(
            #         "Found both knot and endpoint in sliding stop cond")
            if found_endpoint:
                self.close_gripper(sliding_arm)
                self.slide_gripper(pulling_arm)
                self.go_delta_single(
                    pulling_arm, [0, -multiplier*distance_chunk, 0], reltool=True)
                self.sync()
                #confirm endpoint detection
                self.move_gripper(sliding_arm,.001)
                self.move_gripper(pulling_arm,.0003)
                self.sync()
                jostle_dist=.05#distance to pull endpoint
                self.go_delta_single(
                    pulling_arm, [0, multiplier*jostle_dist, 0], reltool=True)
                self.sync()
                _, found_endpoint = done_fn(self, sliding_arm)
                if found_endpoint:
                    self.go_delta_single(
                        pulling_arm, [0, -multiplier*jostle_dist, 0], reltool=True)
                else:
                    self.go_delta_single(
                        pulling_arm, [0, -multiplier*(-distance_chunk+jostle_dist), 0], reltool=True)
                self.sync()
                #end confirm endpoint detection
            elif found_knot:
                # at beginning, pulling arm closed, sliding arm sliding
                # confirm knot detection
                self.shake_R(sliding_arm,num_shakes=5,trans=[.15,0,0])
                # self.slide_gripper(pulling_arm)
                # self.sync()
                # self.close_gripper(sliding_arm)
                # self.sync()
                # self.go_delta_single(sliding_arm, [0.1, 0, 0], reltool=True)
                # self.sync()
                # self.shake_left_J([2], num_shakes=3, ji=[5], speed=(5, 2000)) if sliding_arm=="left" else self.shake_right_J([2], num_shakes=3, ji=[5], speed=(5, 2000))
                # self.sync()
                # self.go_delta_single(sliding_arm, [-0.1, 0], reltool=True)
                # self.sync()
                # self.slide_gripper(sliding_arm)
                logger.debug("Shaking after finding knot")
                self.sync()
                time.sleep(3)
                found_knot,_=done_fn(self,sliding_arm)
                if found_knot:
                    # back off just a tad to give the knot some slack to make regrasping easier
                    slack_dist = .05
                    self.close_gripper(sliding_arm)
                    self.slide_gripper(pulling_arm)
                    self.go_delta_single(
                        pulling_arm, [0, -multiplier*(distance_chunk-slack_dist), 0], reltool=True)
                    self.sync()
                    self.close_gripper(pulling_arm)
                    self.move_gripper(sliding_arm,.001)
                    self.sync()
                    self.go_delta_single(
                        pulling_arm, [0, -multiplier*slack_dist, 0], reltool=True)
                    self.sync()
            if found_knot or found_endpoint:
                break
            self.close_gripper(sliding_arm)
            time.sleep(self.SLIDE_SLEEP_TIME)
            self.slide_gripper(pulling_arm)
            self.go_delta_single(
                pulling_arm, [0, -multiplier*distance_chunk, 0], reltool=True)
            self.sync()
            # close one gripper and make the other one in slide mode
            self.close_gripper(pulling_arm)
            time.sleep(self.SLIDE_SLEEP_TIME)
            self.move_gripper(sliding_arm,.001)
            time.sleep(self.SLIDE_SLEEP_TIME)
        # end in the correct arm
        self.set_speed(self.default_speed)
        if end_in_pull_arm:
            self.close_gripper(pulling_arm)
            self.open_gripper(sliding_arm)
            time.sleep(self.SLIDE_SLEEP_TIME)
            self.sync()
        else:
            self.close_gripper(sliding_arm)
            time.sleep(self.SLIDE_SLEEP_TIME)
            self.sync()
        
        return found_knot, found_endpoint

    def sweep(self, starty, goaly, sweep_x, which_arm, vertical=False):
        if vertical:
            z_height = 0.03
            start_R = goal_R = down_R = self.GRIP_DOWN_R
            sweep_x = sweep_x - 0.1
            pullback = 0.1
        else:
            z_height = 0.065
            tilt_angle = np.deg2rad(15)
            SIDEWAYS_R = RigidTransform.rotation_from_axis_angle(
                np.array((0, np.pi / 2 + tilt_angle, 0))
            )
            start_R = goal_R = down_R = SIDEWAYS_R
            pullback = 0.05
        start_T = np.array((sweep_x, starty, z_height + pullback))
        down_T = np.array((sweep_x, starty, z_height))
        goal_T = np.array((sweep_x, goaly, z_height))
        if which_arm == "left":
            startH = getH(start_R, start_T, "left")
            downH = getH(down_R, down_T, "left")
            goalH = getH(goal_R, goal_T, "left")
            self.go_config_plan(l_q=[-2.52213872, -1.59179269,  1.7195163,
                                     -0.44049843, -2.25671956,  1.19914075, -1.69936962], table_z=.05)
            # self.go_pose_plan(startH, None, table_z=0.05)
            self.open_gripper('left')
            self.y.left.goto_pose(downH, speed=self.speed, linear=True)
            self.sync()
            self.y.left.goto_pose(goalH, speed=self.speed, linear=True)
            self.sync()
        elif which_arm == "right":
            startH = getH(start_R, start_T, "right")
            downH = getH(down_R, down_T, "right")
            goalH = getH(goal_R, goal_T, "right")
            self.go_pose_plan(None, startH, table_z=0.05)
            self.y.right.goto_pose(downH, speed=self.speed, linear=True)
            self.y.right.goto_pose(goalH, speed=self.speed, linear=True)
            self.sync()

    def wiggle(self, jpath, rot, nwiggles, ji):
        """
        applies nwiggles wiggles along the path by distributing them evenly
        """

        def getT(jtraj):
            return np.linspace(0, 1, jtraj.shape[0])

        jpath = np.array(jpath.copy())
        if len(jpath) == 0:
            return []

        def clip(j):
            return max(
                min(j, self.yk.left_joint_lims[1][5]
                    ), self.yk.left_joint_lims[0][5]
            )

        jtraj = jpath[:, ji]
        jspline = inter.CubicSpline(getT(jpath), jtraj)
        wigglespline = inter.CubicSpline(
            np.linspace(0, 1, 2 * nwiggles + 2),
            [0] + nwiggles * [-rot / 2.0, rot / 2.0] + [0],
        )
        newjtraj = [
            clip(jspline(t) + wigglespline(t)) for t in np.linspace(0, 1, len(jpath))
        ]
        jpath[:, ji] = newjtraj
        return jpath

    def go_configs(self, l_q=[], r_q=[], together=False):
        """
        moves the arms along the given joint trajectories
        l_q and r_q should be given in urdf format as a np array
        """
        # import pdb; pdb.set_trace()
        if together and len(l_q) == len(r_q):
            self.y.move_joints_sync(l_q, r_q, speed=self.speed)
            print("move joints")
        else:
            if len(r_q) > 0:
                self.y.right.move_joints_traj(
                    r_q, speed=self.speed, zone="z10")
            if len(l_q) > 0:
                self.y.left.move_joints_traj(
                    l_q, speed=self.speed, zone="z10")
            # if len(r_q) > 0:
            #     self.y.right.move_joints_traj(
            #         r_q, speed=self.speed, zone="z10")

    def grasp_single(self, which_arm, grasp):
        if which_arm == 'left':
            return self.grasp(l_grasp=grasp)
        elif which_arm == 'right':
            return self.grasp(r_grasp=grasp)


    def grasp(self, l_grasp:Grasp=None, r_grasp:Grasp=None, sliding=False, topdown=False, bend_elbow=True, reid_grasp=False):
        """
        Carries out the grasp motions on the desired end poses. responsible for doing the
        approach to pre-grasp as well as the actual grasp itself.
        attempts
        both arguments should be a Grasp object
        """
        # self.open_grippers()
        if topdown:
            # self.open_gripper('right')
            # self.sync()
            # self.sync()
            # self.open_grippers()
            # self.sync()
            l_grasp.pose.translation[2] = 0.04 #CHANGE HERE: 0.05 -> 0.04
            r_grasp.pose.translation[2] = 0.05 
            l_grasp.pregrasp_dist = 0.25
            r_grasp.pregrasp_dist = 0.25
            l_rot_vector = l_grasp.pose.rotation[:3, :3] @ [1, 0, 0]
            l_rot_angle = np.arctan2(l_rot_vector[1], l_rot_vector[0])
            r_rot_vector = r_grasp.pose.rotation[:3, :3] @ [1, 0, 0]
            r_rot_angle = np.arctan2(r_rot_vector[1], r_rot_vector[0])
            # l_grasp.pose.rotation[:2, :2] = [[np.cos(l_rot_angle), -np.sin(l_rot_angle)],
            #                                  [np.sin(l_rot_angle), np.cos(l_rot_angle)]]
            # l_grasp.pose.rotation[2, :2] = [0, 0]
            # r_grasp.pose.rotation[:2, :2] = [[np.cos(r_rot_angle), -np.sin(r_rot_angle)],
            #                                  [np.sin(r_rot_angle), np.cos(r_rot_angle)]]
            # r_grasp.pose.rotation[2, :2] = [0, 0]
            # l_grasp.pose.rotation[:3, 2] = [0, 0, -1]
            # r_grasp.pose.rotation[:3, 2] = [0, 0, -1]
            l_grasp.pose.rotation = RigidTransform.z_axis_rotation(l_rot_angle) @ self.GRIP_DOWN_R
            r_grasp.pose.rotation = RigidTransform.z_axis_rotation(r_rot_angle) @ self.GRIP_DOWN_R
        if reid_grasp:
            if l_grasp is not None and l_grasp.slide:
                l_grasp.gripper_pos = 0.018
            if r_grasp is not None and r_grasp.slide:
                r_grasp.gripper_pos = 0.018
        l_waypoints = []
        r_waypoints = []
        l_pre, r_pre = None, None
        if l_grasp is not None:
            self.close_gripper('left')
            l_pre = l_grasp.compute_pregrasp()
            l_waypoints.append(l_pre)
        if r_grasp is not None:
            self.close_gripper('right')
            r_pre = r_grasp.compute_pregrasp()
            r_waypoints.append(r_pre)

        try:
            self.go_cartesian(l_waypoints, r_waypoints, fine=False)
        except:
            self.go_pose_plan(l_pre, r_pre, fine=False, table_z=0.05)
        self.sync()
        time.sleep(self.GRIP_SLEEP_TIME)
        # move in
        if l_grasp is not None and r_grasp is not None:
            self.go_cartesian(l_targets=[l_grasp.compute_gripopen(topdown=topdown)],r_targets=[r_grasp.compute_gripopen(topdown=topdown)],N=3,mode='Distance')
        elif l_grasp is not None:
            self.go_cartesian(l_targets=[l_grasp.compute_gripopen(topdown=topdown)],N=3,mode='Distance')
        elif r_grasp is not None:
            self.go_cartesian(r_targets=[r_grasp.compute_gripopen(topdown=topdown)],N=3,mode='Distance')
        self.sync()
        # # put grippers in right position
        if l_grasp is not None:
            self.y.left.move_gripper(l_grasp.gripper_pos)
        if r_grasp is not None:
            self.y.right.move_gripper(r_grasp.gripper_pos)
        self.sync()
        time.sleep(self.SLIDE_SLEEP_TIME)
        
        if l_grasp is not None and r_grasp is not None:
            self.go_cartesian(l_targets=[l_grasp.pose],r_targets=[r_grasp.pose],N=10,removejumps=[6],mode='Distance')
        elif l_grasp is not None:
            self.go_cartesian(l_targets=[l_grasp.pose],N=10,removejumps=[6],mode='Distance')
        elif r_grasp is not None:
            self.go_cartesian(r_targets=[r_grasp.pose],N=10,removejumps=[6],mode='Distance')
        self.sync()
        # grasp, changed to call slide_gripper function instead of as instance of yumiarm
        self.y.left.close_gripper()
        self.y.right.close_gripper()
        self.sync()
        time.sleep(self.GRIP_SLEEP_TIME)
        if l_grasp is not None:
            self.slide_gripper(self.y.left) if l_grasp.slide and not reid_grasp else self.y.left.close_gripper()
        if r_grasp is not None:
            self.slide_gripper(self.y.right) if r_grasp.slide and not reid_grasp else self.y.right.close_gripper()
        time.sleep(self.GRIP_SLEEP_TIME)
        self.sync()

        if bend_elbow:
            curr_left = self.y.left.get_joints()
            curr_right = self.y.right.get_joints()
            curr_left[3] += 0.04 #0.08
            curr_right[3] += 0.04 #0.08
            self.go_configs(l_q=[curr_left], r_q=[curr_right])    
            self.sync()

    # def grasp(self, l_grasp:Grasp=None, r_grasp:Grasp=None, sliding=False):
    #     """
    #     Carries out the grasp motions on the desired end poses. responsible for doing the
    #     approach to pre-grasp as well as the actual grasp itself.
    #     attempts
    #     both arguments should be a Grasp object
    #     """
    #     l_waypoints = []
    #     r_waypoints = []
    #     l_pre, r_pre = None, None
    #     if l_grasp is not None:
    #         self.close_gripper('left')
    #         l_pre = l_grasp.compute_pregrasp()
    #         l_waypoints.append(l_pre)
    #     if r_grasp is not None:
    #         self.close_gripper('right')
    #         r_pre = r_grasp.compute_pregrasp()
    #         r_waypoints.append(r_pre)
    #     try:
    #         self.go_cartesian(l_waypoints, r_waypoints, fine=False)
    #     except:
    #         self.go_pose_plan(l_pre, r_pre, fine=False, table_z=0.05)
    #     self.sync()
    #     # move in
    #     if l_grasp is not None and r_grasp is not None:
    #         self.go_cartesian(l_targets=[l_grasp.compute_gripopen()],r_targets=[r_grasp.compute_gripopen()],N=3,mode='Distance')
    #     elif l_grasp is not None:
    #         self.go_cartesian(l_targets=[l_grasp.compute_gripopen()],N=3,mode='Distance')
    #     elif r_grasp is not None:
    #         self.go_cartesian(r_targets=[r_grasp.compute_gripopen()],N=3,mode='Distance')
    #     self.sync()
    #     # put grippers in right position
    #     if l_grasp is not None:
    #         self.y.left.move_gripper(l_grasp.gripper_pos)
    #     if r_grasp is not None:
    #         self.y.right.move_gripper(r_grasp.gripper_pos)
    #     self.sync()
    #     time.sleep(self.SLIDE_SLEEP_TIME)
    #     if l_grasp is not None and r_grasp is not None:
    #         self.go_cartesian(l_targets=[l_grasp.pose],r_targets=[r_grasp.pose],N=10,removejumps=[6],mode='Distance')
    #     if l_grasp is not None:
    #         self.go_cartesian(l_targets=[l_grasp.pose],N=10,removejumps=[6],mode='Distance')
    #     if r_grasp is not None:
    #         self.go_cartesian(r_targets=[r_grasp.pose],N=10,removejumps=[6],mode='Distance')
    #     self.sync()
    #     # grasp
    #     if l_grasp is not None:
    #         self.y.left.close_gripper() if not sliding else self.y.left.slide_gripper()
    #     if r_grasp is not None:
    #         self.y.right.close_gripper() if not sliding else self.y.left.slide_gripper()
    #     time.sleep(self.GRIP_SLEEP_TIME)

    def sync(self, timeout=15): # used to be 15
        logger.debug("Starting sync")
        self.y.left.sync(timeout=timeout)
        # time.sleep(1)
        self.y.right.sync(timeout=timeout)
        # time.sleep(1)
        logger.debug("Ending sync")
        #time.sleep(1) # small fix bc sync not working


    def refine_states(self, left=True, right=True, t_tol=0.02, r_tol=0.2):
        """
        attempts to move the arms into a better joint configuration without deviating much at the end effector pose
        """
        l_q = self.y.left.get_joints()
        r_q = self.y.right.get_joints()
        l_traj, r_traj = self.yk.refine_state(
            l_state=l_q, r_state=r_q, t_tol=t_tol, r_tol=r_tol
        )
        if not left:
            l_traj = []
        if not right:
            r_traj = []
        self.go_configs(l_q=l_traj, r_q=r_traj)
    def shake_J(self,which_arm,rot,num_shakes,speed=(1.5,20),ji=[5]):
        if which_arm=='left':
            self.shake_left_J(rot,num_shakes,speed,ji)
        else:
            self.shake_right_J(rot,num_shakes,speed,ji)
    def shake_left_J(self, rot, num_shakes, speed=(1.5, 2000), ji=[5]):
        curj = self.y.left.get_joints()
        forj = curj.copy()
        backj = curj.copy()
        for i in range(len(ji)):
            joint = ji[i]
            forj[joint] = min(
                curj[joint] + rot[i] / 2.0, self.yk.left_joint_lims[1][joint]
            )
            backj[joint] = max(
                curj[joint] - rot[i] / 2.0, self.yk.left_joint_lims[0][joint]
            )
        traj = []
        for i in range(num_shakes):
            traj.append(backj)
            traj.append(forj)
        traj.append(curj)
        self.y.left.move_joints_traj(traj, speed=speed, zone="z10")

    def shake_right_J(self, rot, num_shakes, speed=(1.5, 2000), ji=[5]):
        curj = self.y.right.get_joints()
        forj = curj.copy()
        backj = curj.copy()
        for i in range(len(ji)):
            joint = ji[i]
            forj[joint] = min(
                curj[joint] + rot[i] / 2.0, self.yk.right_joint_lims[1][joint]
            )
            backj[joint] = max(
                curj[joint] - rot[i] / 2.0, self.yk.right_joint_lims[0][joint]
            )
        traj = []
        for i in range(num_shakes):
            traj.append(backj)
            traj.append(forj)
        traj.append(curj)
        self.y.right.move_joints_traj(traj, speed=speed, zone="z10")

    def shake_left_R(self, rot, num_shakes, trans=[0, 0, 0], speed=(1.5, 2000)):
        """
        Shake the gripper by translating by trans and rotating by rot about the wrist
        rot is a 3-vector in axis-angle form (the magnitude is the amount)
        trans is a 3-vector xyz translation
        arms is a list containing the YuMiArm objects to shake (for both, pass in both left and right)

        """
        # assumed that translation is small so we can just toggle back and forth between poses
        old_l_tcp = self.yk.l_tcp
        old_r_tcp = self.yk.r_tcp
        rot = np.array(rot)
        trans = np.array(trans)
        self.yk.set_tcp(None, None)
        cur_state = self.y.left.get_joints()
        curpose, _ = self.yk.fk(qleft=cur_state)
        R_for = RigidTransform(
            translation=trans / 2,
            rotation=RigidTransform.rotation_from_axis_angle(rot / 2.0),
            from_frame=self.yk.l_tcp_frame,
            to_frame=self.yk.l_tcp_frame,
        )
        R_back = RigidTransform(
            translation=-trans / 2,
            rotation=RigidTransform.rotation_from_axis_angle(-rot / 2.0),
            from_frame=self.yk.l_tcp_frame,
            to_frame=self.yk.l_tcp_frame,
        )
        target_for = curpose * R_for
        target_back = curpose * R_back
        target_for_q, _ = self.yk.ik(
            left_qinit=cur_state, left_pose=target_for)
        target_back_q, _ = self.yk.ik(
            left_qinit=cur_state, left_pose=target_back)
        if np.linalg.norm(target_for_q - target_back_q) > 3:
            logger.debug("aborting shake action, no smooth IK found between shake poses")
            self.yk.set_tcp(old_l_tcp, old_r_tcp)
            return
        traj = []
        for i in range(num_shakes):
            traj.append(target_for_q)
            traj.append(target_back_q)
        traj.append(cur_state)
        self.y.left.move_joints_traj(traj, speed=speed, zone="z10")
        self.yk.set_tcp(old_l_tcp, old_r_tcp)

    def shake_right_R(self, rot, num_shakes, trans=[0, 0, 0], speed=(1.5, 2000)):
        """
        Shake the gripper by translating by trans and rotating by rot about the wrist
        rot is a 3-vector in axis-angle form (the magnitude is the amount)
        trans is a 3-vector xyz translation
        arms is a list containing the YuMiArm objects to shake (for both, pass in both left and right)

        """
        # assumed that translation is small so we can just toggle back and forth between poses
        old_l_tcp = self.yk.l_tcp
        old_r_tcp = self.yk.r_tcp
        rot = np.array(rot)
        trans = np.array(trans)
        self.yk.set_tcp(None, None)
        cur_state = self.y.right.get_joints()
        _, curpose = self.yk.fk(qright=cur_state)
        R_for = RigidTransform(
            translation=trans / 2,
            rotation=RigidTransform.rotation_from_axis_angle(rot / 2.0),
            from_frame=self.yk.r_tcp_frame,
            to_frame=self.yk.r_tcp_frame,
        )
        R_back = RigidTransform(
            translation=-trans / 2,
            rotation=RigidTransform.rotation_from_axis_angle(-rot / 2.0),
            from_frame=self.yk.r_tcp_frame,
            to_frame=self.yk.r_tcp_frame,
        )
        target_for = curpose * R_for
        target_back = curpose * R_back
        _, target_for_q = self.yk.ik(
            right_qinit=cur_state, right_pose=target_for)
        _, target_back_q = self.yk.ik(
            right_qinit=cur_state, right_pose=target_back)
        if np.linalg.norm(target_for_q - target_back_q) > 3:
            logger.debug("aborting shake action, no smooth IK found between shake poses")
            self.yk.set_tcp(old_l_tcp, old_r_tcp)
            return
        traj = []
        for i in range(num_shakes):
            traj.append(target_for_q)
            traj.append(target_back_q)
        traj.append(cur_state)
        self.y.right.move_joints_traj(traj, speed=speed, zone="z100")
        self.yk.set_tcp(old_l_tcp, old_r_tcp)

    def lay_out(self):
        "Action performed after shaking to spread out cable for ease of perception"
        self.sync()
        self.go_configs(
            r_q=[
                [
                    2.60970532,
                    -0.80275964,
                    -1.44774217,
                    0.1211927,
                    -1.86754771,
                    0.44692943,
                    3.12923212,
                ]
            ]
        )
        self.sync()
        self.go_configs(
            r_q=[
                [
                    1.96777151,
                    -0.20877838,
                    -2.05070608,
                    0.394132,
                    -1.9460316,
                    -0.78239412,
                    3.07636438,
                ]
            ]
        )
        self.sync()
        self.go_configs(
            r_q=[
                [
                    1.96759521,
                    -0.20877071,
                    -2.83814276,
                    0.1245243,
                    -2.08828115,
                    -1.53245147,
                    3.04104075,
                ]
            ]
        )
        self.sync()
        # rp0 = RigidTransform(translation=[.55, 0, .2], rotation=self.GRIP_SIDEWAYS,
        #                     from_frame=YK.r_tcp_frame, to_frame='base_link')
        # rp1 = RigidTransform(translation=[.55, .2, .4], rotation=self.GRIP_SIDEWAYS,
        #                     from_frame=YK.r_tcp_frame, to_frame='base_link')
        # rp2 = RigidTransform(translation=[.4, 1, .4], rotation=self.GRIP_SIDEWAYS,
        #                     from_frame=YK.r_tcp_frame, to_frame='base_link')
        # rp3 = RigidTransform(translation=[.4, -.2, .3], rotation=self.GRIP_SIDEWAYS,
        #                     from_frame=YK.r_tcp_frame, to_frame='base_link')
        # self.go_pose('right', rp0, linear=False)
        # self.sync()
        # self.go_pose('right', rp1, linear=False)
        # self.sync()
        # self.go_pose('right', rp2, linear=False)
        # self.sync()
        # self.go_pose('right',rp3,linear=False)
        # self.sync()
        self.open_gripper("right")
        self.sync()
        time.sleep(self.GRIP_SLEEP_TIME)
        # self.go_delta_single('right',[0,0,-.1],reltool=True)
