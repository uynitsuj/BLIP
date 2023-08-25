from shutil import move
from untangling.utils.interface_rws import Interface
import numpy as np
from untangling.utils.tcps import *
import time
import matplotlib.pyplot as plt
import os
from untangling.utils.interface_rws import getH

def move_to_slide_pos(self,sliding_arm,
        slide_direction=-1,
        do_initialization_motion=True,
        distance_chunk=0.1,
        slide_speed=(.25, 1.5*np.pi),
        shake_dist=0.1):
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
    slide_R = RigidTransform.y_axis_rotation(np.pi / 2 + wrist_rot + upwards_angle) @ RigidTransform.x_axis_rotation(np.pi + tilt_angle)
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
        self.sync()
        multiplier = -1 if pulling_arm=='left' else 1
        self.go_delta_single(pulling_arm, [0, multiplier*0.019, 0], reltool=True)
        self.sync()
        self.close_gripper(pulling_arm)
        time.sleep(self.GRIP_SLEEP_TIME)
        self.move_gripper(sliding_arm,.001)
        self.go_delta_single(pulling_arm, [0, multiplier*distance_chunk, 0], reltool=True)
        self.sync()

if __name__ == "__main__":
    #keep_going for simple loop or straight, knot for stopping, endpoint for endpoint
    results_dir = "./config1_2_again"
    os.makedirs(results_dir,exist_ok=True)
    SPEED = (0.4, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    iface.open_arms()
    iface.open_grippers()
    iface.close_grippers()
    iface.sync()
    init=False
    if init:
        iface.open_arms()
        iface.close_grippers()
        iface.sync()
        iface.slide_grippers()
    start_ind = 0
    num_pics = 2 - start_ind
    for i in range(start_ind,num_pics+start_ind):
        input(f"Press enter to take image {i}")
        img = iface.take_image()
        # plt.imshow(img.color._data)
        # plt.show()
        r_pos = iface.y.left.get_pose()
        depth_img = img.depth._data
        color_img = img.color._data.copy()
        # plt.imsave(results_dir + f"/color_{i}.png", color_img)
        np.save(results_dir + f"/depth_{i}.npy", depth_img)
        np.save(results_dir + f"/color_{i}.npy", color_img)
        print("SAVED")
