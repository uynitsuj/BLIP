import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
from autolab_core import Box,Point
import matplotlib.pyplot as plt
import cv2
from fcvision.vision_utils import closest_nonzero_pt
import time

SPOOL_POS = np.array([0.161, -0.249, 0.136])  # TODO measure this
CRANK_RAD = .068
NOTCH_RAD = .06#radius the arm goes away from the centerpoint when placing the rope in
NOTCH_RAD_OUT = 0.16
PLATEUA_ANG = 2 * np.pi / 16

def detect_spool_plateaus(iface: Interface):
    dimg = iface.take_image()
    pc = iface.T_PHOXI_BASE*iface.cam.intrinsics.deproject(dimg.depth)
    xy_crop_size = .2
    crop_box = Box(SPOOL_POS-[xy_crop_size, xy_crop_size, .01],
                   SPOOL_POS+[xy_crop_size, xy_crop_size, .01],pc.frame)
    pc_crop,_ = pc.box_mask(crop_box)
    print(pc_crop.num_points)
    # crop_box is a pointcloud closely cropped to the spool prior
    cropped_dimg=iface.cam.intrinsics.project_to_image(iface.T_PHOXI_BASE.inverse()*pc_crop)
    plt.imshow(cropped_dimg._data)
    plt.show()
    bin_image = np.zeros_like(cropped_dimg._data,dtype=np.uint8)
    bin_image[cropped_dimg._data!=0]=255
    contours, _= cv2.findContours(bin_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_KCOS)
    contour_lens=[cv2.arcLength(c,True) for c in contours]
    del_ids=np.where((np.array(contour_lens)>100) | (np.array(contour_lens)<60))[0]
    contours = list(contours)
    for id in np.flip(del_ids):
        del contours[id]
    contour_lens=[cv2.arcLength(c,True) for c in contours]
    plateau_centers=[]
    for c in contours:
        M=cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        pt = closest_nonzero_pt(cropped_dimg._data.squeeze(),(cy,cx))
        depth = cropped_dimg[pt[0],pt[1]]
        center_3d = iface.cam.intrinsics.deproject_pixel(depth,Point(np.array((cx,cy)),frame=iface.cam.intrinsics.frame))
        center_3d_base = iface.T_PHOXI_BASE*center_3d
        plateau_centers.append(center_3d_base._data)
    return plateau_centers

def compute_notch_centers(plateau_centers):
    # Rotate all the plateau_centers to find notch position.
    s, c = np.sin(PLATEUA_ANG), np.cos(PLATEUA_ANG)
    new_plateau_centers = [np.squeeze(pc) - np.array(SPOOL_POS) for pc in plateau_centers]

    rotated_plateau_centers = []
    for npc in new_plateau_centers:
        xnew = npc[0] * c - npc[1] * s
        ynew = npc[0] * s + npc[1] * c

        # rescale radius.
        theta = np.arctan2(ynew, xnew)

        xnew = NOTCH_RAD * np.cos(theta) + SPOOL_POS[0]
        ynew = NOTCH_RAD * np.sin(theta) + SPOOL_POS[1]
        znew = npc[2] + SPOOL_POS[2]

        rotated_plateau_centers.append([xnew, ynew, znew])

    return rotated_plateau_centers

def compute_gripper_rot_notch(notch_pos, cable_y_aligned=True):
    y = SPOOL_POS - notch_pos
    y[2]=0
    y /= np.linalg.norm(y)
    z = np.array([0, 0, -1])
    x = np.cross(y, z)

    return RigidTransform.rotation_from_axes(x, y, z)

def slide_gripper_out_pos(notch_pos):
    new_notch_pos = notch_pos - np.array(SPOOL_POS)
    theta = np.arctan2(new_notch_pos[1], new_notch_pos[0])

    xnew = NOTCH_RAD_OUT * np.cos(theta) + SPOOL_POS[0]
    ynew = NOTCH_RAD_OUT * np.sin(theta) + SPOOL_POS[1]
    znew = notch_pos[2]
    return np.array([xnew, ynew, znew])

def init_endpoint_orientation(iface:Interface,sliding_hand = 'right'):
    '''
    This function gets called immediately after sliding has terminated and found an endpoint
    '''
    if sliding_hand!='right':
        #in this case we need to re-do the position of the left hand to be behind the right
        #1 open left gripper
        iface.open_gripper('left')
        iface.close_gripper('right')
        time.sleep(iface.GRIP_SLEEP_TIME)
        #2 move left gripper out and back
        switchover_dist=.019*2#distance to travel over the right gripper
        y_dir = iface.y.left.get_pose().rotation[:,1]
        if y_dir.dot((-1,0,0))>0:
            withdraw_dist=.1
            iface.go_delta_single('left',[0,switchover_dist,-withdraw_dist],reltool=True)
            iface.sync()
            #3 rotate the wrist into the right position 
            p=iface.y.left.get_pose()
            p.rotation=p.rotation@RigidTransform.z_axis_rotation(np.pi)
            iface.go_pose('left',p,linear=False)
            iface.go_delta_single('left',[0,0,withdraw_dist],reltool=True)
        else:
            iface.go_delta_single('left',[0,-switchover_dist,0],reltool=True)
    iface.sync()
    iface.close_gripper('left')
    time.sleep(iface.GRIP_SLEEP_TIME)
    orient_speed = (.1,np.pi)
    #first try pulling back the endpoint as far as it can go
    extra_slack_dist=.005
    iface.set_speed(orient_speed)
    iface.y.left.move_gripper(.0005)
    iface.sync()
    iface.y.right.move_gripper(.0015)
    iface.sync()
    time.sleep(iface.SLIDE_SLEEP_TIME)
    pullback_dist=.04
    iface.go_delta_single('left',[0,-pullback_dist,0],reltool=True)
    iface.sync()
    iface.close_gripper('right')
    iface.slide_gripper('left')
    iface.go_delta_single('left',[0,pullback_dist-extra_slack_dist,0],reltool=True)
    iface.sync()
    #then do the nudging motion to get the orientation correct
    #pull right arm away from the other arm
    iface.go_delta_single('right',[0,0,-.02],reltool=True,linear=True)
    iface.sync()
    iface.close_gripper('left')
    time.sleep(iface.SLIDE_SLEEP_TIME)
    iface.open_gripper('right')
    iface.set_speed(iface.default_speed)
    time.sleep(iface.GRIP_SLEEP_TIME)
    iface.go_delta_single('right',[0,0,-.05],reltool=True)
    iface.sync()
    iface.home()
    iface.sync()

def execute_spool(iface:Interface):
    def r_p(trans, rot=Interface.GRIP_DOWN_R):
        return RigidTransform(translation=trans, rotation=rot, from_frame=YK.r_tcp_frame, to_frame=YK.base_frame)
    def l_p(trans, rot=Interface.GRIP_DOWN_R):
        return RigidTransform(translation=trans, rotation=rot, from_frame=YK.l_tcp_frame, to_frame=YK.base_frame)
    iface.sync()
    insert_speed = (.2,np.pi/2)
    plateaus = detect_spool_plateaus(iface)
    notches = compute_notch_centers(plateaus)

    min_dist = float('inf')
    notch_pos = notches[0]
    for notch in notches:
        dist = np.linalg.norm(notch_pos - iface.y.left.get_pose().translation)
        if dist < min_dist:
            notch_pos = notch

    iface.set_speed(insert_speed)
    notch_rot = compute_gripper_rot_notch(notch_pos)
    notch_pose = l_p(notch_pos + np.array([0, 0, 0.03]), notch_rot)
    iface.go_pose('left',notch_pose,linear=True)
    iface.sync()
    down_dist=-.03

    notch_pose = l_p(notch_pos + np.array([0, 0, down_dist]), notch_rot)
    
    iface.go_cartesian(l_targets=[notch_pose])
    iface.sync()
    iface.slide_gripper('left')

    new_notch_pos = slide_gripper_out_pos(notch_pos)
    notch_pose = r_p(new_notch_pos + np.array([0, 0, down_dist]), notch_rot)
    iface.go_pose('left', notch_pose, linear=True)
    iface.sync()

    new_l_rot = RigidTransform.z_axis_rotation(np.pi/6)@\
            RigidTransform.x_axis_rotation(-np.pi/3) @ notch_pose.rotation
    notch_pose = r_p(new_notch_pos + np.array([0, 0, down_dist]), new_l_rot)
    iface.go_pose('left', notch_pose, linear=True)
    iface.sync()
    iface.set_speed(iface.default_speed)

    # here is the actual circular spooling motion
    iface.close_gripper('right')
    circ_waypoints = []
    rot = RigidTransform.y_axis_rotation(.25)@iface.GRIP_DOWN_R
    start_pos = SPOOL_POS + CRANK_RAD * \
        np.array((1, 0, 0)) + np.array((0, 0, -.02))
    start_pose = r_p(start_pos, rot)
    iface.go_pose('right', start_pose, linear=True)
    for th in np.linspace(0, 2*np.pi, 30):
        pos = SPOOL_POS + CRANK_RAD * \
            np.array((np.cos(th), np.sin(th), 0)) + np.array((0, 0, -.01))
        rt = r_p(pos, rot)
        circ_waypoints.append(rt)
    _, path = iface.yk.interpolate_cartesian_waypoints(r_waypoints=circ_waypoints,
                        N=10, r_qinit=iface.y.right.get_joints())
    iface.y.left.move_gripper(.0005)
    iface.sync()
    for loop in range(10):
        iface.go_configs(r_q=np.array(path))
        iface.sync(timeout=10)

def test_spool():
    SPEED = (.4, 4*np.pi)
    iface = Interface("1703005", ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                      ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
    print(iface.y.left.get_joints())
    iface.home()
    iface.sync()
    input("Enter to close grippers")
    iface.close_grippers()
    dotprod = 1
    # this function is passed into slide so it knows when to stop
    def done_fn(iface_nonlocal):
        return False,True
    # slide until done
    iface.slide(sliding_arm='right', distance=3, done_fn=done_fn, slide_direction=dotprod,
                do_initialization_motion=True, end_in_pull_arm=False)
    init_endpoint_orientation(iface)
    execute_spool(iface)

def test_slide_motion():
    SPEED = (.2, 4*np.pi)  # 0.5, 6*np.pi
    iface = Interface("1703005", ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                      ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
    iface.open_grippers()
    

if __name__ == '__main__':
    test_spool()