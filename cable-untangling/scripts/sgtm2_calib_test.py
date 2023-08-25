# 3 lines of detectron imports
from codecs import IncrementalDecoder
from pickle import FALSE
from turtle import done, left
import analysis as loop_detectron
from untangling.utils.grasp import GraspSelector, GraspException
from untangling.utils.interface_rws import Interface
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
import numpy as np
from untangling.utils.tcps import *
# from untangling.utils.circle_BFS import trace, trace_white
from cable_tracing.tracers.simple_uncertain_trace import trace
# from untangling.utils.cable_tracing.tracers.simple_uncertain_trace import trace
#from untangling.slide import FCNNetworkStopCond, knot_in_hand, endpoint_in_hand
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
plt.set_loglevel('info')
# plt.style.use('seaborn-darkgrid')

torch.cuda.empty_cache()

vis = True
alg_output_fig = 'alg_output.png'
topdown_grasps = False
left_arm_mask_poly = (np.array([(0, 174), (0, 36), (24, 24), (48, 18), (96, 24), (148, 48), (186, 75), (231, 148), (236, 186)])*4.03125).astype(np.int32)
right_arm_mask_poly = (np.array([(30, 176), (35, 150), (70, 77), (96, 52), (160, 26), (255, 22), (255, 186)])*4.03125).astype(np.int32)
image_save_path = '/home/justin/yumi/cable-untangling/scripts/live_rollout_sgtm2_calib_test/'
img_dims = np.array([772, 1032])
total_num_observations = 0

def show_img():
    if vis:
        plt.show()
    else:
        plt.savefig(alg_output_fig)

def draw_mask(polygon_coords, shape, blur=None):
    image = np.zeros(shape=shape, dtype=np.float32)
    cv2.drawContours(image, [polygon_coords], 0, (1.0), -1)    
    if blur is not None:        
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    return image

left_arm_reachability_mask = draw_mask(left_arm_mask_poly, img_dims)
right_arm_reachability_mask = draw_mask(right_arm_mask_poly, img_dims)
overall_reachability_mask = (left_arm_reachability_mask + right_arm_reachability_mask) > 0
overall_reachability_mask[:, 900:] = 0.0
overall_reachability_mask[:, :200] = 0.0

def closest_valid_point(color, depth, xy):
    valid_y, valid_x = np.nonzero((color[:, :, 0] > 100) * (depth[:, :, 0] > 0))
    pts = np.vstack((valid_x, valid_y)).T
    return pts[np.argmin(np.linalg.norm(pts - np.array(xy)[None, :], axis=-1))]

parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--debug',
    help="Print more statements",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.INFO,
)

args = parser.parse_args()

logging.root.setLevel(args.loglevel)
logging.basicConfig(level=args.loglevel)
logger = logging.getLogger("Untangling")

detectron = os.path.dirname(os.path.abspath(__file__)) + "/../../detectron2_repo"
sys.path.insert(0,detectron)

def take_and_save_image(iface):
    global total_num_observations
    total_num_observations += 1
    img = iface.take_image()
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    np.save(osp.join(image_save_path, cur_time + '.npy'), img._data)
    return img

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

_FINISHED = False

def get_grasp_points_from_endpoint(endpoint, img):
    # trace from endpoint on image
    img[-130:, ...] = 0
    thresh_img = np.where(img[:,:,:3] > 100, 255, 0)

    full_img = np.concatenate((thresh_img, img[:,:,3:]), axis=-1)
    path, _ = trace(full_img, endpoint, None, exact_path_len=6, viz=vis)
    if path is None:
        return tuple(np.array([endpoint[1], endpoint[0]]).astype(np.int32)), True
    corrected_pick_point = closest_valid_point(img[:, :, :3], img[:, :, 3:], path[-3][::-1]) # takes in x,y and outputs x,y
    return tuple(corrected_pick_point.astype(np.int32)), False

def run_endpoint_separation_incremental_cartesian_moves(endpoints, img, iface:Interface, thresh=0.05, increment=0.1, incremental=False):
    '''
    Pull apart 0.3m, and while detectron detects no knots, keep pulling apart by specified increments 
    until knot detected or ceiling=0.6m reached. 
    
    Tilt not implemented because regrasping should take care of it.
    '''
    MAX_DISTANCE = 0.5
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE,grabbing_endpoint=True)
    endpoints_tuples = [tuple(l[0].tolist()) for l in endpoints]
    endpt1, _ = get_grasp_points_from_endpoint(endpoints_tuples[0], img._data)
    endpt2, _ = get_grasp_points_from_endpoint(endpoints_tuples[1], img._data)
    # endpoints_tuples = endpoints
    l_grasp,r_grasp=g.double_grasp(endpt1,endpt2,.008,.008,iface.L_TCP,iface.R_TCP) #0.005 each
    iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp,topdown=topdown_grasps)
    iface.set_speed((.4,5))
    def state_estimation():
        picture = take_and_save_image(iface)
        detectron_out,viz = loop_detectron.predict(picture.color._data, thresh)
        plt.imshow(viz)
        show_img()
        if len(detectron_out) > 0:
            return True, (True,True)
        else:
            return False, (True,True)
    iface.sync()
    dist = 0.3
    if not incremental:
        increment = MAX_DISTANCE - dist

    iface.pull_apart(dist, slide_left=False, slide_right=False) # initial centered pull apart
    iface.sync()
    while dist+increment <= MAX_DISTANCE:
        if incremental:
            see_knot, _ = state_estimation()
            if see_knot:
                iface.home()
                iface.sync()
                iface.open_grippers()
                iface.open_arms()
                return False
        logger.debug(dist)
        iface.partial_pull_apart(increment, slide_left=False, slide_right=False)
        iface.sync()
        dist += increment

    cur_left_pos = iface.y.left.get_joints()
    cur_right_pos = iface.y.right.get_joints()
    def l_to_r(config):
        return np.array([-1, 1, -1, 1, -1, 1, 1]) * config
    OUT_POS_2_L = np.array([-0.82856348, -1.36882858, -0.5320617 , -0.23686631,  2.62157876,
        1.29712763, -1.7537732 ])
    OUT_POS_2_R = l_to_r(OUT_POS_2_L)
    OUT_POS_2_L[6] = cur_left_pos[6]
    OUT_POS_2_R[6] = cur_right_pos[6]

    l_R = iface.y.left.get_pose().rotation
    if l_R[:, 1].dot(np.array((1, 0, 0))) > 0:  # if y axis points towards pos x
        OUT_POS_2_L[6] -= np.pi
    
    r_R = iface.y.right.get_pose().rotation
    if r_R[:, 1].dot(np.array((1, 0, 0))) > 0:  # if y axis points towards pos x
        OUT_POS_2_R[6] -= np.pi

    iface.go_configs(l_q=[OUT_POS_2_L], r_q=[OUT_POS_2_R])    
    iface.sync()

    if incremental:
        see_knot, _ = state_estimation()
        if see_knot:
            iface.home()
            iface.sync()
            iface.open_grippers()
            iface.open_arms()
            return False

    iface.home()
    iface.sync()
    iface.open_grippers()
    iface.open_arms()

    if incremental:
        return True
    return False

def find_clear_area(image, iface, side=None):
    image = image.copy()
    image[650:,:] = image.max()
    vis_image = image.copy()
    # side reflects the region you're placing in and the opposite gripper is used
    if side == 'left':
        image[:, 500:] = image.max()
        right_arm_reachability = iface.get_right_reachability_mask(image)
        image[right_arm_reachability == 0] = image.max()
    elif side == 'right':
        image[:, :550] = image.max()
        left_arm_reachability = iface.get_left_reachability_mask(image)
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

def run_endpoint_separation_incremental(endpoints, img, iface:Interface, thresh=0.05,
                                                        increment=0.1, incremental=False):
    '''
    Pull apart 0.3m, and while detectron detects no knots, keep pulling apart by specified increments 
    until knot detected or ceiling=0.6m reached. 
    
    Tilt not implemented because regrasping should take care of it.
    '''
    MAX_DISTANCE = 0.5
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE,grabbing_endpoint=True)
    endpoints_tuples = [tuple(l[0].tolist()) for l in endpoints]
    endpt1, on_endpt1 = get_grasp_points_from_endpoint(endpoints_tuples[0], img._data)
    endpt2, on_endpt2 = get_grasp_points_from_endpoint(endpoints_tuples[1], img._data)

    # plt.imshow(img.color._data * overall_reachability_mask[:, :, None])
    # plt.show()

    # if overall_reachability_mask[endpt1[1], endpt1[0]] == 0 or \
    #     overall_reachability_mask[endpt2[1], endpt2[0]] == 0:
    #     raise Exception("Endpoints too far away to grasp, falling back to endpoint reveal.")

    plt.clf()
    plt.scatter(*endpt1)
    plt.scatter(*endpt2)
    plt.imshow(img.color._data * overall_reachability_mask[:, :, None])
    show_img()
    # endpoints_tuples = endpoints
    iface.home()
    iface.sync()
    
    l_grasp,r_grasp=g.double_grasp(endpt1,endpt2,.0085,.0085,iface.L_TCP,iface.R_TCP,slide0=on_endpt1, slide1=on_endpt2) #0.005 each
    #TODO: REMOVE THIS FROM INFO ONCE DEBUGGED
    logging.info(f"l_grasp: {l_grasp.pose.translation}")
    logging.info(f"r_grasp: {r_grasp.pose.translation}")
    l_grasp.pregrasp_dist, r_grasp.pregrasp_dist = 0.05, 0.05
    iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp,topdown=topdown_grasps,reid_grasp=True)
    iface.set_speed((.4,5))
    def state_estimation():
        picture = take_and_save_image(iface)
        detectron_out,viz = loop_detectron.predict(picture.color._data, thresh)
        plt.imshow(viz)
        show_img()
        if len(detectron_out) > 0:
            return True, (True,True)
        else:
            return False, (True,True)

    def l_to_r(config):
            return np.array([-1, 1, -1, 1, -1, 1, 1]) * config

    def get_clear_area(color_img, side='left'):
        try:
            g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE,grabbing_endpoint=True)
            clear_point = find_clear_area(color_img, iface, side=side)
            # plt.imshow(self.img._data[..., :3].astype(np.uint8))
            # plt.scatter(*clear_point[::-1])
            # plt.show()
            clear_point = g.ij_to_point(clear_point[::-1]).data
            clear_point[2] = 0.1
        except Exception as e:
            raise e
            clear_point = None
        return clear_point

    def return_to_center():
        iface.home()
        iface.sync()
        img = take_and_save_image(iface)
        color_img = img.color._data
        color_img = np.array(color_img, dtype=np.uint8)[:, :, 0]

        default_right_pose = [0.35, 0.15, 0.1]
        clear_point_right_arm = get_clear_area(color_img, side='left')
        clear_point_right_arm = default_right_pose if clear_point_right_arm is None else clear_point_right_arm
        if np.linalg.norm(np.array(default_right_pose) - np.array(clear_point_right_arm)) > 0.2:
            clear_point_right_arm = default_right_pose
        rp = RigidTransform(
            translation=clear_point_right_arm,
            rotation=iface.GRIP_DOWN_R,  # 0.1 0.2
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

        iface.home()
        iface.sync()
        try:
            iface.go_cartesian(
                r_targets=[rp]
            )
        except:
            logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
            iface.go_pose_plan(r_target=rp)
        iface.sync()
        iface.release_cable("right")
        iface.sync()
        iface.go_config_plan(r_q=iface.R_HOME_STATE)
        iface.sync()
        time.sleep(1)

        img = take_and_save_image(iface)
        color_img = img.color._data
        default_left_pose = [0.35, -0.15, 0.1]
        color_img = img.color._data
        color_img = np.array(color_img, dtype=np.uint8)[:, :, 0]
        clear_point_left_arm = get_clear_area(color_img, side='right')
        clear_point_left_arm = default_left_pose if clear_point_left_arm is None else clear_point_left_arm
        if np.linalg.norm(np.array(default_left_pose) - np.array(clear_point_left_arm)) > 0.2:
            clear_point_left_arm = default_left_pose

        lp = RigidTransform(
            translation=clear_point_left_arm,
            rotation=iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        try:
            iface.go_cartesian(
                l_targets=[lp]
            )
        except:
            logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
            iface.go_pose_plan(l_target=lp)
        iface.sync()
        iface.release_cable("left")
        iface.sync()
        iface.go_config_plan(l_q=iface.L_HOME_STATE)
        iface.sync()

    # def return_to_center():
    #     iface.home()
    #     # OUT_POS_3_L = np.array([-0.88178888, -1.39528389,  0.96595596, -0.68538608,  1.40693242,
    #         # 1.31388158, -2.21221708])
    #         #np.array([ 0.20857435, -1.0969679 ,  0.5486748 , -0.96094356,  0.49384195,
    #         #0.7884078 , -1.70679424]) #np.array([-0.24853087, -1.0878531 , -1.11695033, -0.77183069,  2.21226595,
    #         #1.37551019, cur_left_pos[6]]) #np.array([-0.82856348, -1.36882858, -0.5320617 , -0.23686631,  2.62157876, 1.29712763,  cur_left_pos[6]])
    #     # OUT_POS_3_R = l_to_r(OUT_POS_3_L)
    #     # OUT_POS_3_R[6] = cur_right_pos[6]
    #     # iface.go_configs(l_q=[OUT_POS_3_L], r_q=[OUT_POS_3_R])
    #     # iface.sync()

    #     lp = RigidTransform(
    #         translation=[0.25, 0.05, 0.1],
    #         rotation=iface.GRIP_DOWN_R,
    #         from_frame=YK.l_tcp_frame,
    #         to_frame="base_link",
    #     )
    #     rp = RigidTransform(
    #         translation=[0.25, -0.05, 0.1],
    #         rotation=iface.GRIP_DOWN_R,  # 0.1 0.2
    #         from_frame=YK.r_tcp_frame,
    #         to_frame="base_link",
    #     )
    #     iface.home()
    #     iface.sync()
    #     try:
    #         iface.go_cartesian(
    #             l_targets=[lp],
    #             r_targets=[rp]
    #         )
    #     except:
    #         logger.debug("Couldn't compute smooth cartesian path, falling back to planner")
    #         iface.go_pose_plan(lp, rp)

    # cur_left_pos = iface.y.left.get_joints()
    # cur_right_pos = iface.y.right.get_joints()
    # CENTER_L = np.array([-1.41362668, -1.20112134,  1.07746664,  0.39039282,  1.47596742, 1.22066055, cur_left_pos[6]])
    # CENTER_R = l_to_r(CENTER_L)
    # CENTER_R[6] = cur_right_pos[6]

    def check_knot(end=False):
        if not incremental:
            return False
        see_knot, _ = state_estimation()
        if see_knot:
            if not end:
                # return to center
                return_to_center()
                iface.sync()
                iface.open_grippers()
                iface.sync()
                iface.open_arms()
                iface.sync()
            return True
        return False

    iface.pull_apart(0.6, slide_left=False, slide_right=False, return_to_center=False)
    iface.sync()
    
    # lift up
    cur_left_pos = iface.y.left.get_joints()
    cur_right_pos = iface.y.right.get_joints()
    OUT_POS_2_L = np.array([-0.80388363, -0.77492968, -1.30770091, -0.73786254,  2.85080849, 0.79577677,  2.01228079])
    # ALTERNATE OUT_POS_2_L = np.array([-0.80290279, -0.77479652, -1.30767015, -0.27173554,  3.06072284, 0.73803378,  2.01229837])
        #np.array([ 0.20857435, -1.0969679 ,  0.5486748 , -0.96094356,  0.49384195,
        #0.7884078 , -1.70679424]) #np.array([-0.24853087, -1.0878531 , -1.11695033, -0.77183069,  2.21226595,
        #1.37551019, cur_left_pos[6]]) #np.array([-0.82856348, -1.36882858, -0.5320617 , -0.23686631,  2.62157876, 1.29712763,  cur_left_pos[6]])
    OUT_POS_2_R = l_to_r(OUT_POS_2_L)
    OUT_POS_2_R[6] = cur_right_pos[6]
    iface.go_configs(l_q=[OUT_POS_2_L], r_q=[OUT_POS_2_R])
    iface.sync()
    time.sleep(5)
    if check_knot():
        return False
    # cur_left_pos = iface.y.left.get_joints()
    # cur_right_pos = iface.y.right.get_joints()
    # CENTER_L[6] = cur_left_pos[6]
    # CENTER_R[6] = cur_right_pos[6]
    # iface.go_configs(l_q=[CENTER_L], r_q=[CENTER_R])
    return_to_center()
    iface.sync()
    iface.open_grippers()
    iface.home()
    iface.sync()
    if check_knot(end=True):
        return False

    return True

    OUT_POS_1_L = np.array([-0.476     , -0.98467079,  0.54838908, -0.52832758,  1.17841736,
        1.00729914, -2.92614212])
    OUT_POS_2_L = np.array([-0.82856348, -1.36882858, -0.5320617 , -0.23686631,  2.62157876,
            1.29712763, -1.7537732 ])
    OUT_POS_3_L = np.array([-1.44924473, -0.83080499,  1.20914265,  0.26091329,  1.72487018,
        1.4053557 , -3.99185077])

    def move_to_joint_pos(l_joint_pos, incremental):
        cur_left_pos = iface.y.left.get_joints()
        cur_right_pos = iface.y.right.get_joints()
        def l_to_r(config):
            return np.array([-1, 1, -1, 1, -1, 1, 1]) * config
        
        r_joint_pos = l_to_r(l_joint_pos)
        l_joint_pos[6] = cur_left_pos[6]
        r_joint_pos[6] = cur_right_pos[6]

        l_R = iface.y.left.get_pose().rotation
        if l_R[:, 1].dot(np.array((1, 0, 0))) > 0:  # if y axis points towards pos x
            l_joint_pos[6] -= np.pi
        
        r_R = iface.y.right.get_pose().rotation
        if r_R[:, 1].dot(np.array((1, 0, 0))) > 0:  # if y axis points towards pos x
            r_joint_pos[6] -= np.pi

        iface.go_configs(l_q=[l_joint_pos], r_q=[r_joint_pos])    
        iface.sync()

        time.sleep(2)
        if incremental:
            see_knot, _ = state_estimation()
            if see_knot:
                iface.home()
                iface.sync()
                iface.go_configs(l_q=[OUT_POS_3_L], r_q=[l_to_r(OUT_POS_3_L)]) 
                iface.sync()
                iface.open_grippers()
                iface.open_arms()
                return False
        return True

    iface.go_delta(l_trans=[0,0,0.05], r_trans=[0,0,0.05])
    
    states = [iface.L_HOME_STATE, OUT_POS_1_L, OUT_POS_2_L, iface.L_HOME_STATE, OUT_POS_3_L]
    for state in states:
        cont = move_to_joint_pos(state, incremental)
        if not cont:            
            return False

    logging.debug("Done with incremental/regular Reidemeister move.")
    iface.open_grippers()
    iface.home()
    iface.sync()
    iface.open_arms()

    if incremental:
        return True
    return False


def get_endpoints(img, vis=False):
    # model not used, already specified in loop_detectron
    endpoint_boxes, out_viz = loop_detectron.predict(img.color._data, thresh=0.99, endpoints=True)
    plt.clf()
    plt.imshow(out_viz)
    plt.title("Endpoints detected")
    show_img()

    endpoints = []
    for box in endpoint_boxes:
        xmin, ymin, xmax, ymax = box
        x = (xmin + xmax)/2
        y = (ymin + ymax)/2
        new_yx = closest_valid_point(img.color._data, img.depth._data, np.array([x, y]))[::-1]
        endpoints.append([new_yx, new_yx])   
    endpoints = np.array(endpoints).astype(np.int32)
    # (n,2,2) (num endpoints, (head point, neck point), (x,y))
    endpoints = np.array([e for e in endpoints if img.depth._data[e[0][0], e[0][1]] > 0])
    logger.debug("Found {} true endpoints after filtering for depth".format(len(endpoints)))

    plt.clf()
    plt.title("Endpoint detections visualization")
    plt.imshow(img.color._data)
    if len(endpoints) > 0:
        plt.scatter(endpoints[:, 0, 1], endpoints[:, 0, 0])
    show_img()

    return endpoints.astype(np.int32).reshape(-1, 2, 2)

def print_and_log(file, *args):
    logger.info(' '.join(map(str, args)))
    file.write(' '.join(map(str, args)) + '\n')
    file.flush()


def get_handler(q):
    # Handles signals, e.g. sigint to close camera streams.
    def signal_handler(sig, frame):
        global t2, _FINISHED
        _FINISHED = True
        t2.join()
        sys.exit(0)
    return signal_handler


def take_video(vfname):
    # Create a VideoCapture object
    logger.debug(vfname)
    cap = cv2.VideoCapture(14)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        logger.debug("Unable to read camera feed")
        return

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

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

def disambiguate_endpoints(endpoints, img, iface, incremental=False, near_point=None):
    """Pulls an endpoint in if 2 aren't visible, otherwise does a reidemeister move.
    
    Returns: True if termination, False otherwise.
    """
    logger.debug(f"endpoints: {endpoints}")
    
    color_img = img.color._data
    plt.clf()
    plt.imshow(color_img)
    plt.title("Original endpoint before revealing")
    plt.scatter(endpoints[:, 0, 1], endpoints[:, 0, 0])
    show_img()

    if len(endpoints) < 2:
        logger.info('Fewer than two endpoints visible, so revealing endpoints.')

        iface.reveal_endpoint(img, closest_to_pos=near_point)
        iface.sync()
        return False
    else:
        logger.info(f'Two endpoints detected, so running endpoint separation. Incremental is {incremental}.')
        try: 
            result = run_endpoint_separation_incremental(endpoints, img, iface, thresh=0.99, incremental=incremental)
        except Exception as e:
            logger.info(e)
            # if reidemeister fails, do an action similar to endpoint reveal
            logger.info("CAUGHT REIDEMEISTER ISSUE, falling back on reveal endpoint.")
            result = False
            iface.y.reset()
            iface.sync()
            iface.open_grippers()
            iface.home()
            iface.sync()
            iface.open_arms()
            iface.sync()
            img = take_and_save_image(iface)
            endpoints= get_endpoints(img)
            
            iface.reveal_endpoint(img)
            iface.sync()
        return result


def run_pipeline():
    global t2, _FINISHED
    vid_queue = Queue()
    start_time = int(time.time())
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_file_name = f'/home/justin/yumi/cable-untangling/scripts/full_rollouts/logs_{date_time}.txt'
    # open logs file
    logs_file = open(logs_file_name, 'w')
    logs_file.write(f'Start time: {start_time}\n')
    print_and_log(logs_file, f'START TIME: {start_time}\n')

    # format as date time
    SPEED = (0.6, 2 * np.pi) #(0.6, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    action_count = 0
    vfname = f'/home/justin/yumi/cable-untangling/scripts/full_rollouts/vid_{date_time}.avi'

    iface.open_grippers()
    iface.open_arms()
    keypoint_model = KeypointNetwork(
        "~/brijen/fc-vision/models/endpoint_fcn.ckpt", params={'task': 'cable_endpoints'})
    stop_cond_net = KeypointNetwork(
        "~/brijen/fc-vision/models/combined_endpoint_fcn.ckpt", params={'task': 'cable_endpoints'})
    # keypoint_model = KeypointNetwork(
    #     "~/brijenfc-vision/models/endpoint_fcn.ckpt")
    # stop_cond_net = KeypointNetwork(
    #     "~/brijenfc-vision/models/combined_endpoint_fcn.ckpt")
    point_nearest_to_knot, failure = None, False
    just_shook = False
    prev_trace_uncertain = False
    
    left_arm_reachability_mask = None #draw_mask(left_arm_mask_poly, img.color.shape[:2])
    right_arm_reachability_mask = None

    while (int(time.time()) - start_time) < 15 * 60:
        try:
            iface.open_grippers()
            iface.open_arms()
            time.sleep(1)
            img = take_and_save_image(iface)
            color_img = img.color._data

            endpoints = get_endpoints(img)
            logger.debug(f"endpoints {endpoints}")

            if endpoints.shape[0] == 0:
                failure = True

            # if failure:
            #     print_and_log(logs_file, action_count, int(time.time()), "Disambiguation move due to previous failure or no endpoints detected...")
            #     failure = False
            #     disambiguate_endpoints(endpoints, img, iface, incremental=False)
            #     action_count += 1
            #     continue

            #1. detect knot
            detectron_out,viz = loop_detectron.predict(img.color._data, thresh=0.99)
            logger.info(f"Detected {len(detectron_out)} knots")
            plt.clf()
            plt.imshow(viz)
            plt.title("Knot detection output")
            show_img()

            #2. If you see knot, untangle. Else, incrememntal reidemeister move to check no knots.
            if len(detectron_out) > 0: # detectron_out is a list of boxes
                g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)

                # iterate over possible endpoints and find the least uncertain one
                best_ret = None
                best_uncertain_type, trace_uncertain_type, ensemble_uncertain_type, no_uncertain_type = -1, 0, 1, 2
                # min_pull_apart_dist = 1e10
                max_ensemble_uncertain_val = 0
                np.random.shuffle(endpoints)
                for hulk_endpoint in endpoints:
                    ret = network_points(img, cond_grip_pose=hulk_endpoint[0], neck_point=hulk_endpoint[1], vis=vis)
                    left_coords, right_coords, trace_uncertain, ensemble_uncertain, ensemble_uncertain_val, pull_apart_dist = ret
                    if trace_uncertain and trace_uncertain_type > best_uncertain_type:
                        best_ret = ret
                        best_uncertain_type = trace_uncertain_type
                    elif ensemble_uncertain and ensemble_uncertain_type > best_uncertain_type:
                        best_ret = ret
                        best_uncertain_type = ensemble_uncertain_type
                    elif (not trace_uncertain and not ensemble_uncertain) and (no_uncertain_type > best_uncertain_type or ensemble_uncertain_val > max_ensemble_uncertain_val):
                        best_ret = ret
                        best_uncertain_type = no_uncertain_type
                        max_ensemble_uncertain_val = ensemble_uncertain_val
                        logger.info(f"Choosing ensemble with val {max_ensemble_uncertain_val}.")
                        # min_pull_apart_dist = pull_apart_dist
                if best_ret is not None:
                    left_coords, right_coords, trace_uncertain, ensemble_uncertain, ensemble_uncertain_val, pull_apart_dist = best_ret
                    logger.info(f"Distance to pull apart: {pull_apart_dist}.")
                    left_coords = closest_valid_point(img.color._data, img.depth._data, left_coords) if left_coords is not None else None
                    right_coords = closest_valid_point(img.color._data, img.depth._data, right_coords) if right_coords is not None else None
                    logger.info(f"trace uncertain {trace_uncertain} and ensemble uncertain {ensemble_uncertain}")
                exit()

                # just_shook = False
                if best_ret is not None and left_coords is not None and not trace_uncertain and not ensemble_uncertain:
                    print_and_log(logs_file, action_count, int(time.time()), "Doing a cage-pinch dilation move...")
                    prev_trace_uncertain = False
                    # saw_no_knots = False
                    try:
                        l_grasp, r_grasp = g.double_grasp(
                            tuple(left_coords), tuple(right_coords), .0085, .0085, iface.L_TCP, iface.R_TCP, slide0=True, slide1=False)
                    except:
                        iface.reveal_endpoint(img, closest_to_pos=((np.array(left_coords) + np.array(right_coords))/2)[::-1])
                        action_count += 1
                        continue
                    
                    s1, s2 = l_grasp.slide, r_grasp.slide

                    plt.clf()
                    plt.imshow(color_img)
                    plt.title("Cage-Pinch In-Place Pull Apart points")
                    cage = plt.scatter([left_coords[0]], [left_coords[1]], color='red', label='cage')
                    pinch = plt.scatter([right_coords[0]], [right_coords[1]], color='blue', label='pinch')
                    plt.legend((cage, pinch), ('Cage', 'Pinch'))
                    show_img()

                    logging.info("Homing arms.")
                    iface.home()
                    iface.sync()
                    logging.info(f"Left arm slide: {s1}, Right arm slide: {s2}")

                    iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp, topdown=topdown_grasps, bend_elbow=False)
                    # iface.go_delta(l_trans=[0, 0, .05], r_trans=[0, 0, .05])
                    # iface.go_delta(l_trans=[0, 0, -.03],
                    #             r_trans=[0, 0, -.03], reltool=True)
                    # 7. execute grasps and CG+CG pull apart motion
                    try:
                        iface.partial_pull_apart(pull_apart_dist, slide_left=s1, slide_right=s2, layout_nicely=True) #, return_to_center=True)
                        iface.sync()
                    except:
                        #TODO: not working
                        iface.y.reset()
                        logging.info("FAILED PULL APART ONCE, trying again")
                        #bring arms in first before pulling again
                        iface.go_delta(l_trans=[0, -0.05, 0], r_trans=[0, 0.05, 0])
                        iface.sync()
                        iface.shake_R('right', rot=[0, 0, 0], num_shakes=4, trans=[0, 0, .05])
                        iface.sync()
                        iface.shake_R('left', rot=[0, 0, 0], num_shakes=4, trans=[0, 0, .05])
                        iface.sync()
                        iface.partial_pull_apart(pull_apart_dist + 0.02, slide_left=s1, slide_right=s2, layout_nicely=True) #, return_to_center=True)
                        iface.sync()
                        # iface.shake_R('right', rot=[0, 0, 0], num_shakes=4, trans=[0, 0, .05])
                        # iface.sync()
                        # iface.shake_R('left', rot=[0, 0, 0], num_shakes=4, trans=[0, 0, .05])
                    iface.open_grippers()
                    iface.open_arms()
                    action_count += 1
                elif best_ret is not None and (ensemble_uncertain and not trace_uncertain): #ensemble uncertain
                    print_and_log(logs_file, action_count, int(time.time()), "Partial pull apart due to ensemble uncertainty...")
                    prev_trace_uncertain = False

                    try:
                        l_grasp, r_grasp = g.double_grasp(
                            tuple(left_coords), tuple(right_coords), .0085, .0085, iface.L_TCP, iface.R_TCP)
                    except:
                        iface.reveal_endpoint(img, closest_to_pos=((np.array(left_coords) + np.array(right_coords))/2)[::-1])
                        action_count += 1
                        continue

                    plt.clf()
                    plt.imshow(color_img)
                    plt.title("Cage-Pinch PARTIAL Pull Apart points")
                    cage = plt.scatter([left_coords[0]], [left_coords[1]], color='red', label='cage')
                    pinch = plt.scatter([right_coords[0]], [right_coords[1]], color='blue', label='pinch')
                    plt.legend((cage, pinch), ('Cage', 'Pinch'))
                    show_img()

                    iface.home()
                    iface.sync()
                    iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp, topdown=topdown_grasps, bend_elbow=False)
                    # iface.go_delta(l_trans=[0, 0, .05], r_trans=[0, 0, .05])
                    # iface.go_delta(l_trans=[0, 0, -.03],
                    #             r_trans=[0, 0, -.03], reltool=True)
                    s1, s2 = l_grasp.slide, r_grasp.slide

                    iface.partial_pull_apart(0.05, slide_left=s1, slide_right=s2)
                    iface.sync()
                    iface.open_grippers()
                    iface.open_arms()
                    action_count += 1
                else:
                    print_and_log(logs_file, action_count, int(time.time()), "Full Reidemeister due to tracing uncertainty ...")
                    prev_trace_uncertain=True
                    disambiguate_endpoints(endpoints, img, iface, incremental=False)
                    action_count += 1
            else:
                prev_trace_uncertain = False
                print_and_log(logs_file, action_count, int(time.time()), "Incremental reidemeister move termination check...")
                # if we succesfully finish without seeing a knot, break out of while loop
                if disambiguate_endpoints(endpoints, img, iface, incremental=True):
                    action_count += 1
                    print_and_log(logs_file, action_count, int(time.time()), "No knots confirmed.")
                    break
                else: # saw a knot
                    action_count += 1
                    print_and_log(logs_file, action_count, int(time.time()), "Saw knot so returning to untangling.")
                    continue
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
                logger.info(f"Caught error, recovering {str(e)}")
                failure = True
                iface.y.reset()
                last_in_dir = os.listdir()
            else:
                # e.printStackTrace()
                logger.info("Uncaught exception, still recovering " + str(e))
                # print traceback of exception
                #raise e # re-raise
                iface.y.reset()
            iface.sync()
    if (time.time() > start_time + 60 * 15):
        print_and_log(logs_file, action_count, f"Timed out at time {int(time.time())} after duration {int(time.time()) - start_time}.")
    else:
        print_and_log(logs_file, action_count, f"Done at time {int(time.time())} after duration {int(time.time()) - start_time}.")
    print_and_log(logs_file, action_count, f"Took {total_num_observations} images total.")
    # close logs file
    logs_file.close()
    exit()

    #after the loop, we're ready to spool
    logger.info("Done, cable is untangled.")
    vid_queue.put('stop')
    _FINISHED = True
    time.sleep(1)
 #   t2.join()


def test_detectron():
    SPEED = (0.6, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    logger.debug("init interface")
    iface.open_grippers()
    iface.open_arms()
    while True:
        input("Enter to try new pic")
        img=take_and_save_image(iface)
        network_points(img, neck_point=(500,500))


trial_idx = len([f for f in os.listdir("/home/justin/yumi/cable-untangling/scripts/full_rollouts") if "vid" in f]) + 1
# going to be slightly off, not sure why i have to put this here...
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

vfname = f'/home/justin/yumi/cable-untangling/scripts/full_rollouts/vid_%d.avi'%trial_idx
vfname = f'/home/justin/yumi/cable-untangling/scripts/full_rollouts/vid_{date_time}.avi'
# t2 = threading.Thread(target=take_video, args=(vfname,))


if __name__ == "__main__":
    run_pipeline()
    #run_endpoint_separation()
    # while True:
    #     test_detectron()
