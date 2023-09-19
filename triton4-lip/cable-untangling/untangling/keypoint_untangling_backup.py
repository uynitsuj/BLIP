import logging
import pickle
import colorsys
from re import L
from cv2 import waitKey
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import torch
torch.cuda.is_available()
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from sklearn import preprocessing, decomposition
import random as rand
import glob
import sys
# from untangling.utils.circle_BFS import *
# from untangling.utils.likelihood_trace import *
from cable_tracing.tracers.simple_uncertain_trace import trace
from cable_tracing.utils.utils import get_dist_cumsum, score_path
from untangling.utils.loop_box import get_loop_box
from untangling.cascading_grasp_predictor.get_points import NetworkGetPoints
from untangling.hparams import *

images_to_show = []
vis_imgs = False

logger = logging.getLogger("Untangling")

def add_image_to_show(image, name):
    global images_to_show
    images_to_show.append((name, image.copy()))

def clear_images_to_show():
    global images_to_show
    images_to_show = []

def get_click_points_on_image(image, num_points=2):
    fig = plt.figure()
    coords = []
    def onclick(event):
        ix, iy = int(event.xdata), int(event.ydata)
        logger.debug('y = %d, x = %d'%(iy, ix))

        nonlocal coords
        coords.append((iy, ix))

        if len(coords) == num_points:
            fig.canvas.mpl_disconnect(cid)

        return coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.imshow(image)
    plt.show()
    return coords

def display_images_to_show(dir_name=None, index=None, show=True):
    # show all on same plot in grid
    num_figs = len(images_to_show)
    num_cols = int(np.ceil(np.sqrt(num_figs)))
    num_rows = int(np.ceil(num_figs/num_cols))
    fig, ax = plt.subplots(num_rows, num_cols)
    fig.tight_layout(pad=1.0)
    for i, (name, img) in enumerate(images_to_show):
        ax[int(i/num_cols), i%num_cols].imshow(img)
        ax[int(i/num_cols), i%num_cols].set_title(name)
    if index == None:
        index = 0
    if dir_name is not None:
        plt.savefig(dir_name + "/actions_" + str(index) + ".png")

    # if show:
        # plt.ion()
    # plt.imshow(fig)
    plt.show()

hulk_keypoints = os.path.dirname(os.path.abspath(__file__)) + "/../../action-keypoints"
sys.path.insert(0,hulk_keypoints)
import mm_analysis

# conditioned_bb = os.path.dirname(os.path.abspath(__file__)) + "/../../detect-loop"
# sys.path.insert(0,conditioned_bb)
# import cond_analysis as loop_detect_conditioned

grasp_predictor = NetworkGetPoints()

detectron = os.path.dirname(os.path.abspath(__file__)) + "/../../detectron2_repo"
sys.path.insert(0,detectron)
import analysis as loop_detectron

os.environ["cuda_visible_devices"]="0"
torch.cuda.set_device(0)

def dist(vec1, vec2, verbose=False):
    vec1 = vec1.astype(float)
    vec2 = vec2.astype(float)
    return np.min(np.abs(vec1 - vec2))

def cartesian_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def sample_keypoints(bounding=(0, 0, 640, 480), model=False, reg_image=None, raw_image=None, masked_image=None, endpoints=False, depth_img=None, bfs_result=None, old_hulk=True, uncertainty=True):
    if reg_image.shape[0] == 3:
        logger.debug("TRANSPOSING REG IMAGE")
        reg_image = np.transpose(reg_image.copy(), (1, 2, 0))
    if endpoints:
        # black out bottom part of input image
        reg_image[310:] = 0
        pt1, pt2 = mm_analysis.make_predictions(reg_image, get_endpoints=True)
    elif model:
        # get click points from user for conditioned region
        # cast bounding to ints
        bounding = (int(bounding[0]), int(bounding[1]), int(bounding[2]), int(bounding[3]))
        bounded_img = raw_image[bounding[1]:bounding[1]+bounding[3], bounding[0]:bounding[0]+bounding[2]]
        # logger.debug("BOUNDING INFO", bounding, bounded_img.shape)

        cond_mask = np.zeros(reg_image.shape[:2], dtype="uint8")
        if len(bfs_result) == 1:
            cond_mask[tuple(bfs_result[0])] = 1
        elif len(bfs_result) > 0:
            last_10 = np.array(bfs_result[-50:])
            top_left, bottom_right = [np.min(last_10[:, 0]), np.min(last_10[:, 1])], [np.max(last_10[:, 0]), np.max(last_10[:, 1])] #get_click_points_on_image(bounded_img, num_points=2) 
            cond_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1
        else:
            logger.debug("WARNING: No BFS result!!") 

        # now crop cond_mask to bounded_img
        cond_mask = cond_mask[bounding[1]:bounding[1]+bounding[3], bounding[0]:bounding[0]+bounding[2]]

        old_hulk = True
        uncertain = False
        if old_hulk: 
            hulkL_heatmaps, hulkL_results = mm_analysis.get_heatmap(bounded_img, cond_mask=cond_mask, uncertainty=uncertainty)
            for hulkL_result in hulkL_results:  add_image_to_show(hulkL_result, "Hulk L Result")
            
            # now we have two heatmaps, so one box from each is what we want
            # box1, box2 = get_loop_box(hulkL_heatmap, largest=1, connectivity=8)
            min_of_heatmaps = np.min(hulkL_heatmaps, axis=0)
            assert min_of_heatmaps.shape[0] == 2

            all_overlays = []
            for i in range(2):
                h = min_of_heatmaps[i]
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                img = (cv2.resize(bounded_img, (200,200)) * 255).astype(np.uint8)
                overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
                all_overlays.append(overlay)
            result = cv2.hconcat((all_overlays[0], all_overlays[1]))
            cv2.putText(result, "Cage", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(result, "Pinch", (210,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            if vis_imgs:
                plt.imshow(result)
                plt.title(f'Visualize cage pinch grasp: Max cage {min_of_heatmaps[0].max()}, Max pinch {min_of_heatmaps[1].max()}')
                plt.show()

            # plt.title("Min heatmap being used: Cage")
            # plt.imshow(min_of_heatmaps[0])
            # plt.show()
            # plt.title("Min heatmap being used: Pinch")
            # plt.imshow(min_of_heatmaps[1])
            # plt.show()

            pt1 = np.unravel_index(min_of_heatmaps[0].argmax(), min_of_heatmaps[0].shape)[::-1]
            pt2 = np.unravel_index(min_of_heatmaps[1].argmax(), min_of_heatmaps[1].shape)[::-1]

            uncertain_val = min_of_heatmaps[0].max()*min_of_heatmaps[1].max()
            uncertain = (uncertain_val < HEATMAP_UNCERTAIN_THRESH) and uncertainty #or min_of_heatmaps[1].max() < HEATMAP_UNCERTAIN_THRESH
            logger.info(f"NETWORK UNCERTAINTIES: {min_of_heatmaps[0].max()} {min_of_heatmaps[1].max()} {min_of_heatmaps[0].max()*min_of_heatmaps[1].max()}")

            img = (reg_image[bounding[1]:bounding[1]+bounding[3], bounding[0]:bounding[0]+bounding[2],:]).copy()
            img = cv2.resize(img, (200,200))
            img = cv2.circle(img, pt1, 7, (200, 0, 200), -1)
            img = cv2.circle(img, pt2, 7, (0, 200, 0), -1)
            # title plt
            add_image_to_show(img, "endpoints" if endpoints else "chosen points on 200x200")

            #relocate pt1 and pt2 to fit aspect of original crop
            h_ratio = bounding[2]/200
            v_ratio = bounding[3]/200
            # logger.debug("ratios", h_ratio, v_ratio, pt1, pt2, bounding[2], bounding[3])
            pt2 = [int(pt2[0] * h_ratio), int(pt2[1] * v_ratio)]
            pt1 = [int(pt1[0] * h_ratio), int(pt1[1] * v_ratio)]
            pt1 = pt1[::-1]
            pt2 = pt2[::-1]
        else:
            # logger.debug(bounded_img.shape, type(bounded_img))
            # assert bounded_img.shape[0] == 3 and len(bounded_img.shape) == 3
            logger.debug("BOUNDING:")
            logger.debug(bounding)
            bounded_img = raw_image[bounding[1]:bounding[1]+bounding[3], bounding[0]:bounding[0]+bounding[2]]
            # logger.debug(bounded_img.shape)
            # bounded_img = cv2.resize(bounded_img, (200,200))
            # bounded_img_tensor = torch.from_numpy(bounded_img).cuda().unsqueeze(0)
            pt1, pt2, heatmap = grasp_predictor.query(bounded_img)
            add_image_to_show(heatmap, "HULKL heatmaps")
            if pt1 is None:
                return None, None

        pts_to_sample = get_all_nonzero_points_as_list(reg_image[bounding[1]:bounding[1]+bounding[3], bounding[0]:bounding[0]+bounding[2],:],
                                                       depth_img[bounding[1]:bounding[1]+bounding[3], bounding[0]:bounding[0]+bounding[2]])
        pt1_dist, pt1_idx = min((cartesian_dist(pt1, pt), idx) for idx, pt in enumerate(pts_to_sample))
        pt2_dist, pt2_idx = min((cartesian_dist(pt2, pt), idx) for idx, pt in enumerate(pts_to_sample))
        pt1 = pts_to_sample[pt1_idx][::-1]
        pt2 = pts_to_sample[pt2_idx][::-1]

    #opt_cp = np.copy(optional_vis)
    bounding_box = reg_image[bounding[1]:bounding[1]+bounding[3], bounding[0]:bounding[0]+bounding[2],:].copy()
    color = [1,1,1] #np.random.rand(3).tolist()

    cv2.circle(bounding_box, pt1, 7, color , -1)
    cv2.circle(bounding_box, pt2, 7, color, -1)

    # title plt
    add_image_to_show(bounding_box, "endpoints" if endpoints else "chosen points")

    pt1 = (pt1[0] + bounding[0], pt1[1] + bounding[1])
    pt2 = (pt2[0] + bounding[0], pt2[1] + bounding[1])

    return pt1, pt2, uncertain, uncertain_val

def gauss_2d_batch(width, height, sigma, U, V, single=False):
    if not single:
        U.unsqueeze_(1).unsqueeze_(2)
        V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-float(U))**2+(Y-float(V))**2)/(2.0*sigma**2))
    return G.double()

def dist_to_segment(ax, ay, bx, by, cx, cy):
    """
    Computes the minimum distance between a point (cx, cy) and a line segment with endpoints (ax, ay) and (bx, by).
    :param ax: endpoint 1, x-coordinate
    :param ay: endpoint 1, y-coordinate
    :param bx: endpoint 2, x-coordinate
    :param by: endpoint 2, y-coordinate
    :param cx: point, x-coordinate
    :param cy: point, x-coordinate
    :return: minimum distance between point and line segment
    """
    # avoid divide by zero error
    a = max(by - ay, 0.00001)
    b = max(ax - bx, 0.00001)
    # compute the perpendicular distance to the theoretical infinite line
    dl = abs(a * cx + b * cy - b * ay - a * ax) / np.sqrt(a**2 + b**2)
    # compute the intersection point
    x = ((a / b) * ax + ay + (b / a) * cx - cy) / ((b / a) + (a / b))
    y = -1 * (a / b) * (x - ax) + ay
    # decide if the intersection point falls on the line segment
    if (ax <= x <= bx or bx <= x <= ax) and (ay <= y <= by or by <= y <= ay):
        return dl
    else:
        # if it does not, then return the minimum distance to the segment endpoints
        return min(np.sqrt((ax - cx)**2 + (ay - cy)**2), np.sqrt((bx - cx)**2 + (by - cy)**2))

def point_in_box(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else :
      return False

def get_smallest_box(loops):
    areas = [0 for i in range(len(loops))]
    for i in range(len(loops)):
        x0, y0, x1, y1 = loops[i]
        a = abs(x1-x0) * abs(y1-y0)
        areas[i] = a
    smallest_loop = loops[np.argmin(areas)]
    x = smallest_loop[0]
    y = smallest_loop[1]
    dx = smallest_loop[2] - smallest_loop[0]
    dy = smallest_loop[3] - smallest_loop[1]
    loop_reformatted = [x, y, dx, dy]
    return loop_reformatted

def get_bounding_box(loops, last_BFS_point, padding=20):
    "takes boxes predicted by detectron (xmin, xmax, ymin, ymax) and the last BFS point, which marks the start of a non-trivial knot and return the closest loop"
    dist_from_point_to_loop = [0 for i in range(len(loops))]
    cy, cx = last_BFS_point if last_BFS_point is not None else (0,0)
    for i in range(len(loops)):
        x0, y0, x1, y1 = loops[i]
        center = [(x0+x1)//2, (y0+y1)//2]
        corners = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
        bl = corners[0]
        tr = corners[2]
        p = [cx, cy]
        if point_in_box(bl, tr, p):
            logger.debug(center)
            logger.debug(p)
            d_to_center = np.linalg.norm(np.array(center) - np.array(p))
            dist_from_point_to_loop[i] =  -1*(1/d_to_center)
        else:
            min_dist  = float('inf')
            for j in range(4):
                next = (j + 1) % 4
                ax, ay = corners[j]
                bx, by = corners[next]
                d = abs(dist_to_segment(ax, ay, bx, by, cx, cy))
                if d < min_dist:
                    min_dist = d
            dist_from_point_to_loop[i] = min_dist
    logger.debug("Distance values")
    logger.debug(dist_from_point_to_loop)
    closest_loop = loops[np.argmin(dist_from_point_to_loop)]
    #convert to format: x, y, dx, dy
    x = closest_loop[0]
    y = closest_loop[1]
    dx = closest_loop[2] - closest_loop[0]
    dy = closest_loop[3] - closest_loop[1]
    loop_reformatted = [max(0, x - padding/2), max(0, y - padding/2), dx + padding, dy + padding]
    return loop_reformatted

def run_inference(img_masked, img_raw, depth_masked=None, random=False, without_thresh=None,
                  save_boxes = False, cond_grip_pose=None, neck_point=None, interactive_trace=False, uncertainty=True):
    trace_uncertain = False
    right_endpoint = cond_grip_pose # in reality, the left endpoint
    without_thresh = np.transpose(without_thresh, (1,2,0))

    img_raw_tensor = torch.from_numpy(img_raw).cuda().unsqueeze(0)

    loops, vis_loops = loop_detectron.predict(img_raw_tensor, thresh=0.99) # 0.7
    old_loops = loops.copy()

    def loop_valid(loop, mask):
        black = (mask[loop[1]:loop[3], loop[0]:loop[2], 0] == 0).astype(np.uint8)
        # plt.imshow(black)
        # plt.show()
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
        logger.debug(f"FOUND {num_labels} labels in Connected components for loop")
        return num_labels >= 3 # background, inside, outside

    loops = []
    for i in range(len(old_loops)):
        old_l = old_loops[i]
        new_l = [old_l[0]/1.61,old_l[1]/1.61,old_l[2]/1.61,old_l[3]/1.61]
        old_loops[i]=new_l
        if loop_valid(np.array(old_l).astype(int), img_masked.transpose(1, 2, 0)):
            loops.append(new_l)
        else:
            logger.debug("Filtered out non-closed loop.")
    loops = np.array(loops)

    vis_loops = cv2.resize(vis_loops, (640, 480))
    logger.debug("Found {} loops".format(len(loops)))

    if len(loops) == 0:
        return None

    add_image_to_show(vis_loops, "loop_mask")

    # get bounding box
    depth_extra_dim = np.expand_dims(depth_masked, axis=0)
    img_masked_T = np.transpose(img_masked, (1,2,0)) * 255
    depth_extra_dim = np.transpose(depth_extra_dim, (1, 2, 0))
    combined = np.concatenate([img_masked_T, depth_extra_dim], axis=-1)

    img_copy = img_masked_T.copy()
    # TODO @Kaushik: fix this, should not need to use masked image here

    # reformat to ymin, xmin, ydelta, xdelta
    loops_reformatted = np.array([loops[:, 1], loops[:, 0], loops[:, 3] - loops[:, 1], loops[:, 2] - loops[:, 0]]).T.astype(int)

    if vis_imgs:
        # Visualize loops
        img_disp = without_thresh.copy()
        for loop in loops_reformatted:
            img_disp = cv2.rectangle(img_disp, (loop[1], loop[0]), (loop[1] + loop[3], loop[0] + loop[2]), (255, 0, 255), 3)
        plt.scatter(*right_endpoint[::-1])
        plt.title('Visualize loops before tracing from endpoint')
        plt.imshow(img_disp)
        plt.show()

    path_ret, all_paths = trace(combined, right_endpoint, None, False, False, 20, loops_reformatted, viz=vis_imgs)
    longest_path_len, longest_path = 0, path_ret
    if uncertainty:
        if path_ret is not None:
            for path in all_paths:
                if len(path) > 1:
                    if get_dist_cumsum(path)[-1] > longest_path_len:
                        longest_path_len = get_dist_cumsum(path)[-1]
                        longest_path = path
    else:
        # repurposes longest path to be the highest scoring path
        for path in all_paths:
            if (len(path) > 1):
                path_score = score_path(combined[:, :, :3], None, path)
                if path_score > longest_path_len:
                    longest_path_len = path_score
                    longest_path = path
        if longest_path is not None and len(longest_path) > 1:
            longest_path_len = get_dist_cumsum(longest_path)[-1]

    if interactive_trace:
        return (None, None, True, True, 0, 0)
    if path_ret is None and uncertainty:
        logging.warning("Uncertain analytic trace result.")
        trace_uncertain = True

    if not trace_uncertain:
        last_point = np.array(longest_path[-1])
    else:
        last_point = np.array([0, 0])
        return (None, None, True, True, 0, 0)

    padding_amt = 80
    bbox = get_bounding_box(loops, last_point, padding=padding_amt)
    PIXELS_TO_METERS = 1000

    longest_path_len = min((longest_path_len + 1.5*(bbox[3] + bbox[2] - 2*padding_amt))/ PIXELS_TO_METERS / 2, 0.55)
    logger.info(f"Pull apart distance components: {bbox[3]}, {bbox[2]}, {2*padding_amt}, {longest_path_len}")

    # bbox = get_smallest_box(loops) #, bfs_result[-1]) # trace(combined, right_endpoint)[-1]
    if save_boxes:
        box_image = img_masked[:, bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        box_image = (box_image * 255.0).astype(np.uint8)
        box_image = np.transpose(box_image, (1,2,0))
        random_num = np.random.randint(1000000)
        cv2.imwrite("bounding_boxes/" + str(random_num) + ".png", box_image)
        return

    # bfs result until box, keep adding pixels until the box has at least 50 pixels
    pixels_to_pass = np.array(longest_path[-3:]).astype(int)
    # logger.debug("Pixels to pass", bbox, pixels_to_pass)
    if vis_imgs:
        plt.scatter(*last_point[::-1])
        plt.title("Conditioned point for determining cage-pinch")
        plt.imshow(img_disp)
        plt.show()

    pixels_to_pass_mask = np.zeros((480, 640))
    for pt in pixels_to_pass:
        pixels_to_pass_mask[pt[0], pt[1]] = 1
    #plt.imshow(img_copy)
    img_copy[:,:,1] = pixels_to_pass_mask
    #plt.show()

    #logger.debug("Sampling " + str(k))
    sample_uv_a, sample_uv_b, ensemble_uncertain, ensemble_uncertain_val = sample_keypoints(bounding=bbox, model=True, reg_image=img_masked,
                                                           raw_image=without_thresh, depth_img=depth_masked,
                                                           bfs_result=pixels_to_pass, uncertainty=uncertainty) #HULK-L predicted pull points
    if sample_uv_a is None:
        return None
    #res = None #np.vstack((desc_a_vis, desc_a_vis_masked, vis_boxes))
    add_image_to_show(depth_masked, "depth masked")

    return (sample_uv_a, sample_uv_b, trace_uncertain, ensemble_uncertain, ensemble_uncertain_val, longest_path_len)

def center_crop(img, dim):
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def rotate_image_to_landscape(image):
    height, width = image.shape[:2]
    if height > width:
        image = cv2.transpose(image)
        image = cv2.flip(image, 1)
    return image

def scale_image_to_width(image, width):
    height, orig_width = image.shape[:2]
    image = cv2.resize(image, (width, int(width*height/orig_width)), interpolation=cv2.INTER_AREA)
    return image

def scale_image_to_height(image, height):
    orig_height, width = image.shape[:2]
    image = cv2.resize(image, (int(width*height/orig_height), height), interpolation=cv2.INTER_AREA)
    return image

def perform_all_operations(image, point_to_track=None):
    thresh = image.copy()

    thresh = rotate_image_to_landscape(thresh)

    if (image.shape[1]/image.shape[0] > 640/480):
        thresh = scale_image_to_height(thresh, 480)
    else:
        thresh = scale_image_to_width(thresh, 640)
    shapexd = thresh.shape
    thresh = center_crop(thresh, (640, 480))

    thresh = (thresh/thresh.max()).astype(np.float32)
    without_thresh = thresh.copy()

    thresh = np.where(thresh > 80.0/255.0, thresh, 0)
    if thresh.shape[2] == 3:
        thresh = np.transpose(thresh, (2, 0, 1))
    if without_thresh.shape[2] == 3:
        without_thresh = np.transpose(without_thresh, (2, 0, 1))
    if point_to_track is None:
        return thresh,shapexd, without_thresh

def get_all_nonzero_points_as_list(img, depth_img=None):
    """Returns list of all nonzero points in descriptors"""
    nonzero_points = []
    height, width = img.shape[:2]
    for h in range(height):
        for w in range(width):
            if any(abs(img[h, w, :])) > 0 and (depth_img is None or (abs(depth_img[h, w])) > 0):
                nonzero_points.append((h, w))
    return nonzero_points

def get_loop_box_DEPRECATED(heatmap, largest=1, random=False, connectivity=4, sample=False, already_box=None):
    heatmap = heatmap[0][0]

    draw_heatmap = heatmap.copy()
    ref_heatmap = heatmap.copy()

    # for each connected component of the heatmap (contiguously > 0.4)
    # get the bounding box of the connected component
    bboxes = []
    while True:
        # first find a mask for the heatmap
        mask = np.where(heatmap > 0.25, 255, 0) # 0.45

        # convert mask to correct format for connectedComponentsWithStats
        mask = mask.astype(np.uint8)

        # find the largest connected component in the mask
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

        # get the bounding box of the largest connected component if one exists
        if n_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            bbox = stats[largest_label, :]
            # logger.debug("Going to pick", bbox)
            bboxes.append(bbox)
        else:
            break

        heatmap[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 0

        # discard the bounding box if it is too small

        if bbox[cv2.CC_STAT_AREA] < 75:
            break

    for i in range(len(bboxes)):
        if i < 2:
            bbox = bboxes[i]
            # white entire bounding box
            draw_heatmap[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1
            # restore inner part of the heatmap except small radius = 2
            pt = 3
            draw_heatmap[bbox[1] + pt:bbox[1] + bbox[3] - pt, bbox[0] + pt:bbox[0] + bbox[2] - pt] = ref_heatmap[
                bbox[1] + pt:bbox[1] + bbox[3] - pt, bbox[0] + pt:bbox[0] + bbox[2] - pt]
    
    # add_image_to_show(draw_heatmap, 'heatmap')

    if already_box is not None:
        # find box farthest away from already_box's center
        farthest_box, farthest_dist = None, 0
        for bbox in bboxes:
            dist = np.linalg.norm(np.array([bbox[1], bbox[1] + bbox[3]/2, bbox[0], bbox[0] + bbox[2]/2]) - np.array([already_box[1], already_box[1] + already_box[3]/2, already_box[0], already_box[0] + already_box[2]/2]))
            if bbox[cv2.CC_STAT_AREA] > 200 and dist > farthest_dist:
                farthest_dist = dist
                farthest_box = bbox
        if farthest_box is not None:
            return farthest_box

    if sample:
        # return bounding box with probability proportional to area
        total_area = 0
        for bbox in bboxes:
            total_area += bbox[cv2.CC_STAT_AREA]
        prob = np.random.rand() * total_area
        for bbox in bboxes:
            prob -= bbox[cv2.CC_STAT_AREA]
            if prob < 0:
                return bbox
        return bboxes[0]

    if largest == 1:
        return bboxes[0]
    return bboxes[:largest]

def closest_valid_point_to_pt(depth_orig, goal_pt, color=None):
    goal_pt = goal_pt[::-1]
    if color is not None:
        color = color.copy()[:, :, 0]
        color = np.where(color > 100, 255, 0)
        color = cv2.erode(color.astype(np.uint8), np.ones((2, 2), dtype=np.uint8))
    closest_to_pt1, closest_dist_to_pt1 = None, 9999999
    for i in range(depth_orig.shape[0]):
        for j in range(depth_orig.shape[1]):
            if depth_orig[i, j] == 0:
                continue
            if color is not None and color[i, j] < 0.01:
                continue
            pt1_dist = cartesian_dist((i, j), goal_pt)
            if pt1_dist < closest_dist_to_pt1:
                closest_dist_to_pt1 = pt1_dist
                closest_to_pt1 = (i, j)
    return closest_to_pt1[::-1]

def get_transformed_point(origshape, pt):
    one_hot = np.zeros(origshape)
    one_hot[pt[0], pt[1]] = 255
    one_hot_masked, _, _ = perform_all_operations(one_hot)
    U, V = np.nonzero(one_hot_masked[0])
    pt = U[0], V[0]
    return pt

def get_good_points_from_image(raw, image, image_completely_raw,depth=None, pairs=1, random=False,
        dir_name=None, index=None, cond_grip_pose=None, neck_point=None, interactive_trace=False, vis=False, uncertainty=True):
    '''
    raw includes thresholding the bottom 130 pixels, image_completely_raw is the literal raw image
    '''
    global vis_imgs
    vis_imgs = vis
    clear_images_to_show()

    origshape = image.shape  #raw image
    image_masked, threshshape, _ = perform_all_operations(image) #pre-processing

    if cond_grip_pose is not None:
        cond_grip_pose = get_transformed_point(origshape, cond_grip_pose)
    if neck_point is not None:
        neck_point = get_transformed_point(origshape, neck_point)

    _, _, without_thresh = perform_all_operations(raw)
    img_temp = np.transpose(image_masked, (1, 2, 0))
    gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    mask_a = np.where(gray > 0, 1.0, 0.0)  #masking the rope
    depth_orig = depth.copy()
    depth = cv2.resize(depth, (640, 480))
    depth = np.float64(depth)
    depth_masked = np.multiply(depth, mask_a)

    add_image_to_show(depth_orig, 'raw depth image')
    
    # convert image to batch-1 tensor
    # image_tensor = torch.from_numpy(image).cuda().unsqueeze(0)
    # image_masked = torch.from_numpy(image_masked).cuda().unsqueeze(0)

    # image must be shape (1, 3, 640, 480) now
    points = run_inference(image_masked, image_completely_raw,depth_masked=depth_masked, random=random,
                            without_thresh=without_thresh, cond_grip_pose=cond_grip_pose, neck_point=neck_point, interactive_trace=interactive_trace, uncertainty=uncertainty) #everything
    if points is None:
        return (None, None, False, False, 0, 0)

    lc,rc, trace_uncertain, ensemble_uncertain, ensemble_uncertain_val, pull_apart_dist = points
    if trace_uncertain:
        return lc, rc, trace_uncertain, ensemble_uncertain, ensemble_uncertain_val, pull_apart_dist

    horiz_shift = (threshshape[1]-640)/2
    vert_shift  = (threshshape[0]-480)/2
    s = origshape[0]/threshshape[0]

    points_to_return = (int(s*(lc[0]+horiz_shift)),int(s*(lc[1]+vert_shift))),\
                (int(s*(rc[0]+horiz_shift)),int(s*(rc[1]+vert_shift)))

    imcopy = depth_orig.copy() * 255.0

    # FIND CLOSEST POINTS TO THE TARGET POINTS WITH VALID DEPTH
    def cartesian_dist(pt1, pt2):
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    first_point = closest_valid_point_to_pt(depth_orig, points_to_return[0])
    second_point = closest_valid_point_to_pt(depth_orig, points_to_return[1])

    #logger.debug("")
    #logger.debug("Chosen points:", first_point, second_point, imcopy[first_point[0], first_point[1]], imcopy[second_point[0], second_point[1]])
    cv2.circle(imcopy, first_point, 6, (255, 255, 255), 2)
    cv2.circle(imcopy, second_point, 6, (255, 255, 255), 2)
    add_image_to_show(imcopy, 'depth with final points')

    # display_images_to_show(dir_name, index, show=True)

    return first_point, second_point, trace_uncertain, ensemble_uncertain, ensemble_uncertain_val, pull_apart_dist

def save_bounding_boxes(image, depth=None, random=False):
    clear_images_to_show()
    image_masked, threshshape, _ = perform_all_operations(image) #pre-processing
    img_temp = np.transpose(image_masked, (1, 2, 0))
    gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    mask_a = np.where(gray > 0, 1.0, 0.0)  #masking the rope
    depth = cv2.resize(depth, (640, 480))
    depth = np.float64(depth)
    depth_masked = np.multiply(depth, mask_a)
    run_inference(image_masked, depth_masked=depth_masked, random=random, without_thresh=None, save_boxes=True)

def get_points_testing(image, depth=None, random=False, dir_name=None, index=None):
    clear_images_to_show()

    origshape = image.shape  #raw image
    image_masked, threshshape, _ = perform_all_operations(image) #pre-processing
    img_temp = np.transpose(image_masked, (1, 2, 0))
    gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    mask_a = np.where(gray > 0, 1.0, 0.0)  #masking the rope
    depth_orig = depth.copy()
    depth = cv2.resize(depth, (640, 480))
    depth = np.float64(depth)
    depth_masked = np.multiply(depth, mask_a)

    add_image_to_show(depth_orig, 'raw depth image')

    # image must be shape (1, 3, 640, 480) now
    lc, rc, uncertain = run_inference(image_masked, depth_masked=depth_masked, random=random, without_thresh=None) #everything

    horiz_shift = (threshshape[1]-640)/2
    vert_shift  = (threshshape[0]-480)/2
    s = origshape[0]/threshshape[0]

    points_to_return = (int(s*(lc[0]+horiz_shift)),int(s*(lc[1]+vert_shift))),\
                (int(s*(rc[0]+horiz_shift)),int(s*(rc[1]+vert_shift)))

    imcopy = depth_orig.copy() * 255.0

    # FIND CLOSEST POINTS TO THE TARGET POINTS WITH VALID DEPTH
    def cartesian_dist(pt1, pt2):
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    def closest_valid_point_to_pt(goal_pt):
        goal_pt = goal_pt[::-1]
        closest_to_pt1, closest_dist_to_pt1 = None, 9999999
        for i in range(imcopy.shape[0]):
            for j in range(imcopy.shape[1]):
                if depth_orig[i, j] == 0:
                    continue
                pt1_dist = cartesian_dist((i, j), goal_pt)
                if pt1_dist < closest_dist_to_pt1:
                    closest_dist_to_pt1 = pt1_dist
                    closest_to_pt1 = (i, j)
        return closest_to_pt1[::-1]

    first_point = closest_valid_point_to_pt(points_to_return[0])
    second_point = closest_valid_point_to_pt(points_to_return[1])

    #logger.debug("")
    #logger.debug("Chosen points:", first_point, second_point, imcopy[first_point[0], first_point[1]], imcopy[second_point[0], second_point[1]])
    cv2.circle(imcopy, first_point, 6, (255, 255, 255), 2)
    cv2.circle(imcopy, second_point, 6, (255, 255, 255), 2)
    add_image_to_show(imcopy, 'depth with final points')

    display_images_to_show(dir_name, index)

    return first_point, second_point

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    output_dir = 'vis'
    if os.path.exists(output_dir):
        os.system('rm -r vis')
    os.mkdir(output_dir)

    test_dataset = [cv2.imread(f) for f in glob.glob('/home/justin/yumi/dense-descriptors/basic_seg/data-justin-2/images/*.jpg')] 

    for idx_a in range(len(test_dataset)):
        img = test_dataset[idx_a]
        img, shapexd, without_thresh = perform_all_operations(img)
        # to tensor
        # logger.debug(img.shape)
        img = torch.from_numpy(img).cuda()

        # make image part of 1-batch
        img = img.unsqueeze(0)

        img_a = img
        
        # get mask from nonzero parts of the image
        img_cpu = img_a.cpu().numpy()[0]
        img_masked = np.where(img_cpu > 0.05, img_cpu, 0.0)
        img_masked = torch.from_numpy(img_masked).cuda()
        
        #res, _ = run_inference(img_a, img_cpu, mask_a.astype(np.int8))
        _, _, _ = run_inference(img_a, img_cpu, img_masked, high_density=True)

        # permute color axis
        img_cpu = np.transpose(img_cpu, (1, 2, 0))
        mask_a = np.transpose(mask_a, (1, 2, 0))

        plt.imsave(os.path.join(os.getcwd(), "mask.jpg"), mask_a)
        plt.imsave(os.path.join(os.getcwd(), "img_cpu.jpg"), img_cpu)
        #plt.imsave(os.path.join(os.getcwd(), '%05d.jpg'%idx_a), res)
