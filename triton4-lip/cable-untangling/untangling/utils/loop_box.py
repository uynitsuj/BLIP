import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_default_bb(heatmap):
    argmax = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    default_bb = [argmax[1] - 1, argmax[0] - 1, 3, 3, 9]
    # clip to image bounds
    default_bb[0] = max(default_bb[0], 0)
    default_bb[1] = max(default_bb[1], 0)
    default_bb[2] = min(default_bb[2], heatmap.shape[1] - default_bb[0])
    default_bb[3] = min(default_bb[3], heatmap.shape[0] - default_bb[1])
    return default_bb

def get_loop_box(heatmap, largest=1, random=False, connectivity=4,
                sample=False, already_box=None, argmax=False):
    heatmap = heatmap[0][0]

    draw_heatmap = heatmap.copy()
    ref_heatmap = heatmap.copy()

    # for each connected component of the heatmap (contiguously > 0.4)
    # get the bounding box of the connected component
    bboxes = []
    # default bounding box is side length 3 around argmax of heatmap
    default_bb = get_default_bb(heatmap)
    if argmax:
        return default_bb
    while True:
        # first find a mask for the heatmap
        mask = np.where(heatmap > 0.3, 255, 0) #0.25 #0.45

        # convert mask to correct format for connectedComponentsWithStats
        mask = mask.astype(np.uint8)
        # find the largest connected component in the mask
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

        # get the bounding box of the largest connected component if one exists
        if n_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            bbox = stats[largest_label, :]
            # print("Going to pick", bbox)
            bboxes.append(bbox)
        else:
            break

        heatmap[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 0

        # discard the bounding box if it is too small

        if bbox[cv2.CC_STAT_AREA] < 50:
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
        return bboxes[0] if len(bboxes) > 0 else default_bb

    if largest == 1:
        return bboxes[0] if len(bboxes) > 0 else default_bb
    return bboxes[:largest]