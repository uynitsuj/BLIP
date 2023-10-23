import numpy as np
import matplotlib.pyplot as plt
# unholy activities happening below
import os
# dense_path = os.path.dirname(os.path.abspath(__file__)) + "/../dense-descriptors"
# import sys
# sys.path.insert(0, dense_path)
# from untangling.keypoint_untangling import get_good_points_from_image
import matplotlib.pyplot as plt

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
        print("CAM_T", CAM_T)
        point=CAM_T*points_3d[lin_ind]
        print("Clicked point in world coords: ",point)
        if(point.z>.5):
            print("Clicked point with no depth info!")
            return
        if(event.button==1):
            left_coords=coords
        elif(event.button==3):
            right_coords=coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords,right_coords

def click_points_single(img):
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img.color.data)
    ax2.imshow(img.depth.data)
    coords = None
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coord=(xind,yind)
        lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
        nonlocal coords
        if(event.button==1):
            coords=coord
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return coords

def click_points_simple(img):
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img.color.data)
    ax2.imshow(img.depth.data)
    left_coords,right_coords = None,None
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
        nonlocal left_coords,right_coords
        if(event.button==1):
            left_coords=coords
        elif(event.button==3):
            right_coords=coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords, right_coords

def click_points_closest(img, endpoints):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    ax.scatter(endpoints[:, 1], endpoints[:, 0], s=5, c='r')

    left_coords,right_coords = None, None
    left_closest_endpoint, right_closest_endpoint = None, None

    def get_closest_endpoint(coords):
        min_dist = np.inf
        closest_endpoint = None
        for endpoint in endpoints:
            dist = np.linalg.norm(endpoint - coords[::-1])
            if dist < min_dist:
                min_dist = dist
                closest_endpoint = endpoint
        return closest_endpoint

    def onclick(event):
        xind, yind = int(event.xdata),int(event.ydata)
        coords = (xind, yind)
        nonlocal left_coords, right_coords, left_closest_endpoint, right_closest_endpoint
        if event.button == 1:
            left_coords = coords
            left_closest_endpoint = get_closest_endpoint(coords)
        elif event.button == 3:
            right_coords = coords
            right_closest_endpoint = get_closest_endpoint(coords)

    _ = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return left_closest_endpoint, right_closest_endpoint

def click_points_show_points(img, point1, point2):
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.scatter(point1[0], point1[1], s=5, c='r')
    ax1.scatter(point2[0], point2[1], s=5, c='b')
    ax1.imshow(img.color.data)
    ax2.imshow(img.depth.data)
    left_coords,right_coords = None,None
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
        nonlocal left_coords,right_coords
        if(event.button==1):
            left_coords=coords
        elif(event.button==3):
            right_coords=coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords,right_coords


def click_points_zed(img, depth_data, CAM_T, intr):
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img)
    ax2.imshow(depth_data)
    points_3d = intr.deproject(depth_data)
    left_coords,right_coords = None,None
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        lin_ind = int(depth_data.ij_to_linear(np.array(xind),np.array(yind)))
        nonlocal left_coords,right_coords
        point=CAM_T*points_3d[lin_ind]
        print("Clicked point in world coords: ",point)
        if(point.z>.5):
            print("Clicked point with no depth info!")
            return
        if(event.button==1):
            left_coords=coords
        elif(event.button==3):
            right_coords=coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords,right_coords

def random_points(img, CAM_T, intr):
    points_3d = intr.deproject(img.depth)
    def pick_point():
        while True:
            xind,yind=np.random.randint(img.width), np.random.randint(img.height)
            lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
            point=CAM_T*points_3d[lin_ind]
            color=img.color[yind,xind,0]
            if(point.x<.1 or point.z>.3 or color<100):continue
            print("Chose point",point)
            return (xind,yind)
    if np.random.rand()>0:
        #single
        return pick_point(),None
    else:
        #double
        p1 = pick_point()
        p2 = pick_point()
        return p1,p2

def network_points(img, vis=True, dir_name=None, index=None, cond_grip_pose=None,
                   neck_point=None, interactive_trace=False, uncertainty=True):
    #darken the bottom part of the image which includes the robot
    dat=img.color.data.copy()
    img_completely_raw=img.color.data.copy()
    dat[-130:,:,:]=0
    depth = img.depth.data
    #dat[dat<60]=0#blacken dark pixels
    filtered = dat.copy()
    filtered[img.depth.data<.1]=0#blacken pixels with invalid depth info
    lc, rc, trace_uncertain, ensemble_uncertain, ensemble_uncertain_val, pull_apart_dist =get_good_points_from_image(dat, filtered, img_completely_raw, depth=depth,
    dir_name=dir_name, index=index, cond_grip_pose=cond_grip_pose, neck_point=neck_point, interactive_trace=interactive_trace, vis=vis, uncertainty=uncertainty)
    if lc is not None and vis:
        pass
        # plt.imshow(img.depth.data)
        # plt.scatter([lc[0],rc[0]],[lc[1],rc[1]],c='r',s=10)
        # plt.title("HULK-L predicted points to grab")
        # plt.show()
    return lc, rc, trace_uncertain, ensemble_uncertain, ensemble_uncertain_val, pull_apart_dist
