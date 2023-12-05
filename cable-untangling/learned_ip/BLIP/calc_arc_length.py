import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb 

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate arc length')
    parser.add_argument('--num', type=int, default=1, help='file number to read') ## USE 1, 4, 6, 7, 8, 9
    args = parser.parse_args()
    return args

def get_trace(num):
    trace = np.load('trace_pts' + str(num) + '.npy')
    trace[:,0] = trace[:,0].max() - trace[:,0]
    trace_x = trace[:,1]
    trace_y = trace[:,0]
    trace = np.stack((trace_x, trace_y), axis=1)
    # print(trace.shape)
    return trace

def view_trace(trace):
    plt.scatter(trace[:,0], trace[:,1])
    plt.scatter(trace[0,0], trace[0,1], c='r')
    # plt.scatter(0,0, c='g')
    for i in range(trace.shape[0]):
        plt.annotate(str(i), (trace[i,0], trace[i,1]))
    plt.show()

def find_intersection_line_circle(slope, intercept, center, radius):
    ## ax^2 + bx + c = 0
    a = 1 + slope**2
    b = 2 * (slope * intercept - slope * center[1] - center[0])
    c = center[0]**2 + center[1]**2 + intercept**2 - 2 * intercept * center[1] - radius**2
    x1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    y1 = slope * x1 + intercept
    pt1 = np.array([x1, y1])
    x2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    y2 = slope * x2 + intercept
    pt2 = np.array([x2, y2])
    return pt1, pt2

def distance(a,b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def is_between(a,c,b, eps=1e-5):
    return distance(a,c) + distance(c,b) - distance(a,b) < eps

def get_new_start_pt(trace, new_next_idx, center, radius):
    if trace[new_next_idx][0] - trace[new_next_idx-1][0] != 0:
        slope = (trace[new_next_idx][1] - trace[new_next_idx-1][1]) / (trace[new_next_idx][0] - trace[new_next_idx-1][0])
        intercept = trace[new_next_idx][1] - slope * trace[new_next_idx][0]
        pt1, pt2 = find_intersection_line_circle(slope, intercept, center, radius)
        if is_between(trace[new_next_idx-1], pt1, trace[new_next_idx]):
            new_start_pt = pt1
        else:
            new_start_pt = pt2
    else:
        y1, y2 = center[1] + np.sqrt(radius**2 - (trace[new_next_idx][0] - center[0])**2), center[1] - np.sqrt(radius**2 - (trace[new_next_idx][0] - center[0])**2)
        pt1 = np.array([trace[new_next_idx][0], y1])
        pt2 = np.array([trace[new_next_idx][0], y2])
        if is_between(trace[new_next_idx-1], pt1, trace[new_next_idx]):
            new_start_pt = pt1
        else:
            new_start_pt = pt2
    return new_start_pt

def get_circle(start, next, next_idx, trace, radius=20.0):
    theta = np.arctan2((next[1] - start[1]), (next[0] - start[0]))
    center_x = start[0] + np.cos(theta) * radius
    center_y = start[1] + np.sin(theta) * radius
    center = np.array([center_x, center_y])
    circle = plt.Circle((center_x, center_y), radius, color='r', fill=False)
    new_next_idx = next_idx
    for i in range(next_idx+1, trace.shape[0]):
        pt = trace[i]
        dist = (pt[0]-center[0])**2 + (pt[1]-center[1])**2
        if dist > radius**2 or i == trace.shape[0] - 1:
            new_next_idx = i
            break
    return circle, new_next_idx

def check_circle(trace, circle, new_start_pt=None, new_end_pt=None):
    _, ax = plt.subplots()
    ax.scatter(trace[:,0], trace[:,1])
    # ax.scatter(circle._center[0], circle._center[1], c='g')
    ax.add_patch(circle)
    if new_start_pt is not None:
        ax.scatter(new_start_pt[0], new_start_pt[1], c='g')
    if new_end_pt is not None:
        ax.scatter(new_end_pt[0], new_end_pt[1], c='y')
    plt.show()

def view_circles(trace, circles, click=False, img = None):
    if not click:
        _, ax = plt.subplots()
        ax.imshow(img)
        ax.scatter(trace[:,1], trace[:,0], s=1)
        for i, circle in enumerate(circles):
            ax.add_patch(plt.Circle((circle.center[1], circle.center[0]), radius = 20, color='r', fill=False))
            ax.annotate(str(i), (circle.center[1], circle.center[0]))
        title = str(len(circles)) + ' circles'
        plt.title(title)
        plt.show()
        return None
    else:
        clicked_circle_index = None

        def onclick(event):
            nonlocal clicked_circle_index
            click_x, click_y = event.xdata, event.ydata
            min_dist = float("inf")
            for idx, circle in enumerate(circles):
                dist = np.sqrt((click_x - circle.center[1])**2 + (click_y - circle.center[0])**2)
                if dist < min_dist:
                    min_dist = dist
                    clicked_circle_index = idx
            plt.close()  # Close the figure after a circle is clicked

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.scatter(trace[:,1], trace[:,0], s=1)
        for i, circle in enumerate(circles):
            ax.add_patch(plt.Circle((circle.center[1], circle.center[0]), radius = 20, color='r', fill=False))
            ax.annotate(str(i), (circle.center[1], circle.center[0]))
        title = str(len(circles)) + ' circles. Click circle to store index'
        plt.title(title)

        # Connect the click event to the function
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        return clicked_circle_index


def in_circle(pt, circle):
    dist = (pt[0]-circle._center[0])**2 + (pt[1]-circle._center[1])**2
    return dist <= circle.radius**2

if __name__ == "__main__":
    args = parse_args()
    trace = get_trace(args.num)
    # view_trace(trace)
    circles = []
    start_pt = trace[0]
    next_idx = 1
    while next_idx != trace.shape[0]-1:
        next_pt = trace[next_idx]
        circle, new_next_idx = get_circle(start_pt, next_pt, next_idx, trace)
        new_start_pt = get_new_start_pt(trace, new_next_idx, circle._center, circle.radius)
        start_pt = new_start_pt
        next_idx = new_next_idx
        circles.append(circle)
    
    if not in_circle(trace[-1], circles[-1]):
        radius = circles[-1].radius
        start, next = start_pt, trace[next_idx]
        theta = np.arctan2((next[1] - start[1]), (next[0] - start[0]))
        center_x = start[0] + np.cos(theta) * radius
        center_y = start[1] + np.sin(theta) * radius
        center = np.array([center_x, center_y])
        circle = plt.Circle((center_x, center_y), radius, color='r', fill=False)
        circles.append(circle)
        
    view_circles(trace, circles)