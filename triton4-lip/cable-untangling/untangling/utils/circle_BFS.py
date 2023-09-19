import numpy as np
import matplotlib.pyplot as plt
import cv2
from circle_fit import least_squares_circle
import queue as Q
import logging

def cartesian_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_momentum(past_k_points):
    pass

def fit_line(points):
    # perform pca to minimize perpendicular distance
    pass

def fit_circle(points):
    return np.array(least_squares_circle(points)[:3])

def distance_to_circle(point, circle):
    return abs(np.sqrt(np.sum((point - circle[:2]) ** 2)) - circle[2])

def distance_to_line(point, segment):
    # segment is a tuple of two points
    # point is a tuple of two points
    # returns distance from point to line
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    x0, y0 = point
    x1, y1 = segment[0]
    x2, y2 = segment[1]

    A = np.array([x2 - x1, y2 - y1])
    B = np.array([x0 - x1, y0 - y1])

    return np.linalg.norm(np.cross(A, B)) / np.linalg.norm(A)

def get_nearest_white_pixel(color_img, start_point, visited):
    unvisited_pixels = color_img[:, :, 0]*(1 - visited)
    U, V = np.nonzero(unvisited_pixels)
    points = np.array([U, V]).T
    if (U.size == 0):
        return None
    closest_point = np.argmin(np.linalg.norm(points - start_point, axis=1))

    # visualize start_point and closest point
    # logger.debug("RESUMING TRACING")
    # plt.scatter(*points[closest_point][::-1])
    # plt.scatter(*start_point[::-1])
    # copy_img = color_img.copy()
    # copy_img[:, :, 0] = visited

    # plt.imshow(copy_img[:, :, :3])
    # plt.show()

    return points[closest_point]

def trace_white(rgb_img, start_point, visited_points=None, stop_when_bleed=False,
                prev_point=None, stop_when_jump_far=False):
    lenience = 900 #300
    point_count = 0
    all_points = []
    buf, buf_size = [], 20
    queue = Q.Queue()
    queue.put(start_point)
    visited = np.zeros(rgb_img.shape[:2], dtype=np.uint8)

    imsh = rgb_img.copy()
    if prev_point is not None:
        prev_point = np.array(prev_point)
        start_point = np.array(start_point)
        norm = max(0, int(np.linalg.norm(prev_point - start_point) / np.sqrt(2)) - 1)
        for di in range(-norm, norm):
            for dj in range(-norm, norm):
                visited[prev_point[0] + di, prev_point[1] + dj] = 1
                imsh[prev_point[0] + di, prev_point[1] + dj, 2] = 255
    # plt.imshow(imsh)
    # if prev_point is not None:
    #     plt.scatter(*prev_point[::-1])
    # plt.scatter(*start_point[::-1])
    # plt.show()

    if visited_points is not None:
        for pt in visited_points:
            visited[pt[0], pt[1]] = 1

    uncovered_white_pixels = True
    while uncovered_white_pixels:
        while not queue.empty():
            lenience -= 1
            point = queue.get()
            visited[point[0], point[1]] = 1
            all_points.append(point)

            if len(buf) > buf_size:
                buf = buf[-buf_size:]
            buf.append(point)

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if point[0] + dx < 0 or point[0] + dx >= rgb_img.shape[0] or point[1] + dy < 0 or point[1] + dy >= rgb_img.shape[1]:
                        continue
                    if visited[point[0] + dx, point[1] + dy] == 1:
                        continue
                    if rgb_img[point[0] + dx, point[1] + dy, 0] > 0:
                        queue.put((point[0] + dx, point[1] + dy))
                        visited[point[0] + dx, point[1] + dy] = 1

            if len(all_points) % 5 == 0 and stop_when_bleed:
                bufar = np.array(buf)
                if np.linalg.norm(np.mean(bufar[-3:], axis=0) - np.array(start_point)) > 35:
                    # logger.debug(f"Mean moved away, removing lenience after {len(all_points)} traced")
                    # logger.debug("Mean:", np.mean(bufar[:], axis=0), start_point)
                    lenience = min(lenience, 20)
                if lenience < 0 and (np.max(bufar[:, 0]) - np.min(bufar[:, 0]) > 12 or np.max(bufar[:, 1]) - np.min(bufar[:, 1]) > 12):
                    # logger.debug("Stopping due to bleeding")
                    # logger.debug(lenience, len(all_points))
                    return all_points, True

        # nearest_white_pixel, nearest_dist = None, 999999
        # for i in range(visited.shape[0]):
        #     for j in range(visited.shape[1]):
        #         if visited[i, j] == 0 and rgb_img[i, j, 0] > 0 and cartesian_distance((i, j), start_point) < nearest_dist:
        #             nearest_white_pixel, nearest_dist = (i, j), cartesian_distance(point, (i, j))
        
        nearest_white_pixel = get_nearest_white_pixel(rgb_img, point, visited)
        if stop_when_jump_far and np.linalg.norm(np.array(start_point) - np.array(nearest_white_pixel)) > 20:
            logger.debug("Stopping due to jump far")
            break

        buf = []
        if nearest_white_pixel is None:
            uncovered_white_pixels = False
        else:
            queue.put(nearest_white_pixel)
    
    return all_points, False

def trace(rgbd_img, start_point, stop_when_undercrossing=False, circle_tolerance=3,
          depth_tolerance=0.0008, max_steps=15000, stop_when_bleeding=False, prev_point=None):
    depth_img = rgbd_img[:, :, 3]
    points_list = []

    depth = {}
    closest_valid_point, closest_valid_distance = None, 999999999999
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            if depth_img[i, j] != 0 and rgbd_img[i, j, 0] > 255 * 0.30:
                depth[(i, j)] = depth_img[i, j]
                if cartesian_distance((i, j), start_point) < closest_valid_distance:
                    closest_valid_point = (i, j)
                    closest_valid_distance = cartesian_distance((i, j), start_point)
            else:
                depth[(i, j)] = 0

    queue = [closest_valid_point]
    visited = set()

    if prev_point is not None:
        prev_point = np.array(prev_point)
        start_point = np.array(start_point)

        norm = max(0, int(np.linalg.norm(prev_point - start_point) / np.sqrt(2)) - 1)
        for di in range(-norm, norm):
            for dj in range(-norm, norm):
                visited.add((start_point[0] + di, start_point[1] + dj))

    # continuously render image with traversed points
    image = rgbd_img[:, :, :3].copy()
    # cv2.imshow('image', image)
    # cv2.waitKey(1)
    stop_cond = False
    depth_thresh = depth_tolerance #0.0009

    start_lenience = 800 #950
    out_of_endpoint = False
    bleeding = False
    num_good_in_a_row = 0

    k = 200
    hysteresis = 5
    past_k_points = []

    # plt.imshow(depth_img)
    # plt.scatter(*start_point[::-1])
    # plt.show()
    old_point = None
    counter = 0
    while not stop_cond:
        while len(queue) != 0 and (not (stop_when_undercrossing and counter > max_steps)):
            start_lenience -= 1
            counter += 1
            point = queue.pop(0)
            if not point:
                logger.debug("?")
            # logger.debug("On point: ", point)
            points_list.append(point)
            if counter % 1000 == 0:
                logger.debug(counter)
            if len(past_k_points) == k:
                past_k_points = past_k_points[1:] + [list(point) + [depth[point]]]
                circle = fit_circle(past_k_points)
                recent_point = np.mean(past_k_points[-k//3:], axis=0)
                old_point = np.mean(past_k_points[:-k//3], axis=0)
            else:
                if start_lenience < 0:
                    past_k_points.append(list(point) + [depth[point]])
                circle = None
                recent_point = None
                old_point = None
            
            # detect bleeding
            if len(past_k_points) >= 2 and start_lenience < 0:
                dist = np.linalg.norm(np.array(points_list[-1]) - np.array(points_list[-2]))
                thresh = 5
                if dist < thresh:
                    num_good_in_a_row += 1
                if num_good_in_a_row > 5:
                    logger.debug("out of endpoint at step", counter)
                    out_of_endpoint = True
                    num_good_in_a_row = float('-inf')
                    # plt.imshow(rgbd_img[:, :, :3]/255.0)
                    # plt.scatter(*points_list[-1][::-1])
                    # plt.show()
                if out_of_endpoint and dist > thresh + 15:
                    logger.debug("out of endpoint and thresh violation at step", counter
                        , 'pointslist', len(points_list))
                    # show the violation
                    # plt.imshow(rgbd_img[:, :, :3]/255.0)
                    # plt.scatter(*points_list[-1][::-1])
                    # plt.scatter(*points_list[-2][::-1])
                    # plt.show()
                
                    points_list = points_list[:-2*k] if len(points_list) > 2*k + 2 else points_list
                    if stop_when_undercrossing or stop_when_bleeding:
                        bleeding = True
                        break

            adaptive_thresh = 0 #recent_z - older_z if counter > k else 0.0
            recent_depth = np.mean(past_k_points[max(0, -2 + len(past_k_points)):], axis=0)[2] if len(past_k_points) > 0 else 0

            for w in [-1, 0, 1]:
                for h in [-1, 0, 1]:
                    if w == 0 and h == 0 :#or abs(w) + abs(h) != 1:
                        continue
                    if point[0] + w < 0 or point[0] + w >= depth_img.shape[0] or point[1] + h < 0 or point[1] + h >= depth_img.shape[1]:
                        continue
                    new_point = (point[0] + w, point[1] + h)
                    if new_point not in visited and depth[new_point] > 0 and \
                        ((abs(depth[new_point] - recent_depth - 2*adaptive_thresh/k) < depth_thresh and start_lenience < 0) or (depth[new_point] > 0 and start_lenience >= 0)) and \
                        (circle is None or start_lenience > 0 or distance_to_circle(new_point, circle) < circle_tolerance):
                        queue.append(new_point)
                        visited.add(new_point)

        if stop_when_undercrossing or (bleeding and stop_when_bleeding):
            # if old_point is None:
            old_point = np.mean(points_list[-100:-75], axis=0) #np.array(closest_valid_point)
            if recent_point is None:
                recent_point = points_list[-1]
            return points_list, (old_point, recent_point)

        # search across all pixels and find closest one to point that is height-viable
        # then add it to the queue
        closest_point, closest_distance = None, None

        # just choose closest point in depth map
        point_to_start_from = np.array(point)
        # perform BFS to find next starting point
        loc_queue = [tuple(point_to_start_from.tolist())]
        loc_visited = visited.copy()
        loc_visited.add(loc_queue[0])
        amount_searched = 0
        while len(loc_queue) != 0:
            start_lenience -= 1
            amount_searched += 1
            if amount_searched > 1000:
                return points_list, (old_point, recent_point)
            loc_point = loc_queue.pop(0)
            # logger.debug("Loc search on point: ", loc_point)

            cart_dist = cartesian_distance(loc_point, point_to_start_from)
            if depth[loc_point] > 0 and \
                loc_point != tuple(point_to_start_from.tolist()):
                break

            for w in [-1, 0, 1]:
                for h in [-1, 0, 1]:
                    if w == 0 and h == 0 :#or abs(w) + abs(h) != 1:
                        continue
                    if loc_point[0] + w < 0 or loc_point[0] + w >= depth_img.shape[0] or loc_point[1] + h < 0 or loc_point[1] + h >= depth_img.shape[1]:
                        continue
                    new_point = (loc_point[0] + w, loc_point[1] + h)
                    if new_point not in loc_visited and depth[new_point] > 0 and \
                        (circle is None or distance_to_circle(new_point, circle) < circle_tolerance - 0.5 + cart_dist/10) and \
                        (abs(depth[new_point] - recent_depth - 2*adaptive_thresh/k) < depth_thresh * (1 + cart_dist/10) or len(past_k_points) < k or (start_lenience > 0 or depth[new_point] > 0)) and \
                        (recent_point is None or np.dot(np.array(recent_point)[:2] - np.array(old_point)[:2], np.array(new_point)[:2] - np.array(recent_point)[:2]) >= 0):
                        loc_queue.append(new_point)
                        loc_visited.add(new_point)

        closest_point = loc_point
        # logger.debug("closest point found", closest_point)
        if closest_point is None or tuple(point_to_start_from.tolist()) == closest_point:
            break
        queue.append(closest_point)
        visited.add(closest_point)
        #past_k_points = past_k_points[k//2:]
    
    return points_list, (old_point, recent_point)