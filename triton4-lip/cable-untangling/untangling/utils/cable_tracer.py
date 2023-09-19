import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import interp
from scipy.interpolate import make_interp_spline, make_lsq_spline, RectBivariateSpline
from autolab_core import PointCloud, Point
from untangling.utils.tcps import ABB_WHITE
import logging

def knot_points(nKnots, x, degree):
    # create the knot locations
    knots = np.linspace(x[0], x[-1], nKnots)
    # we have to add these min and values to   conform by adding preceding and proceeding values
    lo = min(x[0], knots[0])
    hi = max(x[-1], knots[-1])
    augmented_knots = np.append(np.append([lo]*degree, knots), [hi]*degree)
    return augmented_knots


class CableSpline:
    def __init__(self):
        self.points = []
        self.ts = []
        self.spline = None
        self.dirs = []

    def extrapolate(self, delta):
        if self.N == 1:
            return self.points[0]+delta*self.dirs[0]
        extrap_t = self.ts[-1]+delta
        return self.spline(extrap_t)

    def add_point(self, new_point, direction):
        if self.N == 0:
            self.ts.append(0)
            self.dirs.append(direction)
            self.points.append(new_point)
            return
        delta_s = np.linalg.norm(new_point-self.points[-1])
        self.ts.append(self.ts[-1]+delta_s)
        self.points.append(new_point)
        self.dirs.append(direction)
        self._build_spline()
        return delta_s

    def tangent(self, s=None):
        '''
        returns tangent vector to spline at an arc-len s, if None returns direction at end
        '''
        if s is None:
            s = self.ts[-1]
        if self.N == 1:
            return self.dirs[0]
        return self.spline.derivative()(s)

    @property
    def arc_length(self):
        return self.ts[-1]

    def _build_spline(self):
        k = 1 if self.N < 3 else 2
        logger.debug(f"Interpolating on {self.N} pts")
        self.spline = make_interp_spline(self.ts, self.points, k=k)

    def visualize(self, delta_s=.01):
        if self.spline is None:
            return
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ts = np.linspace(0, self.ts[-1], self.N*2)
        pts = self.spline(ts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 'b0')
        extrap_ts = np.linspace(self.ts[-1], self.ts[-1]+delta_s, 5)
        extrap_pts = self.spline(extrap_ts)
        ax.scatter(extrap_pts[:, 0], extrap_pts[:, 1], extrap_pts[:, 2], 'r.')
        plt.show()

    @property
    def N(self):
        return len(self.ts)


class DepthInterp:
    def __init__(self, rgbdimage):
        '''
        xs are
        '''
        self.depth_data = rgbdimage.depth._data
        rows = np.arange(self.depth_data.shape[0])
        cols = np.arange(self.depth_data.shape[1])
        self.depthinterp = RectBivariateSpline(
            rows, cols, self.depth_data, kx=2, ky=2)
        self.visualize()

    def visualize(self):
        vis = np.zeros(self.depth_data.shape).astype(float)
        for row in range(vis.shape[0]):
            for col in range(vis.shape[1]):
                vis[row, col] = self.depth(row, col)
        plt.imshow(vis)
        plt.show()
        vis = np.zeros(self.depth_data.shape)
        for row in range(vis.shape[0]):
            logger.debug("row", row)
            for col in range(2):
                vis[row, col] = self.deriv(row, col, 0, 1)
        plt.imshow(vis)
        plt.show()

    def depth(self, row, col):
        return self.depthinterp.ev(row, col)

    def deriv(self, row, col, dx, dy):
        return self.depthinterp.ev(row, col, dx=dx, dy=dy)


class CableTracer:
    # size of steps to march along with the current curvature/direction estimate
    delta_s: float = .002
    COLOR_MIN = 65

    def __init__(self, rgbd_image, intrinsics, T_CAM_BASE):
        downscale = 1
        if downscale != 1:
            raise Exception(
                "Downscale must be 1 for now otherwise need to change intrinsics")
        self.T_CAM_BASE = T_CAM_BASE
        self.T_BASE_CAM = T_CAM_BASE.inverse()
        self.img = rgbd_image.resize(
            (rgbd_image.height//downscale, rgbd_image.width//downscale))
        self.depth = rgbd_image.depth
        self.color = rgbd_image.color
        self.intr = intrinsics
        self.points_3d = T_CAM_BASE*self.intr.deproject(rgbd_image.depth)
        self.cable_mask = self.color._data[..., 0] > self.COLOR_MIN
        self.depth_interp = DepthInterp(rgbd_image)

    def trace_cable(self, **kwargs):
        def find_nearest_loc(start_point: np.ndarray, dir):
            # dir is the last observed direction of the cable
            loc = self.point_to_ij(
                Point(start_point, frame=self.T_BASE_CAM.from_frame))
            if loc[0] < 0 or loc[1] < 0 or loc[0] > self.img.height or loc[1] > self.img.width:
                return None
            # return loc #for now, just return the same point
            loc = self.point_to_ij(
                Point(start_point, frame=self.T_BASE_CAM.from_frame))
            if np.linalg.norm(self.ij_to_point(loc)._data.flatten()-start_point) < .01:
                return loc
            best_loc = loc
            best_dot = -1
            search_range = 2
            for dx in range(-search_range, search_range+1):
                for dy in range(-search_range, search_range+1):
                    test_loc = loc + np.array((dx, dy))
                    test_point = self.ij_to_point(test_loc)._data.flatten()
                    test_dist = np.linalg.norm(test_point - start_point)
                    test_dir = (test_point-start_point) / \
                        np.linalg.norm(test_point-start_point)
                    dot_with_dir = test_dir.dot(dir)
                    if test_dist < .02 and dot_with_dir > best_dot:
                        best_loc = test_loc
                        best_dot = dot_with_dir
            # logger.debug(
            #     f"Returning point with dot: {best_dot}, locs: {loc}, {best_loc}")
            return best_loc
        pts, centroid, visited = self.segment_cable(kwargs['start_pos'])
        spline = CableSpline()
        dir = self.princ_axis(pts)
        if dir.dot(kwargs['start_direction']) < 0:
            dir *= -1
        spline.add_point(centroid._data.flatten(), dir)
        if kwargs['do_vis']:
            img = self.depth._data
        while spline.arc_length < kwargs['trace_len']:
            # march along the cable and iteratively generate spline points, udpating curvature and accel
            extrap_point = spline.extrapolate(self.delta_s)
            # find closest point with valid depth to extrapolated point
            next_loc = find_nearest_loc(extrap_point, spline.tangent())
            if next_loc is None:
                logger.debug("location out of bounds")
                break  # out of bounds
            dist = np.linalg.norm(
                extrap_point-self.ij_to_point((int(next_loc[0]), int(next_loc[1])))._data.T)
            if dist > .01:
                # if the depth differs a lot from extrapolated point, abort
                logger.debug("Found invalid")
                break
            if kwargs['do_vis']:
                img[next_loc[1], next_loc[0]] = 1
            pts, centroid, visited = self.segment_cable(next_loc)
            dir = self.princ_axis(pts)
            if dir.dot(spline.tangent()) < 0:
                # if the direction we found is backwards, flip it
                dir *= -1
            # No now we detect whether the next point is on a new cable or not
            # .7 corresponds to about 45 degrees
            if dir.dot(spline.tangent()) < .7:
                logger.debug("Detected discontinuity in cable")
                break
            else:
                real_delta_s = spline.add_point(centroid._data.flatten(), dir)
            if real_delta_s < .0001:
                logger.debug("Little progress, exiting")
                break
            if kwargs['vis_floodfill'] and spline.N % 5 == 0:
                for pt in visited:
                    img[pt[1], pt[0]] = 1
                plt.imshow(img)
                plt.show()
        if kwargs['do_vis']:
            plt.imshow(img)
            plt.show()
            spline.visualize()
        return spline

    def ij_to_point(self, loc):
        loc = (int(loc[0]), int(loc[1]))
        lin_ind = self.depth.width*loc[1]+loc[0]
        return self.points_3d[lin_ind]

    def point_to_ij(self, point: Point):
        return self.intr.project(self.T_BASE_CAM*point)

    def segment_cable(self, loc):
        '''
        returns a PointCloud corresponding to cable points along the provided location
        inside the depth image
        '''
        q = [loc]
        pts = []
        closepts = []
        start_point = self.ij_to_point(loc).data
        visited = set()
        RADIUS2 = .01**2  # distance from original point before termination
        CLOSE2 = .002**2
        DELTA = .001  # if the depth changes by this much, stop floodfill
        DELTA_THRESH = .0015  # if the depth is different by this much from start, stop
        NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        # carry out floodfill
        while len(q) > 0:
            next_loc = q.pop()
            next_point = self.ij_to_point(next_loc).data
            visited.add(next_loc)
            diff = start_point-next_point
            dist2 = diff.dot(diff)
            if(dist2 > RADIUS2):
                continue
            pts.append(next_point)
            if(dist2 < CLOSE2):
                closepts.append(next_point)
            # add neighbors if they're within delta of current height
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                if test_loc[0] >= self.depth.width or test_loc[0] < 0 \
                        or test_loc[1] >= self.depth.height or test_loc[1] < 0:
                    continue
                if(test_loc in visited):
                    continue
                test_pt = self.ij_to_point(test_loc).data
                if(abs(test_pt[2]-next_point[2]) < DELTA and
                   self.cable_mask[test_loc[1], test_loc[0]] and
                   abs(test_pt[2]-start_point[2]) < DELTA_THRESH):
                    q.append(test_loc)
        pc = PointCloud(np.array(pts).T, frame=self.T_BASE_CAM.from_frame)
        closepc = PointCloud(np.array(closepts).T,
                             frame=self.T_BASE_CAM.from_frame)
        return pc, closepc.mean(), visited

    def princ_axis(self, points):
        '''
        returns the direction of the principle axis of the points
        points should be a 3xN array
        '''
        # construct moment matrix based on centroid and find the eigen vectors
        centroid = points.mean()
        x = points.x_coords - centroid.x
        y = points.y_coords - centroid.y
        z = points.z_coords - centroid.z
        Ixx = x.dot(x)
        Ixy = x.dot(y)
        Ixz = x.dot(z)
        Iyy = y.dot(y)
        Iyz = y.dot(z)
        Izz = z.dot(z)
        M = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        w, v = np.linalg.eig(M)
        return v[:, np.argmax(w)]


if __name__ == '__main__':
    # t=np.linspace(0,2,20)
    # xs=np.cos(t)
    # ys=np.sin(t)
    # pts=np.vstack((xs,ys)).T
    # s=make_interp_spline(t,pts)
    # evalt=np.linspace(-1,4,40)
    # evalpts=s(evalt)
    # plt.plot(evalpts[:,0],evalpts[:,1])
    # plt.show()
    cable = CableSpline()
    for t in np.linspace(0, .3, 10):
        tan = np.array((-3*np.sin(3*t), 2*np.cos(2*t)))
        cable.add_point(np.array((np.cos(3*t), np.sin(2*t), 0)), direction=tan)
        logger.debug(
            f"tangent: {cable.tangent()}, ground truth: {tan/np.linalg.norm(tan)}")
        # cable.visualize()
