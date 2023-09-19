import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline,make_lsq_spline
from autolab_core import PointCloud, Point
from untangling.utils.tcps import ABB_WHITE
import logging

class CableSpline:
    def __init__(self,first_point):
        self.points=[first_point]
        self.ts=[0]
        self.spline=None
    def extrapolate(self,delta):
        extrap_t = self.ts[-1]+delta
        return self.spline(extrap_t)
    def add_point(self,new_point):
        delta_s=np.linalg.norm(new_point-self.points[-1])
        self.ts.append(self.ts[-1]+delta_s)
        self.points.append(new_point)
        self._build_spline()
    def arc_length(self):
        return self.ts[-1]
    def _build_spline(self):
        k=1 if len(self.ts)<3 else (2 if len(self.ts)<4 else 3)
        logger.debug(f"Interpolating on {len(self.ts)} pts")
        self.spline=make_interp_spline(self.ts,self.points,k=k)
    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ts=np.linspace(0,self.ts[-1],len(self.ts)*2)
        pts = self.spline(ts)
        ax.scatter(pts[:,0],pts[:,1],pts[:,2],'b0')
        extrap_ts=np.linspace(self.ts[-1],self.ts[-1]+.1,5)
        extrap_pts = self.spline(extrap_ts)
        ax.scatter(extrap_pts[:,0],extrap_pts[:,1],extrap_pts[:,2],'r.')
        plt.show()

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
        self.cable_mask = self.color._data[...,0]>self.COLOR_MIN

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
        spline_points = []
        directions = []
        curvatures = []
        pts, centroid, visited = self.segment_cable(kwargs['start_pos'])
        dir = self.princ_axis(pts)
        if dir.dot(kwargs['start_direction']) < 0:
            dir *= -1
        spline_points.append(centroid._data.flatten())
        directions.append(dir)
        curvatures.append(np.zeros(3))
        if kwargs['do_vis']:
            img = self.depth._data
        while self.geodesic_dist(spline_points) < kwargs['trace_len']:
            # march along the cable and iteratively generate spline points, udpating curvature and accel
            extrap_point = self.extrapolate_spline(
                spline_points, directions, curvatures)
            # find closest point with valid depth to extrapolated point
            next_loc = find_nearest_loc(extrap_point, directions[-1])
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
            if dir.dot(directions[-1]) < 0:
                # if the direction we found is backwards, flip it
                dir *= -1
            # No now we detect whether the next point is on a new cable or not
            # .7 corresponds to about 45 degrees
            if dir.dot(directions[-1]) < .95:
                logger.debug("Detected discontinuity in cable")
                break
                spline_points.append(extrap_point)
                real_delta_s = np.linalg.norm(
                    spline_points[-1]-spline_points[-2])
                directions.append(directions[-1] + real_delta_s*curvatures[-1])
            else:
                spline_points.append(centroid._data.flatten())
                real_delta_s = np.linalg.norm(
                    spline_points[-1]-spline_points[-2])
                directions.append(dir)
            curv = (directions[-1]-directions[-2])/real_delta_s
            curvatures.append(curv)
            if real_delta_s < .0001:
                logger.debug("little progress, exiting")
                curvatures.pop()
                directions.pop()
                spline_points.pop()
                break
            # logger.debug(
            #     f"point: {spline_points[-1]}, loc: {next_loc}, dir: {dir}, curv: {curv}, deltas: {real_delta_s}, pts: {pts.num_points}")
            if kwargs['vis_floodfill'] and len(spline_points) % 5 == 0:
                for pt in visited:
                    img[pt[1], pt[0]] = 1
                plt.imshow(img)
                plt.show()
        if kwargs['do_vis']:
            plt.imshow(img)
            plt.show()
        return spline_points

    def extrapolate_spline(self, spline, dirs, curvs, visited=set()):
        # second order approx of the next point
        # next_p = spline[-1] + self.delta_s * \
        #     dirs[-1] + (self.delta_s**2)*curvs[-1]
        next_p = spline[-1] + self.delta_s * dirs[-1]
        return next_p

    def geodesic_dist(self, spline_points, start=0, end=-1):
        '''
        computes distance between points on the curve
        '''
        if(len(spline_points[start:end]) < 2):
            return 0
        dist = 0
        lastp = spline_points[0]
        for p in spline_points[start+1:end]:
            dist += np.linalg.norm(p-lastp)
            lastp = p
        return dist

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
        DELTA_THRESH = .0015
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

if __name__=='__main__':
    # t=np.linspace(0,2,20)
    # xs=np.cos(t)
    # ys=np.sin(t)
    # pts=np.vstack((xs,ys)).T
    # s=make_interp_spline(t,pts)
    # evalt=np.linspace(-1,4,40)
    # evalpts=s(evalt)
    # plt.plot(evalpts[:,0],evalpts[:,1])
    # plt.show()
    cable=CableSpline(np.array((0,0,0)))
    for t in np.linspace(0,.3,10):
        cable.add_point(np.array((np.cos(3*t),np.sin(2*t),0)))
        cable.visualize()