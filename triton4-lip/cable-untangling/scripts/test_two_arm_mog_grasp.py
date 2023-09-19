from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import numpy as np

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

if __name__ == "__main__":
    SPEED = (0.3, 3 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )

    # iface.open_arms()
    iface.home()

    lp = l_p([0.5, 0.2, 0.05], RigidTransform.x_axis_rotation(-np.pi/4) @ Interface.GRIP_DOWN_R)
    rp = r_p([0.5, -0.2, 0.05], RigidTransform.x_axis_rotation(np.pi/4) @ Interface.GRIP_DOWN_R)
    iface.open_grippers()

    iface.go_cartesian(l_targets=[lp], r_targets=[rp])
    iface.sync()

    in_dist = 0.04
    lp = l_p([0.5, in_dist, 0.05], RigidTransform.x_axis_rotation(-np.pi/2.5) @ Interface.GRIP_DOWN_R)
    rp = r_p([0.5, -in_dist, 0.05], RigidTransform.x_axis_rotation(np.pi/2.5) @ Interface.GRIP_DOWN_R)
    iface.open_grippers()

    iface.go_cartesian(l_targets=[lp], r_targets=[rp])
    iface.sync()

    lp = l_p([0.5, in_dist, 0.2], RigidTransform.x_axis_rotation(-np.pi/2.5) @ Interface.GRIP_DOWN_R)
    rp = r_p([0.5, -in_dist, 0.3], RigidTransform.x_axis_rotation(np.pi/2.5) @ Interface.GRIP_DOWN_R)
    iface.open_grippers()

    iface.go_cartesian(l_targets=[lp], r_targets=[rp])
    iface.sync()



class GraspSelector:
    def __init__(self, rgbd_image, intrinsics, T_CAM_BASE,grabbing_endpoint=False, topdown=False):
        self.grabbing_endpoint = grabbing_endpoint
        if topdown:
            self.img = erode_image(rgbd_image[:, :, :3], kernel=(3, 3))
        else:
            self.img = rgbd_image
        self.depth = self.img.depth
        self.color = self.img.color
        self.intr = intrinsics
        self.points_3d = T_CAM_BASE*self.intr.deproject(self.img.depth)
        self.T_CAM_BASE = T_CAM_BASE
        self.col_interface = col_interface
        self.col_interface.setup()
        self.col_interface.set_scancloud(self.points_3d)
        self.yk = YuMiKinematics() nzxm,mn 
        self.planner = Planner()
        self.topdown = topdown