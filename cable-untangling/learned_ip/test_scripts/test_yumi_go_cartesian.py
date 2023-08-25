from untangling.point_picking import click_points_simple
import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage

if __name__ == "__main__":
    SPEED = (0.4, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )

    iface.home()
    iface.sync()
    time.sleep(4)

    img = iface.take_image()
    print("Choose place points")
    place_1, _= click_points_simple(img)


    plt.scatter(place_1[0], place_1[1], c='r')
    plt.imshow(img.color._data)
    plt.show()

    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    place1_point = g.ij_to_point(place_1).data
    print(place1_point)
    place1_transform = RigidTransform(
        translation=place1_point,
        rotation= iface.GRIP_DOWN_R,
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )
    # iface.go_pose_plan(r_target=place1_transform)
    iface.go_cartesian(
        l_targets=[place1_transform], removejumps=[6])
    iface.sync()
    iface.sync()
    time.sleep(5)