from untangling.point_picking import click_points_simple
import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage

STYLUS_OFFSET = np.array([0, -0.0111, 0.1012])

if __name__ == "__main__":
    SPEED = (0.4, 6 * np.pi) # speed of the yumi
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    ) # initialize the interface, set the motion speed of the yumi, tell yumi how to go to initial pose

    iface.home()
    iface.sync() # bug: sometimes skips first command, so sync before moving to second command --> goal of sync but not working
    time.sleep(4) # small fix bc sync not working

    img = iface.take_image() # camera gets snapshot: rgb (color, actually grayscale on photoneo rip) and depth
    print("Choose place points")
    place_1_table, _= click_points_simple(img) #outputs: left and right click


    plt.scatter(place_1_table[0], place_1_table[1], c='r')
    plt.imshow(img.color._data)
    plt.show()

    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi
    place1_table_3d = g.ij_to_point(place_1_table).data # convert pixel coordinates to 3d coordinates
    # print(place1_table_3d.shape, STYLUS_OFFSET.shape)
    place1_offset_3d = place1_table_3d + STYLUS_OFFSET # add stylus offset to 3d coordinates
    place1_intermediate_3d = place1_offset_3d + np.array([0, 0, 0.1]) # add 10 cm to z coordinate
    print(place1_offset_3d)
    place1_intermediate_transform = RigidTransform(
        translation=place1_intermediate_3d,
        rotation= iface.GRIP_DOWN_R,
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )
    # iface.go_pose_plan(r_target=place1_transform)
    iface.go_cartesian(
        l_targets=[place1_intermediate_transform], removejumps=[6])
    iface.sync()


    place1_transform = RigidTransform(
        translation=place1_offset_3d,
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