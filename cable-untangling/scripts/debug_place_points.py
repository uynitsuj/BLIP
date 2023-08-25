from untangling.point_picking import *
from autolab_core import RigidTransform
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
from untangling.utils.grasp import GraspSelector


def run_pipeline(iface):
        # iface.open_grippers()
        # iface.home()
        # iface.sync()

        while True: 

            input_chars = input("Press enter when ready to select a click point, or press x when you are done")
            if input_chars == 'x':
                break

            img = iface.take_image(); 
  
            print("Choose place points")
            place_1, _= click_points_simple(img)
            print(place_1)

            plt.scatter(place_1[0], place_1[1], c='r')
            plt.imshow(img.color._data)
            plt.show()


            g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
            place1_point = g.ij_to_point(place_1).data

            print("place1", place_1)
            print("place1_point", place1_point)

            
            place1_transform = RigidTransform(
                translation=place1_point,
                rotation= iface.GRIP_DOWN_R,
                from_frame=YK.l_tcp_frame,
                to_frame="base_link",
            )

            iface.go_cartesian(
            l_targets=[place1_transform],
            )
            iface.sync()
            iface.home()
            iface.sync()
            iface.close_grippers()
            iface.sync()

                
if __name__ == "__main__":
    SPEED = (0.6, 2 * np.pi)
    iface = Interface(
                "1703005",
                ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
                speed=SPEED,
            )
    # print("cam intsinsics", iface.cam.intrinsics.fx, iface.cam.intrinsics.fy, iface.cam.intrinsics.cx, iface.cam.intrinsics.cy, iface.cam.intrinsics.skew, iface.cam.intrinsics.height, iface.cam.intrinsics.width)
    # print("proj mat", iface.cam.intrinsics.K)
    # print("frame", iface.cam.intrinsics.frame)
    # print("T_PHOXI_BASE", iface.T_PHOXI_BASE)
            
    run_pipeline(iface)







 