from untangling.point_picking import *
from autolab_core import RigidTransform
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
from untangling.utils.grasp import GraspSelector

def grasp_point(iface):
     while True: 
        iface.open_grippers()
        iface.sync()
        iface.home()
        iface.sync()

        input_chars = input("Press enter when ready to select a grasp point, or press x when you are done")
        if input_chars == 'x':
            break
        iface.open_grippers()
        iface.home()
        iface.sync()

        img = iface.take_image(); 

        print("Choose place points")
        grasp1, grasp2= click_points_simple(img)
        print(grasp1)
        print(grasp2)

        plt.scatter(grasp1[0], grasp1[1], c='r')
        plt.imshow(img.color._data)
        plt.show()


        g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
        l_grasp = g.single_grasp(tuple(grasp1), .009, iface.L_TCP)
        iface.grasp(l_grasp=l_grasp)
        iface.sync()







if __name__ == '__main__':
    SPEED = (0.6, 2 * np.pi)
    iface = Interface(
                "1703005",
                ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
                speed=SPEED,
            )
    grasp_point(iface)