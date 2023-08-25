from scripts.full_pipeline_trunk import FullPipeline
import matplotlib
import matplotlib.pyplot as plt
from untangling.point_picking import *
import argparse
import logging 
from untangling.utils.grasp import GraspSelector
from autolab_core import RigidTransform
from untangling.utils.tcps import *

class IPPipeline(FullPipeline):
    def __init__(self, viz, loglevel, initialize_iface):
        FullPipeline.__init__(self, viz, loglevel, initialize_iface)

    def get_closest_trace_idx(self, pix, trace):
        distances = np.linalg.norm(np.array(trace) - np.array(pix)[None, ...], axis=1)
        return np.argmin(distances)
    
    def get_closest_trace_point(self, pix, trace):
        return trace[self.get_closest_trace_idx(pix, trace)]


    def visualize_trace_and_crossings(self):
        self.img = self.iface.take_image()
        self.get_endpoints()
        trace, crossings = self.get_trace_pixels(self.img, endpoint="right")
        self.visualize_all_crossings(self.img, crossings)
        print("crossings and confidences", crossings)

    
    def visualize_all_crossings(self, img, crossings, file_name='crossings'):
        plt.clf()
        u_clr = 'green'
        o_clr = 'orange'
        for ctr, crossing in enumerate(crossings):
            y, x = crossing['loc']
            if crossing['ID'] == 0:
                plt.scatter(x+1, y+1, c=u_clr, alpha=0.5)
                plt.text(x+10, y+10, str(ctr), fontsize=8, color=u_clr)
            if crossing['ID'] == 1:
                plt.scatter(x-1, y-1, c=o_clr, alpha=0.5)
                plt.text(x-10, y-10, str(ctr), fontsize=8, color=o_clr)
        plt.imshow(img.color._data)

        if self.save:
            plt.savefig(self.output_vis_dir + file_name)

        if self.viz:
            plt.show()

    # from learn from demos
    def get_trace_pixels(self, img, endpoint="both"):
        print("started perception")
        traces = []
        crossings = []
        trace_uncertain = True
        uncertain_endpoint = None

        if len(self.endpoints) == 1:
            endpoints_to_trace_from = self.endpoints
        else:
            if endpoint == "both":
                endpoints_to_trace_from = self.endpoints
            else:
                #checking if x value is less 
                if(self.endpoints[0][1] < self.endpoints[1][1]):
                    left_endpoint = self.endpoints[0]
                    right_endpoint = self.endpoints[1]
                else:
                    left_endpoint = self.endpoints[1]
                    right_endpoint = self.endpoints[0]
                print("the left endpoint is ", left_endpoint)
                print("the right endpoint is ", right_endpoint)

                if endpoint == "left":
                    endpoints_to_trace_from = [left_endpoint]
                else:
                    endpoints_to_trace_from = [right_endpoint]

        #for now assuming endpoint is either left or right, not both
        for point in endpoints_to_trace_from:
            starting_pixels, analytic_trace_problem = self.get_trace_from_endpoint(point)
            starting_pixels = np.array(starting_pixels)
            while analytic_trace_problem:
                uncertain_endpoint = starting_pixels
                print(starting_pixels)
                self.g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
                self.iface.perturb_point_knot_tye(self.img, uncertain_endpoint[::-1], self.g)

                self.img = self.iface.take_image()
                self.get_endpoints()
                
                if len(self.endpoints) == 1:
                    point = self.endpoints[0]

                else:
                    if(self.endpoints[0][1] < self.endpoints[1][1]):
                        left_endpoint = self.endpoints[0]
                        right_endpoint = self.endpoints[1]
                    else:
                        left_endpoint = self.endpoints[1]
                        right_endpoint = self.endpoints[0]
                    if endpoint == "left":
                        point = left_endpoint
                    elif endpoint == "right":
                        point = right_endpoint

                starting_pixels, analytic_trace_problem = self.get_trace_from_endpoint(point)
                starting_pixels = np.array(starting_pixels)
                    
            #once there is no analytical trace problem
            self.tkd._set_data(self.img.color._data, starting_pixels)
            perception_result = self.tkd.perception_pipeline(endpoints=self.endpoints, viz=True, vis_dir=self.output_vis_dir) #do_perturbations=True
            traces.append(self.tkd.pixels)
            crossings.append(self.tkd.detector.crossings)
            
        if len(traces) > 0:
            return traces[0], crossings[0] 
    
    def calculate_place_transform(self, place_point, g, iface, arm):
        place_3d = g.ij_to_point(place_point).data
        #so that it does not place too deep
        place_3d = place_3d + np.array([0, 0, 0.01])

        if arm == "left":
            place_transform = RigidTransform(
            translation=place_3d,
            # rotation=iface.GRIP_DOWN_R,
            rotation = iface.y.left.get_pose().rotation,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
            )

        if arm == "right":
            place_transform = RigidTransform(
            translation=place_3d,
            # rotation=iface.GRIP_DOWN_R,
            rotation = iface.y.right.get_pose().rotation,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
            )
        return place_transform








    # manual grasp and place setting
    def push_ip(self):
        self.iface.open_grippers()
        self.iface.sync()
        self.iface.home()
        self.iface.sync()

        #run trace + crossings perception pipeline
        self.img = self.iface.take_image()
        self.get_endpoints()
        trace_pixels, crossings = self.get_trace_pixels(self.img, endpoint="right")
        print("crossings", crossings)
        np.save(self.output_vis_dir + "crossings.npy", crossings)

                                                        
        self.visualize_all_crossings(self.img, crossings)


        print("Choose grasp points")
        grasp_left, grasp_right= click_points_simple(self.img)
        print(grasp_left)
        print(grasp_right)

        #gets closest trace points (trace is (y, x))
        grasp_left = self.get_closest_trace_point(grasp_left[::-1], trace_pixels)[::-1]
        grasp_right = self.get_closest_trace_point(grasp_right[::-1], trace_pixels)[::-1]

        print("Choose push points ")
        place_left, place_right = click_points_show_points(self.img, grasp_left, grasp_right)
        print(place_left)
        print(place_right)

        #showing both grasp and place points
        plt.plot(grasp_left[0], grasp_left[1], c = 'r')
        plt.plot(place_left[0], place_left[1], c = 'r')
        plt.plot(grasp_right[0], grasp_right[1], c = 'b')
        plt.plot(grasp_left[0], grasp_left[1], c = 'b')
        plt.imshow(self.img.color._data)
        plt.show()


        self.g = GraspSelector(self.img, self.iface.cam.intrinsics, self.iface.T_PHOXI_BASE)
        l_grasp, r_grasp = self.g.double_grasp(tuple(grasp_left), tuple(grasp_right), 0.004, 0.002, self.iface.L_TCP, self.iface.R_TCP, slide0=True, slide1=True)
        self.iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp)
        self.iface.sync()

        # placing
        if place_left is not None:
            place_left_transform = self.calculate_place_transform(place_left, self.g, self.iface, "left")
            self.iface.go_cartesian(
                l_targets=[place_left_transform], removejumps=[5, 6]
            )
            self.iface.sync()
        if place_right is not None:
            place_right_transform = self.calculate_place_transform(place_right, self.g, self.iface, "right")
            self.iface.go_cartesian(
                r_targets=[place_right_transform], removejumps=[5, 6]
            )
            self.iface.sync()

        self.iface.open_grippers()
        self.iface.sync()
        self.iface.home()
        self.iface.sync()

        self.get_endpoints()
        trace_pixels, crossings = self.get_trace_pixels(self.img, endpoint="right")
        print("crossings", crossings)
        
        





    
        





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print more statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO,
    )

    args = parser.parse_args()
    logLevel = args.loglevel
    fullPipeline = IPPipeline(viz=True, loglevel=logLevel, initialize_iface=True)
    fullPipeline.output_vis_dir = './test_tkd/'
    fullPipeline.save = True
    fullPipeline.push_ip()



