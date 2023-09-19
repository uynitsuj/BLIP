
from blip_pipeline.add_noise_to_img import run_tracer_with_transform
from blip_pipeline.divergences import get_divergence_pts
from blip_pipeline.refine_pick import refine_pick_pts
from learn_from_demos_ltodo import DemoPipeline
from place import find_grasp_ip_place_point
from untangling.utils.grasp import GraspSelector
from untangling.point_picking import click_points_simple
from push_through_backup2 import get_poi_and_vec_for_push
from push_through_backup2 import perform_push_through
from autolab_core import RigidTransform
from untangling.utils.tcps import *

import numpy as np
import logging 
import argparse
import colorsys
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
import time


def visualize_trace(img, trace):
    img = img.copy()
    def color_for_pct(pct):
        return colorsys.hsv_to_rgb(pct, 1, 1)[0] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[1] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[2] * 255
        # return (255*(1 - pct), 150, 255*pct) if not black else (0, 0, 0)
    for i in range(len(trace) - 1):
        # if trace is ordered dict, use below logic
        if not isinstance(trace, OrderedDict):
            pt1 = tuple(trace[i].astype(int))
            pt2 = tuple(trace[i+1].astype(int))
        else:
            trace_keys = list(trace.keys())
            pt1 = trace_keys[i]
            pt2 = trace_keys[i + 1]
        cv2.line(img, pt1[::-1], pt2[::-1], color_for_pct(i/len(trace)), 4)
    plt.title("Trace Visualized")
    plt.imshow(img)
    plt.show()



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

    fullPipeline = DemoPipeline(viz=False, loglevel=logLevel, initialize_iface=True)
    fullPipeline.iface.open_grippers()
    fullPipeline.iface.sync()
    fullPipeline.iface.home()
    fullPipeline.iface.sync()


    # starting_pixels, analytic_trace_problem = fullPipeline.get_trace_from_endpoint(fullPipeline.endpoints[0])
    # starting_pixels = np.array(starting_pixels)
    # trace_t, noisy_traces = run_tracer_with_transform(img_rgb, starting_pixels)
    # trace_t, noisy_traces = trace_t[0], [nt[0] for nt in noisy_traces]
    # visualize_trace(img_rgb, trace_t)
    # visualize_trace(img_rgb, noisy_traces[0])
    # pick_pts = get_divergence_pts(trace_t, noisy_traces)
    # refined_pick_pts = refine_pick_pts(img_rgb[:,:,0], trace_t, pick_pts, viz=True)

  


    demos = np.load("knot_demos/Sep1_OH.npy", allow_pickle=True)
    i = 0

    choose_ip_point = False
    ip_policy = "push"
    # ip_policy = "grasp"
    # META POLICY
    while i < len(demos):
        img_rgbd = fullPipeline.iface.take_image()
        img_rgb = img_rgbd.color._data

        if choose_ip_point:
            print("CHOOSE IP POINT")
            pick_img, _ = click_points_simple(img_rgbd)
            if pick_img is None:
                refined_pick_pts = []
            else:
                refined_pick_pts = [pick_img]
        
        else:
            fullPipeline.get_endpoints()
            for endpoint in fullPipeline.endpoints:
                print(endpoint)
                plt.scatter(endpoint[0], endpoint[1])
            plt.imshow(img_rgb)
            plt.show()

            # trace from the right endpoint
            right_endpoint = fullPipeline.endpoints[0]
            if fullPipeline.endpoints[1][0] > right_endpoint[0]:
                right_endpoint = fullPipeline.endpoints[1]
            starting_pixels, analytic_trace_problem = fullPipeline.get_trace_from_endpoint(right_endpoint)
            starting_pixels = np.array(starting_pixels)
            trace_t, noisy_traces = run_tracer_with_transform(img_rgb, starting_pixels)
            trace_t, noisy_traces = trace_t[0], [nt[0] for nt in noisy_traces]
            visualize_trace(img_rgb, trace_t)
            visualize_trace(img_rgb, noisy_traces[0])
            visualize_trace(img_rgb, noisy_traces[1])
            visualize_trace(img_rgb, noisy_traces[2])
            visualize_trace(img_rgb, noisy_traces[3])
            visualize_trace(img_rgb, noisy_traces[4])
            visualize_trace(img_rgb, noisy_traces[5])
            visualize_trace(img_rgb, noisy_traces[6])
            pick_pts = get_divergence_pts(trace_t, noisy_traces)
            refined_pick_pts = refine_pick_pts(img_rgb[:,:,0], trace_t, pick_pts, viz=True)

            pick_pts = [[pt[1], pt[0]] for pt in pick_pts]
            refined_pick_pts = [[pt[1], pt[0]] for pt in refined_pick_pts]



            # for pt in pick_pts:
            #     plt.scatter(pt[0], pt[1])
            #     plt.title("Original Divergence Points")
            #     plt.imshow(img_rgb)
            #     plt.show()

            # for pt in refined_pick_pts: 
            #     plt.scatter(pt[0], pt[1])
            #     plt.title("Refined Divergence Points")
            #     plt.imshow(img_rgb)
            #     plt.show()


        while len(refined_pick_pts) > 0:
            pick_img = refined_pick_pts[0]
            if ip_policy == "push":
                poi_trace, cen_poi_vec = get_poi_and_vec_for_push(pick_img, fullPipeline, type="poles", viz=True)
                perform_push_through(poi_trace, cen_poi_vec, img_rgbd, fullPipeline)
                # waypoint1 = np.array([poi_trace[0], poi_trace[1]]) + cen_poi_vec
                # waypoint2 = np.array([poi_trace[0], poi_trace[1]]) - cen_poi_vec*0.65

                # plt.text(waypoint1[0]+7,waypoint1[1]+7, 'pt 1')
                # plt.text(waypoint2[0]+7,waypoint2[1]+7, 'pt 2')
                # plt.scatter(waypoint1[0],waypoint1[1])
                # plt.scatter(waypoint2[0],waypoint2[1])
                # plt.imshow(fullPipeline.img.color._data)
                # plt.show()

                # iface = fullPipeline.iface
                # iface.close_grippers()
                # iface.sync()
                # g = GraspSelector(fullPipeline.img, iface.cam.intrinsics, iface.T_PHOXI_BASE) # initialize grasp selector with image, camera intrinsics, and camera transform calibrated to the phoxi

                # waypoint1 = [int(waypoint1[0]), int(waypoint1[1])]
                # waypoint2 = [int(waypoint2[0]), int(waypoint2[1])]

                # place1 = g.ij_to_point(waypoint1).data # convert pixel coordinates to 3d coordinates
                # place2 = g.ij_to_point(waypoint2).data # convert pixel coordinates to 3d coordinates

                # average_place = np.array(place1) + np.array(place2) / 2
                # if average_place[0] > 590:
                #     # use right Arm
                #     place1_transform = RigidTransform(
                #     translation=place1,
                #     rotation= iface.GRIP_DOWN_R,
                #     from_frame=YK.r_tcp_frame,
                #     to_frame="base_link",
                #     )
                #     place2_transform = RigidTransform(
                #         translation=place2,
                #         rotation= iface.GRIP_DOWN_R,
                #         from_frame=YK.r_tcp_frame,
                #         to_frame="base_link",
                #     )


                #     iface.go_cartesian(
                #     r_targets=[place1_transform], removejumps=[6])
                #     iface.sync()

                #     iface.go_cartesian(
                #         r_targets=[place2_transform], removejumps=[6])
                #     iface.sync()
                # else:
                #     place1_transform = RigidTransform(
                #     translation=place1,
                #     rotation= iface.GRIP_DOWN_R,
                #     from_frame=YK.l_tcp_frame,
                #     to_frame="base_link",
                #     )
                #     place2_transform = RigidTransform(
                #         translation=place2,
                #         rotation= iface.GRIP_DOWN_R,
                #         from_frame=YK.l_tcp_frame,
                #         to_frame="base_link",
                #     )


                #     iface.go_cartesian(
                #     l_targets=[place1_transform], removejumps=[6])
                #     iface.sync()

                #     iface.go_cartesian(
                #         l_targets=[place2_transform], removejumps=[6])
                #     iface.sync()


            if ip_policy == "grasp":
                grasp_left = pick_img
                place_left = find_grasp_ip_place_point(img_rgbd, grasp_left, 300, viz=True)
                place_left = np.array(place_left)
                grasp_left = np.array(grasp_left)
                grasp_left, grasp_right, place_left, place_right = fullPipeline.find_best_arm(grasp_left[::-1], None, place_left[::-1], None)
                if grasp_left is not None:
                    grasp_left = grasp_left[::-1]
                if grasp_right is not None:
                    grasp_right = grasp_right[::-1]
                if place_left is not None:
                    place_left = place_left[::-1]
                if place_right is not None:
                    place_right = place_right[::-1]


                fullPipeline.g = GraspSelector(img_rgbd, fullPipeline.iface.cam.intrinsics, fullPipeline.iface.T_PHOXI_BASE)
    

                l_grasp, r_grasp = fullPipeline.calculate_grasps(grasp_left, grasp_right, fullPipeline.g, fullPipeline.iface)
                fullPipeline.iface.grasp(l_grasp=l_grasp, r_grasp=r_grasp)

                #calculate and execute the places
                if place_left is not None:
                    #switching back to x, y
                    intermediate_transform, place_transform = fullPipeline.calculate_place_transforms(place_left, fullPipeline.g, fullPipeline.iface, "left")
                    
                    fullPipeline.iface.go_cartesian(
                    l_targets=[intermediate_transform], removejumps=[5, 6]
                    )
                    fullPipeline.iface.sync()


                    fullPipeline.iface.go_cartesian(
                    l_targets=[place_transform], removejumps=[5, 6]
                    )
                    fullPipeline.iface.sync()

                    #calculate and execute the places

                if place_right is not None:
                    #switching back to x, y
                    intermediate_transform, place_transform = fullPipeline.calculate_place_transforms(place_right, fullPipeline.g, fullPipeline.iface, "right")

                    fullPipeline.iface.go_cartesian(
                    r_targets=[intermediate_transform], removejumps=[5, 6]
                    )
                    fullPipeline.iface.sync()


                    fullPipeline.iface.go_cartesian(
                    r_targets=[place_transform], removejumps=[5, 6]
                    )
                    fullPipeline.iface.sync() 

            fullPipeline.iface.open_grippers()
            fullPipeline.iface.sync()
            fullPipeline.iface.home()
            fullPipeline.iface.sync()
            time.sleep(2)

            img_rgbd = fullPipeline.iface.take_image()
            img_rgb = img_rgbd.color._data


            
            if choose_ip_point:
                pick_img, _ = click_points_simple(img_rgbd)
                if pick_img is None:
                    refined_pick_pts = []
                else:
                    refined_pick_pts = [pick_img]
            
            else:
                fullPipeline.get_endpoints()
                # trace from the right endpoint
                right_endpoint = fullPipeline.endpoints[0]
                if fullPipeline.endpoints[1][0] > right_endpoint[0]:
                    right_endpoint = fullPipeline.endpoints[1]
                starting_pixels, analytic_trace_problem = fullPipeline.get_trace_from_endpoint(right_endpoint)
                starting_pixels = np.array(starting_pixels)
                trace_t, noisy_traces = run_tracer_with_transform(img_rgb, starting_pixels)
                trace_t, noisy_traces = trace_t[0], [nt[0] for nt in noisy_traces]
                visualize_trace(img_rgb, trace_t)
                visualize_trace(img_rgb, noisy_traces[0])
                visualize_trace(img_rgb, noisy_traces[1])
                visualize_trace(img_rgb, noisy_traces[2])
                visualize_trace(img_rgb, noisy_traces[3])
                visualize_trace(img_rgb, noisy_traces[4])
                visualize_trace(img_rgb, noisy_traces[5])
                visualize_trace(img_rgb, noisy_traces[6])
                pick_pts = get_divergence_pts(trace_t, noisy_traces)
                refined_pick_pts = refine_pick_pts(img_rgb[:,:,0], trace_t, pick_pts, viz=True)

                pick_pts = [[pt[1], pt[0]] for pt in pick_pts]
                refined_pick_pts = [[pt[1], pt[0]] for pt in refined_pick_pts]

        fullPipeline.exec_demo_step(demos, i, "./test_demos/")
        i += 1
    fullPipeline.pull_apart()
    print('DONE')

    


    # for i in range(len(demos)):
    #     fullPipeline.exec_demo_step(demos, i, "./test_demos/")







