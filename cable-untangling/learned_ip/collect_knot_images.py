from untangling.point_picking import click_points_simple
import numpy as np
from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
import os

output_dir = "/home/mallika/triton4-lip/cable-untangling/learned_ip/rope_knot_images"
# print(f"Saving images to {output_dir}")
def collection(iface, folder_name, num_samples):
    print(f"Collecting {num_samples} samples for {folder_name}")
    os.mkdir(f"{output_dir}/{folder_name}")
    print("Press enter to start")
    for s in range(num_samples):
        print(f"Collecting sample {s}")
        img = iface.take_image()
        plt.imshow(img.color._data)
        plt.show()
        np.save(f"{output_dir}/{folder_name}/img_{time.time()}", img)

def save_as_jpg(folder):
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            img = np.load(f"{folder}/{file}", allow_pickle=True)
            plt.imshow(img.item().color._data)
            plt.show()
            plt.imsave(f"{folder}/{file}.jpg", img.item().color._data)
            plt.close()


if __name__ == "__main__":
    # print("Starting collection")
    # SPEED = (0.4, 6 * np.pi)
    # iface = Interface(
    #     "1703005",
    #     ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
    #     ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
    #     speed=SPEED,
    # )

    # collection(iface, "knot_no_distractors/step_1", 5)
    # collection(iface, "knot_no_distractors/step_2", 5)

    # collection(iface, "knot_straight_distractors/step_1", 5)
    # collection(iface, "knot_straight_distractors/step_2", 5)

    # collection(iface, "knot_curvy_distractors/step_1", 5)
    # collection(iface, "knot_curvy_distractors/step_2", 5)


    save_as_jpg(f"{output_dir}/knot_curvy_distractors/step_2")
   






