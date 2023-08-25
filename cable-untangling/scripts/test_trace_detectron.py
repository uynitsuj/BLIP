from untangling.utils.interface_rws import Interface
import numpy as np
import matplotlib.pyplot as plt
from untangling.utils.tcps import *
from untangling.point_picking import *
detectron = os.path.dirname(os.path.abspath(__file__)) + "/../../detectron2_repo"
sys.path.insert(0,detectron)
import analysis as loop_detectron
from phoxipy.phoxi_sensor import PhoXiSensor
import argparse
import re
from datetime import datetime
from full_pipeline import get_endpoints

import glob
from PIL import Image

def test_detectron_robot():
    SPEED = (0.6, 6 * np.pi)
    iface = Interface(
        "1703005",
        ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
        ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
        speed=SPEED,
    )
    print("init interface")
    iface.open_grippers()
    iface.open_arms()
    while True:
        input("Enter to try new pic")
        img = iface.take_image()
        _, out_img = loop_detectron.predict(img.color._data, thresh=0.7)
        plt.imshow(out_img)
        plt.title("Detected Knots")
        plt.show()
        # endpoints = get_endpoints(img)
        # for endpoint in endpoints:
        #     network_points(img,  cond_grip_pose=endpoint[0], neck_point=(500,500), interactive_trace = True)
    
if __name__ == '__main__':
    test_detectron_robot()