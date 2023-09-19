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
        network_points(img)
        _, out_img = loop_detectron.predict(img.color._data, thresh=0.7)
        plt.imshow(out_img)
        plt.show()
    
def test_detectron_camera():
    cam = PhoXiSensor("1703005")
    cam.start()
    # depth_dir = "temp_collection/depth_images/"
    # color_dir = "temp_collection/color_images/"
    # output_dir = "temp_collection/detectron_output/"
    i = 15 # CHANGE THIS
    # while True:
    start = datetime.now()
    cam_only = True
    for _ in range(100):
        img = cam.read(cam_only)
        cam_only = not cam_only
        #import pdb;pdb.set_trace()
        end = datetime.now()
        print(end - start)
        # plt.imshow(img.color._data)
        # plt.show()
        # _, out_img = loop_detectron.predict(img.color._data, thresh=0.7)
        # plt.imshow(out_img)
        # plt.show()
        # command = input("Press y to save or n to discard")
        # if command == "y":
        #     depth_img = img.depth._data
        #     color_img = Image.fromarray(img.color._data)
        #     color_img.save(color_dir + f"color_{i}.jpg","JPEG")
        #     np.save(depth_dir + f"depth_{i}.npy", depth_img)
        #     out_save = Image.fromarray(out_img)
        #     out_save.save(output_dir + f"ouput_{i}.jpg", "JPEG")
        #     i += 1
        # if command == "n":
        #     continue
def test_output():
    cam = PhoXiSensor("1703005")
    cam.start()
    img = cam.read()
    detectron_out, imout = loop_detectron.predict(img.color._data, thresh=0.05)
    plt.imshow(imout)
    plt.show()
    print(len(detectron_out))
    
def test_detectron_folder():
    output_dir = "/home/justin/yumi/temp_collection/agg/"
    count = 0
    for image in glob.glob("/home/justin/yumi/detectron2_repo/datasets/detectron_agg/test/images/*.png"):
        #print()
        with open(image, 'rb') as file:
          #index = re.search("_(.*).png", os.path.basename(file.name)).group(1)
          img = np.asarray(Image.open(file))
          _, out_img = loop_detectron.predict(img, thresh=0.99)
        # SAVE CODE
          out_save = Image.fromarray(out_img)
          out_save.save(output_dir + f"{count}.jpg", "JPEG")# ouput_{index}
          count+=1
        # DISPLAY CODE
        #   plt.imshow(out_img)
        #   plt.show()

        
def compare():
    rows = 1
    columns = 2
    l1 = sorted(glob.glob("temp_collection/thresh2_0.005/*.jpg"))
    l2 = sorted(glob.glob("temp_collection/color_images/*.jpg"))
    for i in range(len(l1)):
        with open(l1[i], 'rb') as f1, open(l2[i], 'rb') as f2:
            fig = plt.figure(figsize=(14,8))
            fig.add_subplot(rows, columns, 1)
            img1 = np.asarray(Image.open(f1))
            plt.imshow(img1)
            fig.add_subplot(rows, columns, 2)
            img2 = np.asarray(Image.open(f2))
            plt.imshow(img2)
            plt.show()
        
        
def pinch_cage(img, box, keypoints):
    output, out_img = loop_detectron.predict(img, thresh=0.005)
    plt.imshow(out_img)
    plt.show()
    
    return left_pinch, right_pinch
    
def test_pinch_cage():
    #output_dir = "temp_collection/test_pinch_cage/"
    count = 0
    for image in glob.glob("temp_collection/test_pinch_cage/*.jpg"):
        with open(image, 'rb') as file:
          index = re.search("_(.*).jpg", os.path.basename(file.name)).group(1)
          img = np.asarray(Image.open(file))
          
        # SAVE CODE
        #   out_save = Image.fromarray(out_img)
        #   out_save.save(output_dir + f"ouput_{index}.jpg", "JPEG")
        # DISPLAY CODE

          pinch_cage(img)
    
if __name__ == "__main__":
    # run_pipeline()
    #while True:
    # parser = argparse.ArgumentParser()
    # parser.add_argument('mode', type = str)
    # args = parser.parse_args()
    # if args.mode == 'robot':
    #     test_detectron_robot()
    # if args.mode == 'camera':
    #     test_detectron_camera()
    # if args.mode == 'folder':
    #     test_detectron_folder()
    # if args.mode == 'compare':
    #     compare()
    test_detectron_folder()
    
        
    
    
