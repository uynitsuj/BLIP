from untangling.keypoint_untangling import get_points_testing, save_bounding_boxes
import numpy as np
from untangling.utils.tcps import *
from untangling.utils.circle_BFS import trace
from point_picking import *
import os

def test_untangling_pipeline():
    #get image from data_bank
    data_bank = "../data_bank"
    results_dir = "HULK_vanilla_results"
    if os.path.exists(results_dir):
        os.system('rm -r ' + results_dir)
    os.mkdir(results_dir)
    for folder in sorted(os.listdir(data_bank)):
        # if folder == "complex_simple" or folder == "complex_drop" or folder == "figure8_drop" or folder == "figure8_simple" or folder == "large_figure8_drop" or folder == "large_figure8_simple" or folder == "large_overhand_drop" or :
        print(folder)
        result_folder_path = os.path.join(results_dir, folder)
        test_data_path = data_bank + "/" + folder
        if os.path.exists(result_folder_path):
            os.system('rm -r ' + result_folder_path)
        os.mkdir(result_folder_path)
        idx = 0
        for inner_folder in sorted(os.listdir(test_data_path)):
            if inner_folder == "README.md":
                continue
            img_path = test_data_path + "/" + inner_folder + "/color_0.npy"
            img = np.load(os.path.join(data_bank, img_path))
            img[-130:,:,:]=0
            depth_path = test_data_path + "/" + inner_folder + "/depth_0.npy"
            depth = np.load(os.path.join(data_bank, depth_path))
            # save_bounding_boxes(img, depth=depth)
            get_points_testing(img, depth=depth, dir_name = result_folder_path, index=idx)
            idx += 1

if __name__ == '__main__':
    test_untangling_pipeline()