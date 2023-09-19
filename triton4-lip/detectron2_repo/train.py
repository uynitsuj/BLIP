"""
Based heavily on: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
"""

import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
import pycocotools

# os.environ["CUDA_VISIBLE_DEVICES"]="4"

def get_loop_dicts(img_dir):
    """
    Return a list of dictionaries that contain the information for each image
    in the dataset.
    """
    # Get the list of images in the dataset
    img_list = os.listdir(os.path.join(img_dir, "images"))
    img_list = [img_name for img_name in img_list if img_name.endswith(".jpg")]
    img_list.sort()

    dataset_dicts = []

    # Create a dictionary for each image and its corresponding annots file
    for i, img_name in enumerate(img_list):
        record = {}
        objs = []
        
        # read image and corresponding annotation
        img_path = os.path.join(img_dir, 'images', img_name)
        img_height,img_width = cv2.imread(img_path).shape[:2]

        record['image_id'] = i
        record['file_name'] = img_path
        record['height'] = img_height
        record['width'] = img_width

        # read annotation
        # annots_path = os.path.join(img_dir, 'annots', img_name.replace(".jpg", ".xml"))
        # try:
        #     with open(annots_path, 'r') as f:
        #         for line in f:
        #             # split line by space
        #             label, xmin, ymin, xmax, ymax = list(map(lambda x: int(x), line.split()))

        #             # create a mask with 1s inside bounding box
        #             mask = np.zeros((img_height, img_width), dtype=np.uint8)
        #             mask[ymin:ymax, xmin:xmax] = 1

        #             obj = {
        #                 'bbox': [xmin, ymin, xmax, ymax],
        #                 'bbox_mode': BoxMode.XYXY_ABS,
        #                 'category_id': label,
        #                 'segmentation': pycocotools.mask.encode(np.asarray(mask, order="F")),
        #             }
        #             objs.append(obj)
        #         record['annotations'] = objs
        # except:
        #     continue
        record["annotations"] = []
        dataset_dicts.append(record)

    return dataset_dicts

def get_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("loop_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (loop). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    return cfg

def train():
    dataset_name = "/home/justin/yumi/detectron2_repo/datasets/loop_detectron"
    for d in ['train', 'test']:
        DatasetCatalog.register("loop_" + d, lambda d=d: get_loop_dicts( dataset_name  + "/" + d))
        MetadataCatalog.get("loop_" + d).set(thing_classes=["knot"])

    loop_metadata = MetadataCatalog.get("loop_train")

    # verify dataset loading
    dataset_dicts = get_loop_dicts(dataset_name + "/train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d['file_name'])
        visualizer = Visualizer(img[:,:,::-1], metadata=loop_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite("loop_train_sample.png", out.get_image()[:,:,::-1])

    # now we train the model
    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("loop_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (loop). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    # cfg.MODEL.PIXEL_MEAN = [0.0] * 3
    # cfg.MODEL.PIXEL_STD = [255.0] * 3

    # check command line flags for --eval-only
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    if not parser.parse_known_args()[0].eval_only:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()

    # evaluate the model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = get_loop_dicts( dataset_name + "/test")
    idx = 0
    for d in dataset_dicts:    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=loop_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("preds/%05d.png"%idx, out.get_image()[:,:,::-1])
        idx += 1
        # boxes = outputs["instances"].to("cpu")
        # for box in boxes:
        #     x0, y0, x1, y1 = box
        #     dx = x1 - x0
        #     dy = y1 - y0
        #     crop = im[bounding[1]:bounding[1]+bounding[3], bounding[0]:bounding[0]+bounding[2],:]
        #     cv2.imwrite("preds/%05d.png"%idx, crop) #may have to do [:, :, ::-1]
        #     idx += 1
