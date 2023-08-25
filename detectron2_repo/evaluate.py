from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from PIL import Image
from torchvision.transforms import functional as func
from detectron2.modeling import build_model
import torch, torchvision
from detectron2.checkpoint import DetectionCheckpointer

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
import fiftyone as fo
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

def get_loop_dicts(img_dir):
    """
    Return a list of dictionaries that contain the information for each image
    in the dataset.
    """
    # Get the list of images in the dataset
    img_list = os.listdir(os.path.join(img_dir, "images"))
    jpg,png = False, False
    if img_list[0].endswith(".jpg"):
        jpg = True
    if img_list[0].endswith(".png"):
        png = True
    img_list = [img_name for img_name in img_list if img_name.endswith(".jpg") or img_name.endswith(".png")]
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

        #read annotation
        if jpg:
            annots_path = os.path.join(img_dir, 'annots', img_name.replace(".jpg", ".xml"))
        if png:
            annots_path = os.path.join(img_dir, 'annots', img_name.replace(".png", ".xml"))
        try:
            with open(annots_path, 'r') as f:
                for line in f:
                    # split line by space
                    label, xmin, ymin, xmax, ymax = list(map(lambda x: int(x), line.split()))

                    # create a mask with 1s inside bounding box
                    mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    mask[ymin:ymax, xmin:xmax] = 1

                    obj = {
                        'bbox': [xmin, ymin, xmax, ymax],
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'category_id': label,
                        'segmentation': pycocotools.mask.encode(np.asarray(mask, order="F")),
                    }
                    objs.append(obj)
                record['annotations'] = objs
        except:
            continue
        dataset_dicts.append(record)

    return dataset_dicts

dataset_name = "/home/justin/yumi/detectron2_repo/datasets/detectron_endpoints"
#for d in ['train', 'test']:
d = "test"
DatasetCatalog.register("loop_" + d, lambda d=d: get_loop_dicts( dataset_name  + "/" + d))
MetadataCatalog.get("loop_" + d).set(thing_classes=["knot"])

loop_metadata = MetadataCatalog.get("loop_test")

# verify dataset loading
dataset_dicts = get_loop_dicts(dataset_name + "/test")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d['file_name'])
    visualizer = Visualizer(img[:,:,::-1], metadata=loop_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite("loop_train_sample.png", out.get_image()[:,:,::-1])

# now we train the model
from detectron2.engine import DefaultTrainer
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
#cfg = get_cfg()
cfg = get_config()

# evaluate the model
cfg.MODEL.WEIGHTS = "/home/justin/yumi/detectron2_repo/models/endpoint_model.pth" 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99 # used to be 0.90   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("loop_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 1
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 16
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (loop). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# cfg.INPUT.MASK_FORMAT = 'bitmask'
# # cfg.MODEL.PIXEL_MEAN = [0.0] * 3
# # cfg.MODEL.PIXEL_STD = [255.0] * 3
# # evaluate the model
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/justin/yumi/detectron2_repo/models/endpoint_model.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)

# COCOEVAL:
#import json

# os.mkdir("/home/justin/yumi/detectron2_repo/datasets/dir0.999") 
# evaluator = COCOEvaluator("loop_test", output_dir="/home/justin/yumi/detectron2_repo/datasets/dir0.999")
# val_loader = build_detection_test_loader(cfg, "loop_test")
# #with open('/home/justin/yumi/cable-untangling/data/eval_results/0825_10_result.json', 'a') as f:
# print(inference_on_dataset(predictor.model, val_loader, evaluator))
    
    
# graph ROC curve
# precision = [0.682, ]
# recell = [0.462, ]
#print(inference_on_dataset(predictor.model, val_loader, evaluator))

# FIFTYONE EVAL:
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = build_model(cfg)
# DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
# # model.load_state_dict(torch.load("/home/justin/yumi/detectron2_repo/models/model_0825.pth"))
# model.to(device)
# model.eval()

#name = "allie_test"
# dataset_dir = "/home/justin/yumi/detectron2_repo/datasets/detectron_agg/test"

# # # Create the dataset
# dataset = fo.Dataset.from_dir(
#     dataset_dir=dataset_dir,
#     dataset_type=fo.types.VOCDetectionDataset,
#     name="test-allie",
# )
# dataset = fo.load_dataset(dataset)

# try with the FiftyOneImageDetectionDataset format! should have asked  first LOL

# for d in ['train', 'test']:
#         DatasetCatalog.register("loop_" + d, lambda d=d: get_loop_dicts( dataset_name  + "/" + d))
#         MetadataCatalog.get("loop_" + d).set(thing_classes=["knot"])

#loop_metadata = MetadataCatalog.get("loop_train")
# evaluate the model
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)

#classes = dataset.default_classes
#import glob
# Add predictions to samples
# session = fo.launch_app(dataset)
# with fo.ProgressBar() as pb:
#     for sample in pb(dataset):
#        # with open(sample, 'rb') as image:
#         # Load image
#         #index = re.search("_(.*).jpg", os.path.basename(file.name)).group(1)
        
#         #glob.glob(f"/home/justin/yumi/detectron2_repo/datasets/detectron_eval/test//.jpg"): f"ouput_{index}.jpg"
#         # run the corresponding image through detectron
#         image = np.array(Image.open(sample.filepath))
#         #img_height,img_width = cv2.imread(sample).shape
#         #image = func.to_tensor(image).to(device)
#         c, h, w = image.shape

#         # Perform inference
#         preds = predictor(image)
#         scores = preds["instances"].scores #.cpu().detach().numpy()
#         boxes = preds["instances"].pred_boxes #.cpu().detach().numpy()

#         # Convert detections to FiftyOne format
#         detections = []
#         for score, box in zip(scores, boxes):
#             # Convert to [top-left-x, top-left-y, width, height]
#             # in relative coordinates in [0, 1] x [0, 1]
#             x1, y1, x2, y2 = box
#             rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

#             detections.append(
#                 fo.Detection(
#                     label="knot",
#                     bounding_box=rel_box,
#                     confidence=score
#                 )
#             )
#         # Save predictions to dataset
#         sample["mask_rcnn"] = fo.Detections(detections=detections)
#         sample.save()
