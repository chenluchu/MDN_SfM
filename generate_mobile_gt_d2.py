# coding=utf-8
import os
import glob
import math
import shutil
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
import PIL.Image as pil
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from path import Path
from os.path import join

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import Resize

from utils import create_dir
from cityscapesscripts.helpers.labels import id2label, trainId2label
# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
import detectron2CustomDataset as CustomDataset


img_resize = Resize((128, 416))


def get_argparser():
    parser = argparse.ArgumentParser()
    # General options
    parser.add_argument("--input", type=str,
                        default="/home/infres/cchu/StageEnsta/kitti/data_semantics/training/image_2",
                        help="chemin d'accès vers fichier d'images")
    parser.add_argument("--pred_output",
                        default="/home/infres/cchu/StageEnsta/mobilenet/output/prediction/detectron2/pred_masks",
                        help="repertoire de sauvegarde des prédictions de segmentation")
    parser.add_argument("--gt_output",
                        default="/home/infres/cchu/StageEnsta/mobilenet/output/mobile_objects_ground_truth",
                        help="directory to save the ground-truth masks")
    parser.add_argument("--no_cuda", type=bool, default=True,
                        help="use CUDA device")

    # Detectron2 options
    parser.add_argument("--dataset", type=str, default='kitti',
                        choices=['kitti', 'cityscapes', 'kitti8'], help='Nom de la dataset')
    parser.add_argument("--ckpt", default='/home/infres/cchu/StageEnsta/mobilenet/log/model_final_detectron2.pth',
                        type=str, help="chemin modele pre-entraine")
    return parser


def init_model_detectron2(args):
    setup_logger()

    ##### Setup model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    if args.dataset == "kitti":
        os.environ['KITTI_SEG_DATASET'] = "~/StageEnsta/kitti/data_semantics/"
        CustomDataset.create_kitti_dataset()
        cfg.DATASETS.TRAIN = ("kitti_seg_instance_train",)
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    elif args.dataset == "kitti8":
        os.environ['KITTI_SEG_DATASET'] = "/media/nicolas/data/KITTI_seg/"
        CustomDataset.create_kitti_dataset8()
        cfg.DATASETS.TRAIN = ("kitti_seg_instance_train8",)
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    if args.ckpt == None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    else:
        # cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        cfg.MODEL.WEIGHTS = args.ckpt
        cfg.INPUT.MIN_SIZE_TEST = 1024
        # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    return cfg, model


def get_prediction(img, cfg, model):
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    if cfg.INPUT.FORMAT == "RGB":
        # whether the model expects BGR inputs or RGB
        img = img[:, :, ::-1]

    height, width = img.shape[:2]
    image = aug.get_transform(img).apply_image(img)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}
    outputs = model([inputs])[0]
    return outputs["instances"].to("cpu")


def draw_box(image, bboxes, classes):
    image = torch.as_tensor(image.transpose(2, 0, 1)).type(torch.uint8)
    labels = ["{}:{}".format(i, trainId2label[cls + 1].name) for i, cls in enumerate(classes)]
    colors = [trainId2label[cls + 1].color for cls in classes]
    return draw_bounding_boxes(image, bboxes.tensor.type(torch.int), labels, colors, width=1, font_size=2)


def collecte_detectron2_masks(dir_path, image, pred_masks):
    create_dir(dir_path)
    cv.imwrite(join(dir_path, "image.png"), image.astype(np.uint8))

    for i, pred_mask in enumerate(pred_masks):
        mask = torch.zeros_like(pred_mask, dtype=torch.uint8)
        mask[pred_mask] = 255
        mask = mask.unsqueeze(0).permute(1, 2, 0).repeat(1, 1, 3).numpy().astype(np.uint8)
        cv.imwrite(join(dir_path, "{}.png".format(i)), mask)


def predict(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ##### Setup model detectron2
    cfg, model = init_model_detectron2(args)
    model.eval()

    ##### Setup dataloader
    image_files = []
    if os.path.isdir(args.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob.glob(os.path.join(args.input, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(args.input):
        image_files.append(args.input)
    image_files.sort()

    # image_files = image_files[:1]
    print("There are %d images to predict." % len(image_files))
    with torch.no_grad():
        for i, file in enumerate(tqdm(image_files)):
            img_bgr = cv.imread(file)

            # Output object detection
            detectron_output = get_prediction(img_bgr, cfg, model)
            bboxes = detectron_output.pred_boxes  # Tensor (N,4) bboxes for each detected instance, format (x1,y1,x2,y2)
            pred_masks = detectron_output.pred_masks  # Tensor (N,H,W) masks for each detected instance
            # scores = detectron_output.scores  # Tensor of N confidence score for each detected instance
            classes = detectron_output.pred_classes  # Tensor of N labels for each detected instance
            # cls = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None)[int(obj_cls)]

            pred_img = draw_box(img_bgr, bboxes, classes.numpy()).permute(1, 2, 0).numpy().astype(np.uint8)
            # cv.imwrite(os.path.join(args.pred_output, "detectron2_pred_imgs", str(i)+".png"),
            #            pred_img.permute(1, 2, 0).numpy().astype(np.uint8))

            dir_path = join(args.pred_output, str(i))
            collecte_detectron2_masks(dir_path, pred_img, pred_masks)


def generate_maksks(args, instance_numbers, n_samples=200):
    """ create ground-truth mask with values 0 or 1 and its original image shape.
    """
    assert len(instance_numbers) == n_samples, "Invalid instance numbers input!"
    for n in tqdm(range(n_samples)):
        gt_mask = np.zeros((1, 1, 1))
        for num in instance_numbers[n]:
            mask = cv.imread(join(args.pred_output, "{}/{}.png".format(n, num)))
            if gt_mask.shape != mask.shape:
                gt_mask = np.zeros_like(mask)
            gt_mask[mask != 0] = 255

        # img = cv.imread(join(args.input, "{:06d}_10.png".format(n)))
        # viz = np.hstack([img, 255 * gt_mask, gt_mask*img]).astype(np.uint8)
        cv.imwrite(join(args.gt_output, "{}.png".format(n)), gt_mask.astype(np.uint8))


if __name__ == "__main__":
    args = get_argparser().parse_args()
    # predict(args)

    file_path = join(os.getcwd(), args.gt_output, "instance_numbers.txt")
    with open(file_path, 'r') as f:
        instance_numbers = [l.split() for l in f.readlines()]
        generate_maksks(args, instance_numbers)

