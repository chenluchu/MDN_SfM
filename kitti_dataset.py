# coding=utf-8
# Code based on Monodepth2

from __future__ import absolute_import, division, print_function
from path import Path
import os
import cv2
import numpy as np
import pycocotools
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as T

from .mono_dataset import MonoDataset, pil_loader
from .custom_transforms import Compose, Normalize, ResizeImage
from eval_utils import get_intrinsics
from utils import load_as_float
from detectron2.structures import BoxMode


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def get_color(self, folder, frame_index, side, target=False):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if target:  # take camera intrinsic
            K_path = os.path.join(self.data_path, folder.split('/')[0], "calib_cam_to_cam.txt")
            with open(K_path) as f:
                for line in f.readlines():
                    l = line.split()
                    if l[0] == 'P_rect_0{}:'.format(self.side_map[side]):
                        self.K = np.array(l[1:], dtype=np.float32).reshape((3, 4))[:, :3]
                        break

            self.K = np.vstack((np.hstack((self.K, [[0], [0], [0]])), [0, 0, 0, 1])).astype(np.float32)
            self.K = torch.from_numpy(self.K)
            w, h = color.size
            self.K[0, :] *= (self.width / w)
            self.K[1, :] *= (self.height / h)

        return np.array(color, dtype=np.float32)

    def get_image_path(self, folder, frame_index, side):
        pass


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset for loading the original image in KITTI Raw dataset
    """

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path


class KITTISegDataset(data.Dataset):
    """Dataset for loading images in KITTI semantic instance segmentation benchmark
    """
    def __init__(self, root, decoder, height, width, n=200, phase='training', mean=None, std=None):
        super(KITTISegDataset, self).__init__()
        self.root = Path(root)
        self.n = n
        self.height = height
        self.width = width
        self.full_res_shape = (1242, 375)
        self.phase = phase
        self.decoder = decoder
        self.resize = T.Resize((self.height, self.width))
        if mean is not None and std is not None:
            self.compose = Compose([ResizeImage((self.height, self.width)),
                                    Normalize(mean, std)])
        else:
            self.compose = Compose([ResizeImage((self.height, self.width)),
                                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])

    def __getitem__(self, index):
        """Returns a single validation item from the semantic dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are string or tuples:

            ("color", <frame_id>)          for resized and normalized colour images (128, 416)
            "K" or "inv_K"                 for camera intrinsics (4, 4)
            "instance_img"                 for original colour image, like (375, 1242, 3)
            "annotations"                  for list of dictionaries of instance in target images

        <frame_id> is either:
            an integer 0 for target image and 1 for reference image

        dictionary in "annotations":
            ["bbox"] = 4 corner coordinates of bbx of one instance w.r.t image size (3, 375, 1242)
            ["bbox_mode"] = BoxMode.XYXY_ABS
            ["segmentation"] = 不知道
            ["category_id"] = category id

        total num of items in inputs : 6
        """
        if index >= len(self): raise IndexError
        inputs = {}

        # read images
        img_number = str(index).zfill(6)
        tgt_img_path = self.root.joinpath('data_scene_flow',
                                          self.phase,
                                          'image_2',
                                          img_number + '_10.png')
        next_tgt_img_path = self.root.joinpath('data_scene_flow',
                                               self.phase,
                                               'image_2',
                                               img_number + '_11.png')
        cam_calib_path = self.root.joinpath('data_scene_flow_calib',
                                            self.phase,
                                            'calib_cam_to_cam',
                                            img_number + '.txt')
        instance_imag_path = self.root.joinpath('data_semantics',
                                                self.phase,
                                                'image_2',
                                                img_number + '_10.png')

        inputs[('color', 0)] = torch.from_numpy(load_as_float(tgt_img_path).transpose(2, 0, 1)).float() / 255
        inputs[('color', 1)] = torch.from_numpy(load_as_float(next_tgt_img_path).transpose(2, 0, 1)).float() / 255
        instance_img = np.array(pil_loader(instance_imag_path), dtype=np.float32)
        intrinsics = get_intrinsics(cam_calib_path).astype('float32')[:, :3]

        # scale factor
        instance_img = np.round(cv2.resize(instance_img, self.full_res_shape))
        inputs['instance_img'] = torch.from_numpy(instance_img)

        # image pre-processing
        _, h, w = inputs[('color', 0)].size()
        zoom_y = self.height / h
        zoom_x = self.width / w
        self.compose(inputs)

        # camera calibration
        intrinsics[0] *= zoom_x
        intrinsics[1] *= zoom_y
        K = np.vstack((np.hstack((intrinsics, [[0], [0], [0]])), [0, 0, 0, 1])).astype(np.float32)
        inputs['K'] = torch.from_numpy(K)
        inputs['inv_K'] = torch.linalg.pinv(inputs['K'])

        # get instance bounding box ground-truth
        inputs["annotations"] = []
        mask = instance_img.copy()
        for label in np.unique(instance_img):
            object_dict = {}
            target = np.zeros(instance_img.shape)
            roi = np.zeros(instance_img.shape)
            trainId = self.decoder(label)
            if trainId != 255 and trainId != 0:
                target[:, :][mask == label] = 255
                roi[:, :][mask == label] = 1
                target = target.astype(np.uint8)
                roi = roi.astype(np.uint8)
                contours, _ = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # if len(contours)>1:
                xmin = w
                ymin = h
                xmax = 0
                ymax = 0
                for cont in contours:
                    x, y, w, h = cv2.boundingRect(cont)
                    if x < xmin:
                        xmin = x
                    if y < ymin:
                        ymin = y
                    if x + w > xmax:
                        xmax = x + w
                    if y + h > ymax:
                        ymax = y + h
                object_dict["bbox"] = [xmin, ymin, xmax, ymax]
                object_dict["bbox_mode"] = BoxMode.XYXY_ABS
                object_dict["segmentation"] = pycocotools.mask.encode(np.asarray(roi, order="F"))
                object_dict["category_id"] = trainId - 1
                inputs["annotations"].append(object_dict)

        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)
        return inputs

    def __len__(self):
        return self.n
