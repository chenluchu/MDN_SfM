# Code based on Monodepth2

from __future__ import absolute_import, division, print_function
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision.transforms import Resize

from .custom_transforms import *


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def normalize(inputs, mean, std):
    for t, m, s in zip(inputs, mean, std):
        t.sub_(m).div_(s)


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path: path to the training data
        filenames: like "2011_09_26/2011_09_26_drive_0101_sync 667 r"
        height
        width
        frame_idxs: [0, -1, 1]
        num_scales
        img_ext
    """

    def __init__(self, data_path, filenames, height, width, frame_idxs, num_scales=4, is_train=False, img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = np.array(filenames).astype(np.string_)
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.img_ext = img_ext

        self.mean = (0.45, 0.45, 0.45)
        self.std = (0.225, 0.225, 0.225)
        # self.full_res_shape = (1242, 375)

        if is_train:
            self.compose = Compose([
                ColorJitter(),
                RandomHorizontalFlip(),
                RandomScaleCrop(),
                ArrayToTensor()
            ])
        else:
            self.compose = None

        self.normalize = Normalize(self.mean, self.std)
        self.seg_img_resize = Resize((375, 1242))

        self.loader = pil_loader

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are tuples or string:

            ("color", <frame_id>, <scale>)          for augmented colour images
            ("K", scale) or ("inv_K", scale)        for camera intrinsics.
            "instance_img"                          for instance model input image -- (1242, 375, 3) with RGB order

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index'

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)

        total num of items in inputs : 4(scales) * 3(frames) + 4(scales) * 2(intrinsics) + 1 = 21
        """
        inputs = {}
        line = str(self.filenames[index], encoding='utf-8').split()
        folder = line[0]
        frame_index = int(line[1]) if len(line) == 3 else 0
        side = line[2] if len(line) == 3 else None

        # original image loading ...
        for i in self.frame_idxs:
            original_image = self.get_color(folder, frame_index + i, side, i == 0)  # color img
            inputs[("color", i, 0)] = cv2.resize(original_image, (self.width, self.height))

        # image data augmentation for single scale
        for scale in range(self.num_scales):
            if scale == 0:
                inputs[("K", 0)] = self.K
                if self.compose is not None:
                    self.compose(inputs)
                inputs["instance_img"] = self.seg_img_resize(255 * inputs[("color", 0, 0)]).permute(1, 2, 0)
                self.normalize(inputs)
                inv_K = torch.linalg.pinv(inputs[("K", 0)])
                inputs[("inv_K", 0)] = inv_K
            else:
                factor = 2 ** scale
                K = inputs[("K", 0)].clone()
                K[0, :] /= factor
                K[1, :] /= factor
                inputs[("K", scale)] = K
                inputs[("inv_K", scale)] = torch.linalg.pinv(K)
                scale_resize = Resize((self.height // factor, self.width // factor))

                for i in self.frame_idxs:
                    inputs[("color", i, scale)] = scale_resize(inputs[("color", i, 0)])

        return inputs

    def get_color(self, folder, frame_index, side, target=False):
        raise NotImplementedError
