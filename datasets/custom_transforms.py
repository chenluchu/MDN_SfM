import numpy as np
import random
from skimage.transform import resize

import torch
from torchvision import transforms


class Compose(object):
    def __init__(self, transformers):
        self.transforms = transformers

    def __call__(self, inputs):
        for t in self.transforms:
            t(inputs)


class ResizeImage(object):
    def __init__(self, shape):
        self.resize = transforms.Resize(shape)

    def __call__(self, inputs):
        for k in list(inputs):
            if 'color' in k:
                inputs[k] = self.resize(inputs[k])


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, inputs):
        if random.random() < 0.5:
            for k in list(inputs):
                if k[0] in ('color', 'seg_class', 'seg_height') or (k[0] == 'seg' and type(inputs[k]) != list):
                    inputs[k] = np.fliplr(inputs[k])
                elif k[0] == 'seg':
                    inputs[k][0] = np.fliplr(inputs[k][0])
                    inputs[k][1] = np.fliplr(inputs[k][1])
            w = inputs[("color", 0, 0)].shape[1]
            inputs[("K", 0)][0, 2] = w - inputs[("K", 0)][0, 2]


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, inputs):
        in_h, in_w, _ = inputs[("color", 0, 0)].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)

        for k in list(inputs):
            if k[0] in ('color', 'seg_height'):
                inputs[k] = resize(inputs[k], (scaled_h, scaled_w))
                inputs[k] = inputs[k][offset_y:offset_y + in_h, offset_x:offset_x + in_w]
            elif k[0] == 'seg_class' or (k[0] == 'seg' and type(inputs[k]) != list):
                inputs[k] = resize(inputs[k], (scaled_h, scaled_w), order=0, preserve_range=True).astype('uint8')
                inputs[k] = inputs[k][offset_y:offset_y + in_h, offset_x:offset_x + in_w]
            elif k[0] == 'seg':
                inputs[k][0] = resize(inputs[k][0], (scaled_h, scaled_w), order=0, preserve_range=True).astype('uint8')
                inputs[k][0] = inputs[k][0][offset_y:offset_y + in_h, offset_x:offset_x + in_w]
                inputs[k][1] = resize(inputs[k][1], (scaled_h, scaled_w), order=0, preserve_range=True).astype('uint8')
                inputs[k][1] = inputs[k][1][offset_y:offset_y + in_h, offset_x:offset_x + in_w]

        inputs[("K", 0)][0] *= x_scaling
        inputs[("K", 0)][1] *= y_scaling
        inputs[("K", 0)][0, 2] -= offset_x
        inputs[("K", 0)][1, 2] -= offset_y


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of
     shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, inputs):
        for k in list(inputs):
            if 'color' in k:
                inputs[k] = torch.from_numpy(inputs[k].transpose(2, 0, 1)).float() / 255


class ColorJitter(object):
    def __init__(self, theta=1):
        self.brightness = (0.9, 1.1)
        self.contrast = (0.85, 1.15)
        self.saturation = (0.85, 1.15)
        self.hue = (-0.1, 0.1)
        self.theta = theta

    def __call__(self, inputs):
        brightness = 0 if random.random() < self.theta else self.brightness
        contrast = 0 if random.random() < self.theta else self.contrast
        saturation = 0 if random.random() < self.theta else self.saturation
        hue = 0 if random.random() < self.theta else self.hue

        color_aug = transforms.ColorJitter(brightness, contrast, saturation, hue)

        for k in list(inputs):
            if 'color' in k:
                inputs[k] = color_aug(inputs[k])


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, inputs):
        for k in list(inputs):
            if 'color' in k:
                for t, m, s in zip(inputs[k], self.mean, self.std):
                    t.sub_(m).div_(s)
