from __future__ import absolute_import, division, print_function
from .resnet_encoder import ResnetEncoder
import torch
import torch.nn as nn
from collections import OrderedDict

SCALE_TRANSLATION = 0.01
SCALE_ROTATION = 0.01
CONSTRAINT_MIN = 0.001


# POSE BASED ON MONODEPTH2
# TODO: check influence of stride + elu + scale learning + align_corners=True


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1, use_elu=False,
                 scale_trainable=False):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.num_output_channels = 3

        self.pose_convs = OrderedDict()
        self.pose_convs["squeeze"] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.pose_convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.pose_convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.pose_convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        if use_elu:
            self.nonlin = nn.ELU(inplace=True)
        else:
            self.nonlin = nn.ReLU(inplace=True)
        self.pose_net = nn.ModuleList(list(self.pose_convs.values()))

        if scale_trainable:
            self.rotation_scale = nn.Parameter(torch.tensor(0.01))
            self.translation_scale = nn.Parameter(torch.tensor(0.01))
        else:
            self.rotation_scale = torch.tensor(SCALE_ROTATION)
            self.translation_scale = torch.tensor(SCALE_TRANSLATION)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_features):
        last_features = input_features[-1]
        out = self.nonlin(self.pose_convs["squeeze"](last_features))
        for i in range(3):
            out = self.pose_convs[("pose", i)](out)
            if i != 2:
                out = self.nonlin(out)

        out = out.mean([2, 3], keepdim=True)

        out = out.view(-1, self.num_frames_to_predict_for, 1, 6)
        rotation_scale = self.relu(self.rotation_scale - CONSTRAINT_MIN) + CONSTRAINT_MIN
        translation_scale = self.relu(self.translation_scale - CONSTRAINT_MIN) + CONSTRAINT_MIN

        axisangle = rotation_scale * out[..., :3]
        translation = translation_scale * out[..., 3:]

        return axisangle, translation


class PoseNet_v3(nn.Module):

    def __init__(self, num_layers=18, pretrained=True, use_elu=False, scale_trainable=False):
        super(PoseNet_v3, self).__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers,
                                     pretrained=pretrained,
                                     num_input_images=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=2,
                                   use_elu=use_elu, scale_trainable=scale_trainable)

    def init_weights(self):
        pass

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], 1)
        features = self.encoder(x)
        pose = self.decoder(features)
        return pose
