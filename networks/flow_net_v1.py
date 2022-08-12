from __future__ import absolute_import, division, print_function
from .resnet_encoder import ResnetEncoder
import torch
import torch.nn as nn
from collections import OrderedDict
from .layers import upsample, ConvBlock, Conv3x3

SCALE_FLOW = 0.1
CONSTRAINT_MIN = 0.001


# ENCODER BASED ON MONODEPTH2
# DECODER BASED ON GEONET IMPLEMENTATION
# https://github.com/google-research/google-research/blob/master/depth_from_video_in_the_wild/motion_prediction_net.py
# TODO: check influence of stride + elu + scale learning + align_corners=True

class FlowDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, scales=range(4), num_frames_to_predict_for=None, use_elu=False,
                 scale_trainable=False):
        super(FlowDecoder, self).__init__()

        self.scales = scales
        self.num_ch_enc = num_ch_enc  # [64, 64, 128, 256, 512]
        self.num_input_features = num_input_features
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.num_output_channels = 2

        if use_elu:
            self.nonlin = nn.ELU(inplace=True)
        else:
            self.nonlin = nn.ReLU(inplace=True)

        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.flow_convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.flow_convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, use_elu=use_elu)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if 0 < i:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.flow_convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, use_elu=use_elu)
        for s in self.scales:
            self.flow_convs[("pred_flow", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
        self.flow_net = nn.ModuleList(self.flow_convs.values())

        if scale_trainable:
            self.flow_scale = nn.Parameter(torch.tensor(0.01))
        else:
            self.flow_scale = torch.tensor(SCALE_FLOW)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_features, frame_id, invert):
        self.outputs = {}
        output_features = []

        name = "flow"
        if invert:
            name += "_inv"

        flow_scale = self.relu(self.flow_scale - CONSTRAINT_MIN) + CONSTRAINT_MIN

        # FLOW PART
        x = input_features[-1]
        output_features.append(x)
        for i in range(4, -1, -1):
            x = upsample(x)
            x = [self.flow_convs[("upconv", i, 0)](x)]
            if 0 < i:
                x = x + [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.flow_convs[("upconv", i, 1)](x)
            output_features.append(x)

            if i in self.scales:
                self.outputs[(name, frame_id, i)] = flow_scale * self.flow_convs[("pred_flow", i)](x)
        # 4, 8, 16, 32, 64, 128
        return self.outputs, output_features[::-1]
    # 128, 64, 32, 16, 8, 4


class FlowNet_v1(nn.Module):

    def __init__(self, num_layers=18, pretrained=True, use_elu=True, scale_trainable=False, n_ch=0):
        super(FlowNet_v1, self).__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers,
                                     pretrained=pretrained,
                                     num_input_images=2,
                                     n_ch=n_ch)
        self.decoder = FlowDecoder(self.encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=1,
                                   use_elu=use_elu, scale_trainable=scale_trainable)
        self.n_ch = n_ch

    def init_weights(self):
        pass

    def forward(self, img1, img2, rigid_warp_img=None, rigid_flow=None, rigid_warp_err=None, frame_id=0, invert=False):
        if self.n_ch == 0:
            x = torch.cat([img1, img2], 1)
        else:
            x = torch.cat([img1, img2, rigid_warp_img, rigid_flow, rigid_warp_err], 1)
        features = self.encoder(x)
        flow, features_connection = self.decoder(features, frame_id, invert)
        return flow, features_connection
