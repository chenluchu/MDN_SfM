from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from collections import OrderedDict
from .layers import upsample, ConvBlock, Conv3x3


# ENCODER BASED ON MONODEPTH2
# DECODER BASED ON GEONET IMPLEMENTATION
# https://github.com/google-research/google-research/blob/master/depth_from_video_in_the_wild/motion_prediction_net.py
# TODO: check influence of stride + elu + scale learning + align_corners=True


class MobileDecoder(nn.Module):
    def __init__(self,
                 num_ch_enc=None,
                 num_input_features=2,
                 scales=range(4),
                 num_frames_to_predict_for=None,
                 use_elu=True):
        super(MobileDecoder, self).__init__()

        if num_ch_enc is None:
            num_ch_enc = [16, 32, 64, 128, 256, 512]
        self.scales = scales
        self.num_ch_enc = num_ch_enc  # [64, 64, 128, 256, 512]
        self.num_input_features = num_input_features
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.num_output_channels = 1

        if use_elu:
            self.nonlin = nn.ELU(inplace=True)
        else:
            self.nonlin = nn.ReLU(inplace=True)

        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.mobile_convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1]+6*self.num_frames_to_predict_for if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.mobile_convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, use_elu=use_elu)

            # upconv_1  (256*2,256)/(128*2,128)/(64*2,64)/(32*2,32)/(16*2,16)
            num_ch_in = self.num_ch_dec[i]
            num_ch_in += self.num_ch_enc[i]
            num_ch_out = self.num_ch_dec[i]
            self.mobile_convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, use_elu=use_elu)
        for s in self.scales:
            self.mobile_convs[("pred_mobile", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.mobile_net = nn.ModuleList(self.mobile_convs.values())
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.mobile_net:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_features, axisangle, translation, frame_id=0):
        self.outputs = {}
        b, _, h, w = input_features[-1].size()
        axisangle = axisangle.view(b, -1, 1, 1).repeat(1, 1, h, w)  # [B, 3, H, W]
        translation = translation.view(b, -1, 1, 1).repeat(1, 1, h, w)

        name = "mobile"

        # MOBILE PART
        x = torch.cat([input_features[-1], axisangle, translation], 1)
        for i in range(4, -1, -1):
            x = upsample(x)
            x = [self.mobile_convs[("upconv", i, 0)](x)]
            x = x + [input_features[i]]
            x = torch.cat(x, 1)
            x = self.mobile_convs[("upconv", i, 1)](x)

            if i in self.scales:
                out = self.mobile_convs[("pred_mobile", i)](x)
                self.outputs[(name, frame_id, i)] = self.sigmoid(out)

        return self.outputs
