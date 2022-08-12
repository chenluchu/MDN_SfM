# The code is based on https://github.com/ClementPinard/SfmLearner-Pytorch
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F

from networks.layers import SSIM, get_scale_factor
from loss_utils import *


class LossModule(nn.Module):
    def __init__(self, opt, batch=None, ssim=None, padding_mode="zeros", cuda=True):
        super(LossModule, self).__init__()
        self.options = opt
        self.ssim = ssim
        self.alpha = opt.alpha
        self.padding_mode = padding_mode
        self.device = torch.device("cuda" if cuda else "cpu")

        self.losses = {"consis": 0, "epip": 0, "smooth": 0}
        self.outputs = {"warps": {}, "diffs": {}, "valids": {}, "epipolars": {}, "flows": {}, "epipolar_ori": {}}
        if batch is None:
            self.pix_coords = self.create_coords(opt.batch_size, opt.height, opt.width)
        else:
            self.pix_coords = self.create_coords(batch, opt.height, opt.width)

    def forward(self, inputs, frame_ids, flow, mobile, instances_info, cam_T_cam, scale):
        """
        :param inputs: target/reference/intrinsic
        :param frame_ids: without 0
        :param flow: multi frame flow map -- F * (B, 2, H, W)
        :param mobile: frame averaged mobile mask -- (B, 1, H, W)
        :param instances_info: instance segmentation bboxes, masks, classes and scores
        :param cam_T_cam: all frames camera extrinsic matrix -- F * (B, 4, 4)
        :param scale: current scale
        """
        tgt_img = inputs[("color", 0, scale)]
        b, _, h, w = tgt_img.size()
        self.pix_coords = self.create_coords(b, h, w)
        avg_factor = 2 ** scale
        scale_factor = get_scale_factor(b, h, w).to(self.device)

        for i in frame_ids:
            f = scale_factor * flow[('flow', i, scale)]
            ro = cam_T_cam[i][:, :3, :3]
            tran = cam_T_cam[i][:, :3, -1]

            # if not self.options.disable_photoloss:
            #     photo_frame_loss, warped, diff, valid = self.photo_metric_loss(tgt_img, ref_img, f)
            #     self.losses["photo"] = self.losses["photo"] + (photo_frame_loss / avg_factor)

            if not self.options.disable_smoothloss:
                smooth_frame_loss = smooth_loss(tgt_img, mobile)
                # print("SmoLoss/", self.scale, "  ", smooth_frame_loss, "\n")
                self.losses["smooth"] = self.losses["smooth"] + (smooth_frame_loss / avg_factor)

            epip_frame_loss, epip_map, epip_ori = self.epipolar_loss(
                f, mobile, instances_info, inputs[("inv_K", scale)], ro, tran)
            self.losses["epip"] = self.losses["epip"] + (epip_frame_loss / avg_factor)

            if scale == 0:
                # self.outputs["warps"][(i, scale)] = warped
                # self.outputs["diffs"][(i, scale)] = diff
                # self.outputs["valids"][(i, scale)] = valid
                self.outputs["epipolars"][(i, scale)] = epip_map
                self.outputs["flows"][(i, scale)] = f
                self.outputs["epipolar_ori"][(i, scale)] = epip_ori

    def single_mobile_mask_forward(self, inputs, frame_id, flow, mobile, instances_info, cam_T_cam, scale):
        """
        :param inputs: target/reference/intrinsic
        :param frame_id: current frame id
        :param flow: multi frame flow map -- F * (B, 2, H, W)
        :param mobile: frame mobile mask -- (B, 1, H, W)
        :param instances_info: instance segmentation bboxes, masks, classes and scores
        :param cam_T_cam: all frames camera extrinsic matrix -- F * (B, 4, 4)
        :param scale: current scale
        """
        tgt_img = inputs[("color", 0, scale)]
        b, _, h, w = tgt_img.size()
        self.pix_coords = self.create_coords(b, h, w)
        avg_factor = 2 ** scale
        scale_factor = get_scale_factor(b, h, w).to(self.device)

        f = scale_factor * flow[('flow', frame_id, scale)]
        ro = cam_T_cam[frame_id][:, :3, :3]
        tran = cam_T_cam[frame_id][:, :3, -1]

        # if not self.options.disable_photoloss:
        #     photo_frame_loss, warped, diff, valid = self.photo_metric_loss(tgt_img, ref_img, f)
        #     self.losses["photo"] = self.losses["photo"] + (photo_frame_loss / avg_factor)

        if not self.options.disable_smoothloss:
            smooth_frame_loss = smooth_loss(tgt_img, mobile)
            # print("SmoLoss/", self.scale, "  ", smooth_frame_loss, "\n")
            self.losses["smooth"] = self.losses["smooth"] + (smooth_frame_loss / avg_factor)

        epip_frame_loss, epip_map, epip_ori = self.epipolar_loss(
            f, mobile, instances_info, inputs[("inv_K", scale)], ro, tran)
        self.losses["epip"] = self.losses["epip"] + (epip_frame_loss / avg_factor)

        if scale == 0:
            self.outputs["epipolars"][(frame_id, scale)] = epip_map
            self.outputs["flows"][(frame_id, scale)] = f
            self.outputs["epipolar_ori"][(frame_id, scale)] = epip_ori

    def photo_metric_loss(self, target, reference, flow_map):
        ref_img_warped, valid_points = inverse_warp(reference, flow_map, self.pix_coords, self.padding_mode)
        diff_map = (target - ref_img_warped).abs() * valid_points
        loss = diff_map.mean()
        if self.ssim is not None:
            loss = 0.15 * loss + 0.85 * self.ssim(target, ref_img_warped).mean()
        # print("photo loss {:.2f}".format(loss))

        return loss, ref_img_warped, diff_map, valid_points

    def epipolar_loss(self, flow_map, mobile_mask, instances_info, inv_K, ro, tran):
        b, _, h, w = flow_map.size()

        ones = torch.ones_like(mobile_mask)  # (B, 1, H, W)
        p1 = torch.cat([self.pix_coords, ones], 1).view(b, 3, -1)  # (B, 3, H*W)
        p2 = torch.cat([self.pix_coords + flow_map, ones], 1).view(b, 3, -1)  # (B, 3, H*W)
        epipolar_map = get_epipolar_new(p1, p2, inv_K[:, :3, :3], ro, tran).view(b, 1, h, w).abs()  # (B, 1, H, W)
        epipolar_post = post_process_epipolar_1(epipolar_map)  # (B, 1, H, W)
        # epipolar_post = post_process_epipolar_2(epipolar_map, instances_info)  # (B, 1, H, W)

        background = 1 - mobile_mask
        epipolar = (background * epipolar_post).mean()
        non_trivial = (mobile_mask * torch.log(background + 1e-5)).abs().mean()
        # loss = epipolar + self.alpha * non_trivial

        cross_entropy_loss = detectron2_similarity_loss(mobile_mask, instances_info).mean()
        loss = epipolar + self.alpha * non_trivial + self.options.w_d2_sim * cross_entropy_loss

        # print("EpiLoss: epipolar {:.5f} | nontrivial {:.5f} | cross_entropy {:.5f}\n".format(
        #     epipolar, self.alpha*non_trivial, cross_entropy_loss))

        return loss, epipolar_post.expand(b, 3, h, w), epipolar_map.expand(b, 3, h, w)

    def consistency_loss(self, mobile1, mobile2, scale):
        """Compute the difference between mobile mask from two frames
        :param mobile1: mobile mask --  (B, 1, H, W)
        :param mobile2: mobile mask --  (B, 1, H, W)
        :param scale: current scale
        """
        loss = derivable_consistency_loss(mobile1, mobile2).mean()
        self.losses["consis"] = self.losses["consis"] + loss / (2 ** scale)
        # print("ConsisLoss/", scale, "  ", loss, "\n")

    def create_coords(self, batch_size=64, height=128, width=416):
        grid_coords = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(grid_coords, axis=0).astype(np.float32)
        id_coords = nn.Parameter(torch.from_numpy(id_coords), requires_grad=False)
        pix_coords = torch.unsqueeze(torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
        self.pix_coords = pix_coords.repeat(batch_size, 1, 1).view(batch_size, -1, height, width)  # (B, 2, H, W)

        return self.pix_coords.to(self.device)


class Loss(nn.Module):
    def __init__(self, opt, no_ssim=True, padding_mode="zeros", alpha=1):
        super(Loss, self).__init__()
        if not no_ssim:
            self.ssim = SSIM()

        self.opt = opt
        self.alpha = alpha
        self.padding_mode = padding_mode

    def forward(self, inputs, frame_id, flow, mobile, instances_info, scales, cam_T_cam):

        loss_compute = LossModule(self.opt, ssim=self.ssim, padding_mode=self.padding_mode)
        min_mobiles, avg_mobiles, max_mobiles = {}, {}, {}

        for s in scales:
            m1 = mobile[("mobile", -1, s)]
            m2 = mobile[("mobile", 1, s)]
            min_mobiles[s] = torch.cat([m1, m2], dim=1).min(1, True)[0]

            if not self.opt.disable_consisloss:
                loss_compute.consistency_loss(m1, m2, s)

            if self.opt.disable_min:
                for i in frame_id:
                    loss_compute.single_mobile_mask_forward(
                        inputs, i, flow, mobile[("mobile", i, s)], instances_info, cam_T_cam, s)
            else:
                loss_compute(inputs, frame_id, flow, min_mobiles[s], instances_info, cam_T_cam, s)

        losses = loss_compute.losses
        losses["loss"] = self.opt.w_e * losses["epip"] + \
                         self.opt.w_s * losses["smooth"] + \
                         self.opt.w_c * losses["consis"]  # + \
        # self.opt.w_p * losses["photo"]

        outputs = loss_compute.outputs
        outputs["min_mobiles"] = min_mobiles

        # line = "epip_loss : {:.4f} | smooth_loss : {:.4f} | consis_loss : {:.4f}\n"
        # print(line.format(
        #     self.opt.w_e * losses["epip"],
        #     self.opt.w_s * losses["smooth"],
        #     self.opt.w_c * losses["consis"]))

        return outputs, losses
