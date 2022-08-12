from __future__ import division
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize

sigmoid = torch.nn.Sigmoid()


def inverse_warp(ref_img, flow_map, pix_coords, padding_mode):
    """ Warp reference image

    Args:
        ref_img: the reference images (where to sample pixels) -- (B, 3, H, W)
        flow_map: the optical flow maps -- (B, 2, H, W)
        pix_coords: pixels coordinate grid -- (B, 2, H, W)
        padding_mode: padding_mode
    Returns:
        ref_img_warped: Reference image warped to the target image plane -- (B, 3, H, W)
        valid_points: Boolean array indicating point validity  -- (B, 3, H, W)
    """
    # create pixel id coordinate
    _, _, h, w = flow_map.size()

    pix_warped = pix_coords + flow_map  # (B, 2, H, W)
    grid = pix_warped.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 2)
    grid[..., 0] /= (w - 1)
    grid[..., 1] /= (h - 1)
    grid = 2 * grid - 1
    ref_img_warped = F.grid_sample(ref_img, grid, padding_mode=padding_mode, align_corners=True)

    valid_points = (grid.abs().max(dim=-1)[0] <= 1).unsqueeze(1).repeat(1, 3, 1, 1)

    return ref_img_warped, valid_points


def get_epipolar_new(p1, p2, inv_K, rotation, translation):
    """ Compute epipolar maps
    Args:
        p1: original homogenous coordinates -- (B, 3, H*W)
        p2: warped homogenous coordinates -- (B, 3, H*W)
        inv_K: inverse of intrinsic matrix -- (B, 3, 3)
        rotation: -- (B, 3, 3)
        translation: -- (B, 3)
    Returns:
        epipolar_map: epipolar loss map for one target-reference pair in one scale -- (B, 1, H*W)
    """
    t_x = torch.zeros_like(rotation)
    t_x[..., 0, 1] = -translation[..., 2]
    t_x[..., 1, 0] = translation[..., 2]
    t_x[..., 0, 2] = translation[..., 1]
    t_x[..., 2, 0] = -translation[..., 1]
    t_x[..., 1, 2] = -translation[..., 0]
    t_x[..., 2, 1] = translation[..., 0]

    # mean_t_x = t_x.abs().mean([1, 2], True) + 1e-7
    # t_x = t_x / mean_t_x

    F_matrix = torch.matmul(t_x, rotation)  # (B, 3, 3)
    F_matrix = torch.matmul(torch.transpose(inv_K, -2, -1), torch.matmul(F_matrix, inv_K))  # (B, 3, 3)

    Fp1 = torch.matmul(F_matrix, p1)  # (B, 3, H*W)
    epipolar = (Fp1 * p2).sum(1, True)  # (B, 1, H*W)
    epipolar_map = epipolar / (
                (torch.sum(Fp1[:, :2, :] ** 2, dim=1, keepdim=True) + 1e-10).sqrt() + 1e-10)  # (B, 1, H*W)

    return epipolar_map


def detectron2_similarity_loss(mobile_mask, instances_info):
    pred_mask = get_batch_instance_mask(instances_info)
    myResize = Resize(mobile_mask.size()[2:])
    mask = myResize(pred_mask)
    cross_entropy = - (mask * torch.log(mobile_mask + 1e-10) +
                       (1 - mask) * torch.log(1 - mobile_mask + 1e-10))
    return cross_entropy


def post_pro_epipolar_weighted(epipolar_map, weight=None, threshold=None):
    """ Post-processing operations with weighting and thresholding
    """
    post_epipolar = epipolar_map.clone()
    if threshold is not None:
        post_epipolar /= threshold
    if weight is not None:
        post_epipolar /= weight
    return post_epipolar ** 2


def post_process_epipolar_1(epipolar_map):
    """ Post-processing operations with normalizing and squaring
    """
    b, c, h, w = epipolar_map.size()
    norms = torch.max(epipolar_map.view(b, -1), dim=1, keepdim=True)[0]
    norms = norms[..., None, None].repeat(1, c, h, w)
    epipolar_map /= norms
    return epipolar_map**2


def get_batch_instance_mask(instances_info):
    """"
     Returns:
         mask: instance binary mask,
                (B, C, H, W) for multiple samples or a batch
                (1, C, H, W) for a single sample
    """
    # for multiple samples, process them separately, cause samples have different numbers of instances
    if isinstance(instances_info, list):
        m = []
        for info in instances_info:
            pred_masks = info["instances"].pred_masks  # (N, H, W)
            m.append(torch.sum(pred_masks, dim=0, keepdim=True).repeat(3, 1, 1))
        m = torch.stack(m, dim=0)  # (B, 3, H, W)
    # for single sample
    else:
        pred_masks = instances_info.pred_masks  # Tensor (N,H,W) masks for each detected instance
        # scores = instances_info.scores  # Tensor of N confidence score for each detected instance
        m = torch.sum(pred_masks, dim=0, keepdim=True).repeat(3, 1, 1).unsqueeze(0)  # (1, C, H, W)

    mask = torch.zeros_like(m)
    mask[m != 0] = 1
    return mask


def post_process_epipolar_2(epipolar_map, instances_info):
    """ Post-processing operations with masking by instance binary mask
    Args:
        epipolar_map: epipolar map -- (B, C, H, W)
        instances_info: detectron2 model output,
                        a list of dicts (during training/validation) or
                        a single dict (during evaluation)
    """
    mask = get_batch_instance_mask(instances_info)
    myResize = Resize(epipolar_map.size()[2:])
    mask = myResize(mask)
    return mask * epipolar_map


def create_coords(batch_size=64, height=128, width=416):
    grid_coords = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(grid_coords, axis=0).astype(np.float32)
    id_coords = nn.Parameter(torch.from_numpy(id_coords), requires_grad=False)
    pix_coords = torch.unsqueeze(torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
    pix_coords = pix_coords.repeat(batch_size, 1, 1).view(batch_size, -1, height, width)  # (B, 2, H, W)

    return pix_coords


def smooth_loss(target, mobile):
    """ Smooth loss according to target image gradients
    """

    # (B, 1, H, W)
    grad_img_x = torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True)

    # (B, 1, H, W)
    grad_mobile_x = torch.abs(mobile[:, :, :, :-1] - mobile[:, :, :, 1:])
    grad_mobile_y = torch.abs(mobile[:, :, :-1, :] - mobile[:, :, 1:, :])

    # compute edge-aware loss for mobile probability maps
    grad_mobile_x *= torch.exp(-grad_img_x)
    grad_mobile_y *= torch.exp(-grad_img_y)
    loss = grad_mobile_x.mean() + grad_mobile_y.mean()

    return loss


def derivable_consistency_loss(mobile1, mobile2, threshold=0.5):
    """ Compute the consistency of forward and backward mobile probability mask
    """
    around1 = sigmoid(20 * (mobile1 - threshold))
    around2 = sigmoid(20 * (mobile2 - threshold))
    loss = (around1 - around2)**2
    return loss


def divergence(foreground, feature):
    """
    Args:
        foreground: binary mobile region mask (B, 1, H, W)
        feature: intermediate feature maps (B, C, H, W)
    """
    softmax = nn.Softmax(dim=1)
    dynamic = foreground.expand_as(feature) * feature
    center = torch.mean(dynamic, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

    dy_distribution = softmax(dynamic)  # (B, C, H, W)
    center_distribution = softmax(center).expand_as(dy_distribution)
    div = (dy_distribution * torch.log(dy_distribution / center_distribution + 1e-5)).abs()
    div = div.sum() / foreground.sum()
    return div


def compute_quantiles(flow, cam_T_cam, inv_K, p1, pix_coords, ones, scale_factor, percentage, i, b):
    flow_map = scale_factor * flow[('flow', i, 0)]
    p2 = torch.cat([pix_coords + flow_map, ones], 1).view(b, 3, -1)  # (B, 3, H*W)
    epipolar_map = get_epipolar_new(
        p1, p2, inv_K[:, :3, :3], cam_T_cam[:, :3, :3], cam_T_cam[:, :3, -1]).view(b, -1).abs()  # (B, H*W)
    return torch.quantile(epipolar_map, percentage, dim=1)  # (num_quantile, B)
