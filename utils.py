import os
import numpy as np
import png
from imageio import imread

import torch
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes

from cityscapesscripts.helpers.labels import trainId2label
import detectron2.data.transforms as T


# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'.encode()
UNKNOWN_FLOW_THRESH = 1e7
EPS = 1e-7
LC = [[0, 0.0625, 49, 54, 149],
      [0.0625, 0.125, 69, 117, 180],
      [0.125, 0.25, 116, 173, 209],
      [0.25, 0.5, 171, 217, 233],
      [0.5, 1, 224, 243, 248],
      [1, 2, 254, 224, 144],
      [2, 4, 253, 174, 97],
      [4, 8, 244, 109, 67],
      [8, 16, 215, 48, 39],
      [16, 1000000000.0, 165, 0, 38]]


def draw_box(image, detectron_output):
    """Draw bounding box on image

    Args:
        image: a torch tensor (C, H, W) uint8
        detectron_output: a dict
            bboxes is Tensor (N,4) bouding boxes for each detected instance, format (x1,y1,x2,y2)
            scores is Tensor of N confidence score for each detected instance
            classes is Tensor of N labels for each detected instance
    """
    bboxes = detectron_output.pred_boxes
    scores = detectron_output.scores
    classes = detectron_output.pred_classes

    labels = ["{}:{}".format(trainId2label[cls.item()].name, scores[i].item()) for i, cls in enumerate(classes)]
    colors = [trainId2label[cls.item()].color for cls in classes]
    seg_img = draw_bounding_boxes(image.cpu(), bboxes.tensor.type(torch.int), labels, colors, width=2, font_size=4)

    return seg_img.type(torch.uint8)


def get_detectron2_input(images, cfg):
    """Create suitable inputs for the detectron2 network

    Args:
        images: batch of image  -- numpy array (B, H, W, C)
        cfg: detectron2 model configuration
    """
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    # whether the model expects BGR inputs or RGB
    if cfg.INPUT.FORMAT == "BGR":
        images = images.flip(3)

    _, height, width, _ = images.size()
    inputs = []
    for image in images:
        img = aug.get_transform(image).apply_image(image)  # (H, W, C)
        img = img.permute(2, 0, 1)
        inputs.append({"image": img, "height": height, "width": width})

    return inputs


def load_as_float(path):
    return imread(path).astype(np.float32)


def write_result2(f, errs, err_names):
    """Write result into a txt file

    Args:
        f (IOWrapper)
        errs (list): [ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot]
        err_names: name
    """
    lines = []
    for err, name in zip(errs, err_names):
        l = "{}: \t".format(name)
        for i in err:
          l += "{:.3f}  ".format(i)
        lines.append(l + "\n")

    for line in lines:
        f.writelines(line)


def binary_image(x, threshold=0.5):
    bi_x = torch.zeros_like(x)
    bi_x[x >= threshold] = 1
    return bi_x


def write_result(f, errs, err_names):
    """Write result into a txt file

    Args:
        f (IOWrapper)
        seq (int): sequence number
        errs (list): [ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot]
        err_names: name
    """
    lines = []
    for err, name in zip(errs, err_names):
        lines.append("{}: \t {:.3f} \n".format(name, err))

    for line in lines:
        f.writelines(line)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def flow_to_image(flow, max_rad=None):
    """Convert flow into middlebury color code image

    Args:
        flow: optical flow map (H, W, 3), while first two chanel are pixel displacement.
    Return:
        img: optical flow image in middlebury color, numpy array (H, W, C)
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    if max_rad is None:
        rad = np.sqrt(u ** 2 + v ** 2)
        max_rad = max(-1, np.max(rad))

    # print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(max_rad + np.finfo(float).eps)
    v = v/(max_rad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def get_flow_error_image(gt_flow_occ, valid_flow_occ, valid_flow_noc, flow):
    h, w, _ = flow.shape
    flow_err_img = np.zeros([h, w, 3])

    dfu = gt_flow_occ[..., 0] - flow[..., 0]
    dfv = gt_flow_occ[..., 1] - flow[..., 1]
    f_err = np.sqrt(dfu**2 + dfv**2)
    f_mag = np.sqrt(gt_flow_occ[..., 0]**2 + gt_flow_occ[..., 1]**2) + 1e-6
    n_err = np.minimum(f_err / 3.0, 20.0 * f_err / f_mag)
    for i in range(len(LC)):
        cond = np.logical_and((LC[i][0] <= n_err), (n_err < LC[i][1]))
        flow_err_img[cond, 0] = LC[i][2]
        flow_err_img[cond, 1] = LC[i][3]
        flow_err_img[cond, 2] = LC[i][4]
    flow_err_img[~valid_flow_noc] *= 0.5
    flow_err_img[~valid_flow_occ] = 0

    return flow_err_img


def compute_color(u, v):
    """Compute optical flow color map, return optical flow in color code

    Args:
        u: optical flow horizontal map
        v: optical flow vertical map
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """Generate color wheel according Middlebury color code
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def flow_read_png(fpath):
    """Read KITTI optical flow, returns u,v,valid mask
    """

    R = png.Reader(fpath)
    width, height, data, _ = R.asDirect()
    I = np.array([x for x in data]).reshape((height, width, 3))
    u_ = I[:, :, 0]
    v_ = I[:, :, 1]
    valid = I[:, :, 2]

    u = (u_.astype('float64')-2**15)/64.0
    v = (v_.astype('float64')-2**15)/64.0

    return u, v, valid


class FlowWarp(nn.Module):
    def __init__(self, batch_size, height, width):
        super(FlowWarp, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1).view(batch_size, -1, self.height, self.width)
        self.pix_coords = nn.Parameter(self.pix_coords, requires_grad=False)

    def forward(self, flow):
        pix_coords = self.pix_coords + flow
        pix_coords_norm = pix_coords.permute(0, 2, 3, 1).contiguous()
        pix_coords_norm[..., 0] /= self.width - 1
        pix_coords_norm[..., 1] /= self.height - 1
        pix_coords_norm = (pix_coords_norm - 0.5) * 2

        valid_points = pix_coords_norm.abs().max(dim=-1)[0] <= 1

        return pix_coords, pix_coords_norm, valid_points


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max())
    mi = float(x.min())
    d = ma - mi if ma != mi else 1e-5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def gauss_distance_weight(num_scale, height=128, width=416, sigma1=30, sigma2=120):
    """Create num_scale list gauss weight, it is independent to scale.
    Pixels at the same relative position with respect to each scale image have the same weight.
    W0[i, j, 0] == W1[i//2, j//2, 0]
    """
    rho = 0
    weights = []

    for s in range(num_scale):
        num = 2 ** s
        h, w = height//num, width//num
        distance = np.zeros((h, w))
        x_center, y_center = h//2, w//2
        for i in range(h):
            for j in range(w):
                a = (i-x_center)**2 / (sigma1/num)**2
                b = (j-y_center)**2 / (sigma2/num)**2
                c = 2*rho*(i-x_center)*(j-y_center) / (sigma1*sigma2)
                factor = num**2 / (2*np.pi*sigma1*sigma2*np.sqrt(1-rho**2)) / num**2
                gauss = factor * np.exp(-(a + b - c)/(2*(1-rho**2)))
                distance[i, j] = gauss
        distance = 2e5 * (distance.max() - distance) + 5
        distance = torch.tensor(distance).unsqueeze(0).unsqueeze(0).type(torch.float32)  # (1, 1, H, W)
        weights.append(distance)
    return weights
