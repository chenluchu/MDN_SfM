from imageio import imread
from path import Path
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.utils.data as data

from utils import load_as_float, flow_read_png
from networks import FlowNet_v1, PoseNet_v3, MobileDecoder
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ValidationSet(data.Dataset):
    """ Parent dataset for all validation dataset

    Images from Kitti 2015 scene flow --> 200 test image pairs
    """

    def __init__(self, root, n=200, phase='training', occ='flow_occ'):
        super(ValidationSet, self).__init__()
        self.root = Path(root)
        self.files = n
        self.phase = phase
        self.occ = occ

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.files


class ValidationFlow(ValidationSet):
    """ Dataset for pure FlowNet evaluation
    """

    def __init__(self, *args, **kwargs):
        super(ValidationFlow, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """ returns are all numpy array

        Returns:
            'tgt', 'next_tgt', 'gt_flow_occ', 'gt_flow_noc':  -- original (H, W, C)
            'intrinsics': -- (3, 3)
            'translation': tranlation between two camera in car -- (3, 1)
            'gt_transformation': transformation matrix from frame_10 to frame_11 -- (3, 4)
        """
        if index >= len(self): raise IndexError
        tgt_img_path = self.root.joinpath('data_scene_flow',
                                          self.phase,
                                          'image_2',
                                          str(index).zfill(6) + '_10.png')
        next_tgt_img_path = self.root.joinpath('data_scene_flow',
                                               self.phase,
                                               'image_2',
                                               str(index).zfill(6) + '_11.png')
        gt_flow_occ_path = self.root.joinpath('data_scene_flow',
                                              self.phase,
                                              'flow_occ',
                                              str(index).zfill(6) + '_10.png')
        gt_flow_noc_path = self.root.joinpath('data_scene_flow',
                                              self.phase,
                                              'flow_noc',
                                              str(index).zfill(6) + '_10.png')
        cam_calib_path = self.root.joinpath('data_scene_flow_calib',
                                            self.phase,
                                            'calib_cam_to_cam',
                                            str(index).zfill(6) + '.txt')

        tgt_img = load_as_float(tgt_img_path)
        next_tgt_img = load_as_float(next_tgt_img_path)
        u, v, valid = flow_read_png(gt_flow_occ_path)
        gt_flow_occ = np.dstack((u, v, valid))
        u, v, valid = flow_read_png(gt_flow_noc_path)
        gt_flow_noc = np.dstack((u, v, valid))
        intrinsics = get_intrinsics(cam_calib_path).astype('float32')[:, :3]
        translation = get_intrinsics(cam_calib_path, cam_id=3).astype('float32')[:, 3][:, np.newaxis]
        gt_transformation = np.hstack([np.eye(3), translation]).astype(np.float32)

        return {'tgt': tgt_img,
                'next_tgt': next_tgt_img,
                'gt_flow_occ': gt_flow_occ,
                'gt_flow_noc': gt_flow_noc,
                'intrinsics': intrinsics,
                'translation': translation,
                'gt_transformation': gt_transformation}


class GroundTruth(ValidationSet):
    """ Dataset to generate ground truth images by geometry epipolar
    """

    def __init__(self, *args, **kwargs):
        super(GroundTruth, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """ returns are all torch tensor
        Returns:
            'tgt'  -- (3, H, W)
            'full_flow' -- (2, H, W)
            'intrinsics' -- (3, 3)
            'translation' -- (3, 4)
        """
        if index >= len(self): raise IndexError
        tgt_img_path = self.root.joinpath('data_scene_flow',
                                          self.phase,
                                          'image_2',
                                          str(index).zfill(6) + '_10.png')
        gt_flow_occ_path = self.root.joinpath('data_scene_flow',
                                              self.phase,
                                              'flow_occ',
                                              str(index).zfill(6) + '_10.png')
        cam_calib_path = self.root.joinpath('data_scene_flow_calib',
                                            self.phase,
                                            'calib_cam_to_cam',
                                            str(index).zfill(6) + '.txt')

        tgt_img = torch.from_numpy(load_as_float(tgt_img_path).transpose(2, 0, 1))
        _, h, w = tgt_img.size()
        u, v, valid = flow_read_png(gt_flow_occ_path)
        sf = torch.FloatTensor([w, h]).repeat(h, w, 1)
        full_flow = (torch.from_numpy(np.dstack((u, v))) * sf).permute(2, 0, 1)
        intrinsics = torch.from_numpy(get_intrinsics(cam_calib_path).astype('float32')[:, :3])
        translation = torch.from_numpy(get_intrinsics(cam_calib_path, cam_id=3).astype('float32')[:, 3][:, np.newaxis])
        cam_T_cam = torch.cat([torch.eye(3), translation], dim=1)

        return {'tgt': tgt_img,
                'full_flow': full_flow,
                'intrinsics': intrinsics,
                'cam_T_cam': cam_T_cam}


class ValidationMobileMask(ValidationSet):
    """
    Returns:
        'tgt', 'next_tgt'  -- original (H, W, C)
        'intrinsics' -- (3, 3)
    """

    def __init__(self, *args, **kwargs):
        super(ValidationMobileMask, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        tgt_img_path = self.root.joinpath('data_scene_flow',
                                          self.phase,
                                          'image_2',
                                          str(index).zfill(6) + '_10.png')
        next_tgt_img_path = self.root.joinpath('data_scene_flow',
                                               self.phase,
                                               'image_2',
                                               str(index).zfill(6) + '_11.png')

        tgt_img = load_as_float(tgt_img_path)
        next_tgt_img = load_as_float(next_tgt_img_path)

        return tgt_img, next_tgt_img


class ValidationMobileMaskMore(ValidationSet):
    """ Kitti 2015 Eigen full val_files loader

        Images are chosen from eigen_zhou validation dataset
        @:param self.files: contains 200 file paths to eigen_zhou validation dataset
    """

    def __init__(self, *args, **kwargs):
        super(ValidationMobileMaskMore, self).__init__(*args, **kwargs)
        self.filenames = np.array(self.files).astype(np.string_)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        folder, num, side = str(self.filenames[index], encoding='utf-8').split()
        num = int(num)
        side = self.side_map[side]
        tgt_img_path = self.root.joinpath(self.phase,
                                          folder,
                                          'image_0{}/data'.format(side),
                                          "{:010d}{}".format(num, '.png'))
        next_tgt_img_path = self.root.joinpath(self.phase,
                                               folder,
                                               'image_0{}/data'.format(side),
                                               "{:010d}{}".format(num + 1, '.png'))
        cam_calib_path = self.root.joinpath(self.phase,
                                            folder.split('/')[0],
                                            'calib_cam_to_cam' + '.txt')

        tgt_img = load_as_float(tgt_img_path)
        next_tgt_img = load_as_float(next_tgt_img_path)
        intrinsics = get_intrinsics(cam_calib_path, side).astype('float32')[:, :3]

        return {'tgt': tgt_img,
                'next_tgt': next_tgt_img,
                'intrinsics': intrinsics}

    def __len__(self):
        return len(self.files)


def load_models(opt, device, motion=True, pose=True, mobile=True, detectron2=False):
    """ Models load function for evaluation
    """
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)
    print("-> Loading weights from \n{}".format(opt.load_weights_folder))

    models = []
    if motion:
        n_ch = 6 if opt.use_rigid else 0
        motion_net = FlowNet_v1(opt.flow_num_layers,
                                pretrained=False,
                                scale_trainable=opt.scale_trainable,
                                n_ch=n_ch
                                ).to(device)
        model_dict = motion_net.state_dict()
        net_path = os.path.join(opt.load_weights_folder, "flownet.pth")
        net_dict = torch.load(net_path)
        motion_net.load_state_dict({k: v for k, v in net_dict.items() if k in model_dict})
        motion_net.eval()
        models.append(motion_net)

    if pose:
        pose_net = PoseNet_v3(opt.num_layers,
                              pretrained=False,
                              ).to(device)
        model_dict = pose_net.state_dict()
        net_path = os.path.join(opt.load_weights_folder, "posenet.pth")
        net_dict = torch.load(net_path)
        pose_net.load_state_dict({k: v for k, v in net_dict.items() if k in model_dict})
        pose_net.eval()
        models.append(pose_net)

    if mobile:
        mobile_weights_folder = os.path.join(os.getcwd(),
                                             "log/{}/models/weights_{}".format(opt.version, opt.idx))
        mobile_net = MobileDecoder().to(device)
        model_dict = mobile_net.state_dict()
        net_path = os.path.join(mobile_weights_folder, "mobile_decoder.pth")
        net_dict = torch.load(net_path)
        mobile_net.load_state_dict({k: v for k, v in net_dict.items() if k in model_dict})
        mobile_net.eval()
        models.append(mobile_net)
        print(mobile_weights_folder)

    if detectron2:
        # setup detectron2 model
        setup_logger()
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        os.environ['KITTI_SEG_DATASET'] = os.path.join(opt.data_root, "data_semantics")
        cfg.DATASETS.TRAIN = ("kitti_seg_instance_train",)
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
        cfg.MODEL.WEIGHTS = os.path.join(opt.log_dir, "model_final_detectron2.pth")
        cfg.INPUT.MIN_SIZE_TEST = 1024
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
        instance_model = build_model(cfg)
        checkpointer = DetectionCheckpointer(instance_model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        instance_model.to(device).eval()
        models.append(instance_model)
        models.append(cfg)

    return models


def get_quantitative_results(pred_mask, gt_mask):
    """" Input single numpy array masks
    """
    tp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
    tn = np.sum(np.logical_and(pred_mask == 0, gt_mask == 0))
    fp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 0))
    fn = np.sum(np.logical_and(pred_mask == 0, gt_mask == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2*precision*recall / (precision + recall)
    dice = 2*tp / (2*tp + fn + fp)

    return accuracy, precision, recall, f1_score, dice


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def get_intrinsics(calib_file, cam_id=2):
    cam_id = '{:02d}'.format(cam_id)
    file_data = read_calib_file(calib_file)
    P_rect = np.reshape(file_data['P_rect_' + cam_id], (3, 4))
    return P_rect


def binary_image(x, threshold=0.5):
    bi_x = np.zeros_like(x)
    bi_x[x >= threshold] = 1
    return bi_x


class test_framework_KITTI(object):
    def __init__(self, root, sequence_set, seq_length=3, step=1):
        self.root = root
        self.img_files, self.poses, self.sample_indices = read_scene_data(self.root, sequence_set, seq_length, step)

    def generator(self):
        """
        Args:
            'imgs': list of seq_length images
            'poses': normalized transform matrix -- (seq_length, 3, 4)
        """
        for img_list, pose_list, sample_list in zip(self.img_files, self.poses, self.sample_indices):
            for snippet_indices in sample_list:
                imgs = [imread(img_list[i]).astype(np.float32) for i in snippet_indices]  # seq_length

                poses = np.stack([pose_list[i] for i in snippet_indices])
                first_pose = poses[0]
                poses[:, :, -1] -= first_pose[:, -1]
                compensated_poses = np.linalg.inv(first_pose[:, :3]) @ poses

                yield {'imgs': imgs,
                       'path': img_list[0],
                       'poses': compensated_poses
                       }

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return sum(len(imgs) for imgs in self.img_files)


def read_scene_data(data_root, sequence_set, seq_length=3, step=1):
    """
    Returns:
        im_sequences: list of image paths -- [0-n]
        poses_sequences: pose transformation matrix -- (n, 3, 4)
        indices_sequences: indices matrix -- sequence_set * (n-seq_length+1, seq_length)
    """
    data_root = Path(data_root)  # "kitti/odometry_data"
    im_sequences = []
    poses_sequences = []
    indices_sequences = []
    demi_length = (seq_length - 1) // 2
    shift_range = np.array([step * i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)
    # [-2, -1, 0, 1, 2]

    sequences = set()
    for seq in sequence_set:
        corresponding_dirs = set(data_root.dirs(seq))
        sequences = sequences | corresponding_dirs

    print('getting test metadata for theses sequences : {}'.format(sequences))
    for sequence in tqdm(sequences):
        poses = np.genfromtxt(data_root / 'poses' / '{}.txt'.format(sequence.name)).astype(np.float64).reshape(-1, 3, 4)
        imgs = sorted(Path(sequence / 'image_2').files('*.png'))
        tgt_indices = np.arange(demi_length, len(imgs) - demi_length).reshape(-1, 1)
        snippet_indices = shift_range + tgt_indices
        im_sequences.append(imgs)
        poses_sequences.append(poses)
        indices_sequences.append(snippet_indices)
    return im_sequences, poses_sequences, indices_sequences


def get_output_img_name(files, j):
    line = files[j].split()
    return line[0].split("/")[1] + "_" + line[1]


def compute_epe(gt, pred, mask):
    u_gt, v_gt = gt[:, :, 0], gt[:, :, 1]
    u_pred, v_pred = pred[:, :, 0], pred[:, :, 1]

    epe = np.sqrt((u_gt - u_pred) ** 2 + (v_gt - v_pred) ** 2)
    error = np.sum(epe * mask) / np.sum(mask)
    return error


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1])/np.sum(pred[:, :, -1] ** 2)  # using median scaling
    ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1]-R[1, 0],
                            R[1, 2]-R[2, 1],
                            R[0, 2]-R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE/snippet_length, RE/snippet_length


def normalize(inputs, mean, std):
    for t, m, s in zip(inputs, mean, std):
        t.sub_(m).div_(s)


def array_to_tensor(image):
    return torch.from_numpy(image.transpose(2, 0, 1)).float() / 255


def checkNextFrame(files, opt, choose=400):
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    root = Path(opt.raw_dataset_dir)
    not_exist = 0
    total = len(files)

    for i, f in enumerate(files):
        folder, num, side = f.split()
        num = int(num)
        side = side_map[side]
        next_tgt_img_path = root.joinpath('raw_data',
                                          folder,
                                          'image_0{}/data'.format(side),
                                          "{:010d}{}".format(num + 1, '.png'))

        if not os.path.exists(next_tgt_img_path):
            not_exist += 1
            del files[i]

    newFiles = random.sample(files, choose)
    line = "There are {} files in training dataset, delete {} items, {} items left; now we choose {} items to evaluate."
    print(line.format(total, not_exist, len(files), len(newFiles)))
    return newFiles
