# Code based on Monodepth2

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.abspath(os.path.dirname(__file__))  # the directory that options.py resides in
file_parent_dir = os.path.abspath(os.path.dirname(file_dir))


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MobileNet options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_parent_dir, "kitti/raw_data"))
        self.parser.add_argument("--data_root",
                                 type=str,
                                 help="path to the dataset directory",
                                 default=os.path.join(file_parent_dir, "kitti"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="weights and tensorboard log directory",
                                 default="log")  # os.path.join(file_dir, "log")
        self.parser.add_argument("--other_files_path",
                                 type=str,
                                 help="other useful files load directory",
                                 default="files")  # os.path.join(file_dir, "files")

        # TRAINING options
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--use_elu",
                                 help="use elu activation layer",
                                 default=True)
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 default=True)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=128)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=416)
        self.parser.add_argument("--w_p",
                                 type=float,
                                 help="optical flow photometric loss weight",
                                 default=1)
        self.parser.add_argument("--w_e",
                                 type=float,
                                 help="mobile masked epipolar loss weight",
                                 default=1)
        self.parser.add_argument("--w_s",
                                 type=float,
                                 help="flow maps and mobile masks smoothness loss weight",
                                 default=1)
        self.parser.add_argument("--w_c",
                                 type=float,
                                 help="mobile mask consistency loss weight",
                                 default=0.5)
        self.parser.add_argument("--w_d2_sim",
                                 type=float,
                                 help="detectron2 similarity loss weight",
                                 default=0.05)
        self.parser.add_argument("--threshold",
                                 type=float,
                                 help="95 percentile of epipolar map over whole dataset",
                                 default=9.22)
        self.parser.add_argument("--alpha",
                                 type=float,
                                 help="weight for non trivial term in epipolar loss",
                                 default=0.55)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument('--seed',
                                 type=int,
                                 help='seed for random functions and network initialization',
                                 default=42)
        self.parser.add_argument("--clip_grad",
                                 type=float,
                                 help="the max norm of gradient",
                                 default=1)

        # OPTIMIZATION options
        self.parser.add_argument('--fine_tune_flow_motion',
                                 help='fine tuning existing flow, pose and mobile net simultaneously',
                                 action="store_true")
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=4)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument('--momentum',
                                 type=float,
                                 help='momentum for sgd, alpha parameter for adam',
                                 default=0.9)
        self.parser.add_argument('--beta',
                                 type=float,
                                 help='beta parameters for adam',
                                 default=0.999)
        self.parser.add_argument('--weight_decay',
                                 type=float,
                                 help='weight decay',
                                 default=0)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=0.5)

        # ABLATION options
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch for ResNetEncoder",
                                 default="scratch",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--disable_photoloss",
                                 help="if set, doesn't compute photometric loss",
                                 action="store_true")
        self.parser.add_argument("--disable_consisloss",
                                 help="if set, doesn't compute consistency loss",
                                 action="store_true")
        self.parser.add_argument("--disable_min",
                                 help="if set, doesn't pixel-wise min between masks of two frames",
                                 action="store_true")
        self.parser.add_argument("--disable_smoothloss",
                                 help="if set, doesn't compute smooth loss",
                                 action="store_true")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["flownet", "posenet", "mobile_decoder"])
        self.parser.add_argument("--load_adam",
                                 help="load adam status",
                                 action="store_true")
        self.parser.add_argument("--v_load",
                                 type=str,
                                 help="models version to load",
                                 default="v0")
        self.parser.add_argument("--idx_load",
                                 type=int,
                                 help="pretrained models index to load",
                                 default=0)

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=100)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of iterations between each save",
                                 default=1e3)
        self.parser.add_argument("--v_save",
                                 type=str,
                                 help="models version to save",
                                 default="v")

        # EVALUATION options
        self.parser.add_argument("--data_eval_dir",
                                 type=str,
                                 help="Path to kitti stereo dataset",
                                 default=os.path.join(file_parent_dir, "kitti/data_semantics"))
        self.parser.add_argument("--idx_eval",
                                 type=int,
                                 help="The index of model after to evaluate",
                                 default=0)
        self.parser.add_argument("--eval_flow",
                                 help="evaluate flownet",
                                 default=True)
        self.parser.add_argument("--eval_mobile",
                                 help="evaluate mobile mask",
                                 default=True)
        self.parser.add_argument("--eval_odometry",
                                 help="evaluate odometry",
                                 default=True)
        self.parser.add_argument("--odometry_eval_sequences",
                                 help="Number of sequences used for odometry evaluation in KITTI odometry dataset",
                                 default=[0, 3, 4, 5, 7])

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
