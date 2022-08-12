from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.abspath(os.path.dirname(__file__))  # the directory that options.py resides in
file_parent_dir = os.path.abspath(os.path.dirname(file_dir))


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--root",
                                 type=str,
                                 help="path to the directory that options_eval.py resides in",
                                 default=file_dir)
        self.parser.add_argument("--data_root",
                                 type=str,
                                 help="path to the dataset directory",
                                 default=os.path.join(file_parent_dir, "kitti"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="weights and tensorboard log directory",
                                 default=os.path.join(file_dir, "log"))
        self.parser.add_argument("--raw_dataset_dir",
                                 type=str,
                                 help="path to kitti raw dataset",
                                 default=os.path.join(file_parent_dir, "kitti"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--flow_num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=128)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=416)
        self.parser.add_argument("--threshold",
                                 type=float,
                                 help="epipolar 95 percentage threshold",
                                 default=0.8625471)
        self.parser.add_argument("--alpha",
                                 type=float,
                                 help="weight for non trivial term in epipolar loss",
                                 default=0.1)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss and networks",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument('--sequence_length',
                                 type=int,
                                 default=3)

        # COMPONENT WEIGHTS
        self.parser.add_argument("--ssim_loss_weight",
                                 type=float,
                                 help="ssim smoothness weight",
                                 default=0.85)
        self.parser.add_argument("--l1_loss_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.15)
        self.parser.add_argument("--photo_loss_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1)
        self.parser.add_argument("--mvs_loss_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--epipolar_loss_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--disp_smooth_loss_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--flow_smooth_loss_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--flow_smooth_loss_2_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--flow_consistency_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--cross_task_consistency_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--cross_depth_consistency_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--flow_consistency_alpha",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.01)
        self.parser.add_argument("--flow_consistency_beta",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.5)
        self.parser.add_argument("--hard-occu-th",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.)
        self.parser.add_argument("--depth_variance_loss_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--rigid_flow_consistency_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--photo_smooth_loss_weight",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0)
        self.parser.add_argument("--pool-size",
                                 type=int,
                                 help="disparity smoothness weight",
                                 default=0)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=4)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="learning rate",
                                 default=0)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument('--seed',
                                 type=int,
                                 default=0)

        # ABLATION options
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument('--interp',
                                 type=str,
                                 default='bilinear')
        self.parser.add_argument('--padding',
                                 type=str,
                                 default='zeros')
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--dilate",
                                 action="store_true")
        self.parser.add_argument("--align-corners",
                                 action="store_true")
        self.parser.add_argument("--scale-trainable",
                                 action="store_true")
        self.parser.add_argument("--use-depth",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--use-full-resolution",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--direct-flow",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--use-rigid",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--smooth-res",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--mode",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["rigid", "flow", "mix"])
        self.parser.add_argument("--burning-epochs",
                                 type=int,
                                 default=0)
        self.parser.add_argument("--load-depth",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--register-hook",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--start-replay",
                                 type=float,
                                 default=0)
        self.parser.add_argument("--end-replay",
                                 type=float,
                                 default=0)
        self.parser.add_argument("--load-freeze",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--start-percentage",
                                 type=float,
                                 default=0)
        self.parser.add_argument("--end-percentage",
                                 type=float,
                                 default=0)

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="folder where flownet and posenet weights are loaded from",
                                 default=os.path.join(file_dir, "log/v0/models/weights_0"))
        self.parser.add_argument("--eval_name",
                                 type=str,
                                 help="the evaluation name",
                                 default="mobile_masks")
        self.parser.add_argument("--version",
                                 type=str,
                                 help="version of mobilenet to load",
                                 default="v3")
        self.parser.add_argument("--idx",
                                 type=int,
                                 help="num of mobilenet weights to load",
                                 default=14)
        self.parser.add_argument("--pred_errors",
                                 help="predict quantization errors",
                                 action="store_true")

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--save_pred_poses",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--save_pred_motions",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--save_pred_masks",
                                 help="if set saves predicted mobile masks",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str,
                                 default=os.path.join(file_dir, "output/prediction"))
        self.parser.add_argument("--gt_mask_path",
                                 help="folder for generated ground-truth masks",
                                 default=os.path.join(file_dir, "output/mobile_objects_ground_truth"))
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--seqs",
                                 nargs="+",
                                 type=int,
                                 default=[9, 10])
        self.parser.add_argument("--alignment",
                                 help="if set will output the disparities to this folder",
                                 type=str,
                                 default="7dof")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
