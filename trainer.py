# Code based on Monodepth2
from __future__ import absolute_import, division, print_function
import time
from tqdm import tqdm
import json
import os
from os.path import join as join

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# from eval_utils import ValidationSet
from utils import readlines, gauss_distance_weight, flow_to_image, \
    binary_image, normalize_image, sec_to_hm_str, draw_box, get_detectron2_input
from loss_utils import get_epipolar_new, create_coords
from loss_functions import Loss, LossModule
from datasets.kitti_dataset import KITTISegDataset, KITTIRAWDataset
from networks.layers import *
import networks
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2CustomDataset import kitti_decode


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.save_path = os.path.join(self.opt.log_dir, self.opt.v_save)
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.num_scales = len(self.opt.scales)
        self.scale_factor = get_scale_factor(1, self.opt.height, self.opt.width).to(self.device)

        self.writer = {"train": SummaryWriter(os.path.join(self.save_path, "tb_train")),
                       "val": SummaryWriter(os.path.join(self.save_path, "tb_val"))}

        # initialize models and dataset
        self.initialize_models()
        self.initialize_dataset()

        print("{}: training model {}".format(self.device, self.opt.v_save))
        print("Models and tensorboard files save to: {}/{} \n".format(self.opt.log_dir, self.opt.v_save))

        self.save_opts()

    def initialize_dataset(self):
        self.dataset = KITTIRAWDataset
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        self.train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            self.train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=~self.opt.no_cuda, drop_last=True)
        self.total_sample = len(self.train_dataset)
        self.iters = len(self.train_loader)

        # evaluation dataset
        val_dataset = KITTISegDataset(self.opt.data_root, kitti_decode, self.opt.height, self.opt.width)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=~self.opt.no_cuda, drop_last=True)
        self.val_iter = iter(self.val_loader)

        print("\n{:d} training items and {:d} validation items\n".format(self.total_sample, len(val_dataset)))

    def initialize_models(self):
        self.models = {}
        self.load_model(self.opt.learning_rate, load_adam=self.opt.load_adam)
        self.model_lr_scheduler = CosineAnnealingLR(self.model_optimizer, self.opt.scheduler_step_size)

        # self.weights = [w.to(self.device) for w in gauss_distance_weight(self.num_scales)]
        # self.loss = Loss(
        #     self.opt, self.weights, no_ssim=self.opt.no_ssim, alpha=self.opt.alpha).to(self.device)
        # self.epipolar_loss = LossModule(self.opt, self.weights, alpha=self.opt.alpha).to(self.device)

        self.loss = Loss(self.opt, no_ssim=self.opt.no_ssim, alpha=self.opt.alpha).to(self.device)
        self.epipolar_loss = LossModule(self.opt, batch=self.opt.batch_size).to(self.device)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.save_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.save_path, "models", "weights_{}".format(self.idx_save))
        self.idx_save += 1
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if self.opt.fine_tune_flow_motion:
            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                to_save = model.state_dict()
                if model_name == 'encoder':
                    # save the sizes - these are needed at prediction time
                    to_save['height'] = self.opt.height
                    to_save['width'] = self.opt.width
                torch.save(to_save, save_path)
        else:
            save_path = os.path.join(save_folder, "{}.pth".format("mobile_decoder"))
            to_save = self.models["mobile_decoder"].state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self, learning_rate, load_adam=False):
        """Create and load PoseNet/FlowNet/MobileDecoder models
        Load detectron2 model
        """

        # create models
        self.models["posenet"] = networks.PoseNet_v3(
            self.opt.num_layers, self.opt.weights_init == "pretrained").to(self.device)
        self.models["flownet"] = networks.FlowNet_v1(use_elu=self.opt.use_elu).to(self.device)
        self.models["mobile_decoder"] = networks.MobileDecoder(use_elu=self.opt.use_elu).to(self.device)
        # self.models["mobile_decoder"] = networks.MobileSimple(use_elu=self.opt.use_elu).to(self.device)

        # load models
        folder = os.path.join(self.opt.log_dir, "{}", "models", "weights_{}")
        for n in self.opt.models_to_load:
            if n == "mobile_decoder":
                if self.opt.fine_tune_flow_motion or load_adam:
                    path = os.path.join(folder.format(self.opt.v_load, self.opt.idx_load), "{}.pth".format(n))
                else:
                    self.models["mobile_decoder"].init_weights()
                    print("Loading {} model from scratch".format(n))
                    continue
            else:
                path = os.path.join(folder.format("v0", 0), "{}.pth".format(n))
            print("Loading {} model from folder {}".format(n, path))

            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # setup detectron2 model
        setup_logger()
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        os.environ['KITTI_SEG_DATASET'] = os.path.join(self.opt.data_root, "data_semantics")
        self.cfg.DATASETS.TRAIN = ("kitti_seg_instance_train",)
        self.cfg.INPUT.MASK_FORMAT = "bitmask"
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
        self.cfg.MODEL.WEIGHTS = os.path.join(self.opt.log_dir, "model_final_detectron2.pth")
        self.cfg.INPUT.MIN_SIZE_TEST = 1024
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
        self.instance_model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(self.instance_model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.instance_model.to(self.device).eval()

        # trainable parameters set
        self.parameters_to_train = []
        self.parameters_to_train += list(self.models["mobile_decoder"].parameters())
        if self.opt.fine_tune_flow_motion:
            self.parameters_to_train += list(self.models["posenet"].parameters())
            self.parameters_to_train += list(self.models["flownet"].parameters())

        # set optimizer
        self.model_optimizer = optim.Adam(self.parameters_to_train, learning_rate)

        # loading adam state
        if load_adam:
            optimizer_load_path = os.path.join(folder.format(self.opt.v_load, self.opt.idx_load), "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights...")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.idx_save = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=self.opt.clip_grad)
            self.model_optimizer.step()
            self.model_lr_scheduler.step(self.epoch + batch_idx / self.iters)

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0
            print_log = early_phase or late_phase
            if print_log:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
            if batch_idx % 50 == 0:
                self.log(inputs, outputs, losses, log_image=print_log)
                self.val()

            self.step += 1
            if self.step % self.opt.save_frequency == 0:
                self.save_model()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate losses and warped images
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        frames_ids_new = self.opt.frame_ids.copy()[1:]
        tgt_img = inputs[("color", 0, 0)]
        outputs, flows, mobiles, cam_T_cams = {}, {}, {}, {}

        for i in frames_ids_new:
            ref_img = inputs[("color", i, 0)]

            flow, features_enc = self.models["flownet"](tgt_img, ref_img, frame_id=i)
            axis_angle, translation = self.models["posenet"](tgt_img, ref_img)
            frame_mobiles = self.models["mobile_decoder"](features_enc, axis_angle, translation, frame_id=i)
            cam_T_cam = transformation_from_parameters(axis_angle, translation)

            flows.update(flow)
            mobiles.update(frame_mobiles)
            cam_T_cams[i] = cam_T_cam

        detectron2_input = get_detectron2_input(inputs["instance_img"], self.cfg)
        detectron2_out = self.instance_model(detectron2_input)
        output, losses = self.loss(
            inputs, self.opt.frame_ids[1:], flows, mobiles, detectron2_out, self.opt.scales, cam_T_cams)

        outputs.update(mobiles)
        outputs.update(output)
        outputs["seg_info"] = detectron2_out

        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            tgt = inputs[('color', 0)].to(self.device)
            next_tgt = inputs[('color', 1)].to(self.device)

            # prediction
            flow, features_enc = self.models["flownet"](tgt, next_tgt)
            axis_angle, translation = self.models["posenet"](tgt, next_tgt)
            mobile = self.models["mobile_decoder"](features_enc, axis_angle, translation)
            detectron2_input = get_detectron2_input(inputs["instance_img"], self.cfg)
            detectron2_out = self.instance_model(detectron2_input)

            cam_T_cam = transformation_from_parameters(axis_angle, translation)
            flow_map = self.scale_factor * flow[('flow', 0, 0)]
            epip_loss, epip_map, epip_ori = self.epipolar_loss.epipolar_loss(
                flow_map, mobile[('mobile', 0, 0)], detectron2_out,
                inputs['inv_K'].to(self.device), cam_T_cam[:, :3, :3], cam_T_cam[:, :3, -1])

            self.writer["val"].add_scalar("{}".format("epipolar loss"), epip_loss, self.step)
            for j in range(self.opt.batch_size):
                self.writer["val"].add_image("{}/target".format(j), normalize_image(tgt[j].data), self.step)
                self.writer["val"].add_image("{}/epip".format(j), epip_map[j] / epip_map[j].max(), self.step)
                self.writer["val"].add_image("{}/epip_ori".format(j), epip_ori[j] / epip_ori[j].max(), self.step)
                self.writer["val"].add_image("{}/mobile".format(j), mobile[('mobile', 0, 0)][j], self.step)
                self.writer["val"].add_image("{}/mobile_bi".format(j),
                                             binary_image(mobile[('mobile', 0, 0)][j], 0.4),
                                             self.step)
                self.writer["val"].add_image("{}/instances".format(j),
                                             draw_box(inputs["instance_img"][j].permute(2, 0, 1).type(torch.uint8),
                                                      detectron2_out[j]["instances"]),
                                             self.step)
            del inputs

        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / (self.step + 1) - 1.0) * time_sofar
        print_string = "epoch {} | batch {:>6} | loss: {:.5f} | examples/s: {:5.1f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, loss, samples_per_sec,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, inputs, outputs, losses, num=4, log_image=False):
        """Write an event to the tensorboard events file
        :param log_image: write images outputs into writer
        :param num: number of samples write into event every time
        """
        frame_ids = self.opt.frame_ids[1:]
        s = 0
        b = inputs[("color", 0, s)].size()[0]
        num = min(num, b)

        # write scalar terms
        for l, v in losses.items():
            self.writer["train"].add_scalar("{}".format(l), v, self.step)
        if log_image:
            for j in range(num):
                # add image terms concatenate frames together to compare
                # warp, diff, valid = [], [], []
                epip, flow_maps, mobile, epip_ori = [], [], [], []
                tgt = normalize_image(inputs[("color", 0, s)][j].data)
                min_mobile = outputs["min_mobiles"][s][j]

                for i in frame_ids:
                    # 11 is the maximum value for gauss distance weighted epipolar map over whole training dataset
                    epip.append(outputs["epipolars"][(i, s)][j] / outputs["epipolars"][(i, s)][j].max())
                    epip_ori.append(outputs["epipolar_ori"][(i, s)][j] / outputs["epipolar_ori"][(i, s)][j].max())
                    f = outputs["flows"][(i, s)][j].detach().cpu().numpy().transpose(1, 2, 0)
                    flow_maps.append(flow_to_image(f))
                    mobile.append(outputs[("mobile", i, s)][j])

                self.writer["train"].add_image("{}/target".format(j), tgt, self.step)
                self.writer["train"].add_image("{}/epip".format(j), torch.hstack(epip), self.step)
                self.writer["train"].add_image("{}/epip_ori".format(j), torch.hstack(epip_ori), self.step)
                self.writer["train"].add_image("{}/mobile".format(j), torch.hstack(mobile), self.step)
                self.writer["train"].add_image("{}/mobile_min".format(j), min_mobile, self.step)
                self.writer["train"].add_image("{}/mobile_min_bi".format(j), binary_image(min_mobile, 0.4), self.step)
                self.writer["train"].add_image("{}/flow".format(j), np.vstack(flow_maps), self.step, dataformats='HWC')
                self.writer["train"].add_image("{}/instances".format(j),
                                               draw_box(inputs["instance_img"][j].permute(2, 0, 1).type(torch.uint8),
                                                        outputs["seg_info"][j]["instances"]),
                                               self.step)

    def hyperparameter_try(self, name):
        """ Play with hyperparameters
        """
        param_grid = {
            # "alpha": [0.53, 0.56, 0.58, 0.6, 0.62],   # saved in hyper_alpha_v2_first
            # "alpha": [0.53, 0.54, 0.55, 0.56, 0.57, 0.58],  # saved in hyper_alpha_v2_second (0.56)
            # "alpha": [10, 8, 5, 1, 0.8, 0.5],  # saved in hyper_alpha_v8
            # "alpha": [0.8, 0.6, 0.5, 0.3, 0.2, 0.1],  # saved in hyper_alpha_mix_v0
            # "threshold": [0.4294, 0.5123, 0.5799, 0.6368, 0.7077],  # saved in hyper_threshold_gauss  (0.5123)
            # "threshold": [4.571, 5.4798, 6.2176, 6.8344, 7.5927, 9.2186, 12.7505, 15.7806],  # saved in hyper_threshold
            # "batch_size": [1, 4, 8, 16],  # saved in hyper_batch_size (8)
            # "learning_rate": [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  # saved in hyper_learning_rate_derivable_consis (1e-3)
            # "w_c": [0.5, 0.3, 0.1, 0.08, 0.06, 0.04, 0.01]  # saved in hyper_weight_consistency
            "w_d2_sim": np.linspace(0.01, 0.3, 10)[:7]  # saved in hyper_weight_d2_sim
        }

        print("The hyperparameter {} : ".format(name), param_grid[name])
        for turn, hyper_value in enumerate(param_grid[name]):

            # # weight of consistency loss
            # self.opt.w_c = hyper_value
            # print("\nEpoch {} | ConsisWeight={:.2f}: ".format(turn, hyper_value))

            # # learning rate
            # self.initialize_models(hyper_value)
            # print("\nEpoch {} | LearningRate={}: ".format(turn, hyper_value))

            # # alpha
            # self.opt.alpha = hyper_value
            # print("\nEpoch {} | alpha={:.2f}: ".format(turn, hyper_value))
            # self.initialize_models()

            # self.opt.threshold = hyper_value
            # print("\nEpoch {} | threshold={:.4f}: ".format(turn, hyper_value))
            # self.initialize_models()

            # weight of detectron2 similarity loss
            self.opt.w_d2_sim = hyper_value
            print("\nEpoch {} | D2SimWeight={:.4f}: ".format(turn, hyper_value))

            self.set_train()
            for batch_idx, inputs in enumerate(self.train_loader):

                outputs, losses = self.process_batch(inputs)

                self.model_optimizer.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=self.opt.clip_grad)
                self.model_optimizer.step()
                self.model_lr_scheduler.step(batch_idx / len(self.train_loader))

                # log less frequently after the first 2000 steps to save time & disk space
                early_phase = batch_idx % 100 == 0 and batch_idx < 1000
                late_phase = batch_idx % 500 == 0 and batch_idx >= 1000
                image = early_phase or late_phase
                self.log_hyper("train", hyper_value, batch_idx, image, inputs, outputs, losses)
                self.log_hyper("val", hyper_value, batch_idx, image)

    def log_hyper(self, mode, hyper_value, step, image=False, inputs=None, outputs=None, losses=None):
        """Write an event to the tensorboard events file
        """
        scale, frame, item = 0, 1, 0
        tgt, flow, mobile, min_mobile, epip, seg_img = 0, 0, 0, 0, 0, 0

        if mode == "train":
            if step % 50 == 0:
                for name, value in losses.items():
                    self.writer["train"].add_scalar("{}/{}".format(hyper_value, name), value, step)
            if image:
                line = "EpipLoss : {:.4f} | ConsisLoss : {:.4f} | SmoothLoss : {:.4f} \n"
                print(line.format(
                    self.opt.w_e * losses["epip"],
                    self.opt.w_c * losses["consis"],
                    self.opt.w_s * losses["smooth"]))

                tgt = normalize_image(inputs[("color", 0, scale)][item].data)
                epip = outputs["epipolars"][(frame, scale)][item] / outputs["epipolars"][(frame, scale)][item].max()
                f = outputs["flows"][(frame, scale)][item].detach().cpu().numpy().transpose(1, 2, 0)
                flow = flow_to_image(f)
                mobile = outputs[("mobile", frame, scale)][item]
                min_mobile = outputs["min_mobiles"][scale][item]
                seg_img = draw_box(inputs["instance_img"][item].permute(2, 0, 1).type(torch.uint8),
                                   outputs["seg_info"][item]["instances"])

                self.writer[mode].add_image("{}/minimal_mask".format(hyper_value), min_mobile, step)
                self.writer[mode].add_image("{}/minimal_bi_mask".format(hyper_value),
                                            binary_image(min_mobile, 0.4), step)

        elif mode == "val":
            """Validate the model on a single minibatch
            """
            self.set_eval()
            try:
                inputs = self.val_iter.next()
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                inputs = self.val_iter.next()

            img = inputs[('color', 0)].to(self.device)
            next_img = inputs[('color', 1)].to(self.device)

            with torch.no_grad():
                flows, features_enc = self.models["flownet"](img, next_img)
                axis_angle, translation = self.models["posenet"](img, next_img)
                mobiles = self.models["mobile_decoder"](features_enc, axis_angle, translation)
                cam_T_cam = transformation_from_parameters(axis_angle, translation)
                flow_map = self.scale_factor * flows[('flow', 0, 0)]

                detectron2_input = get_detectron2_input(inputs["instance_img"], self.cfg)
                detectron2_out = self.instance_model(detectron2_input)[item]["instances"]
                epip_loss, epip_map, epip_ori = self.epipolar_loss.epipolar_loss(
                    flow_map, mobiles[('mobile', 0, 0)], detectron2_out,
                    inputs['inv_K'].to(self.device), cam_T_cam[:, :3, :3], cam_T_cam[:, :3, -1])

                tgt = normalize_image(inputs[('color', 0)][item].data)
                epip = epip_map[item] / epip_map[item].max()
                flow = flow_to_image(flow_map[item].detach().cpu().numpy().transpose(1, 2, 0))
                mobile = mobiles[('mobile', 0, 0)][item]
                seg_img = draw_box(inputs["instance_img"][item].permute(2, 0, 1).type(torch.uint8), detectron2_out)
                epip_ori = epip_ori[item] / epip_ori[item].max()

                if step % 50 == 0:
                    self.writer[mode].add_scalar("{}/epipolar".format(hyper_value), epip_loss, step)
                if image:
                    self.writer[mode].add_image("{}/binary_mask".format(hyper_value), binary_image(mobile, 0.4), step)
                    self.writer[mode].add_image("{}/ori_epipolar".format(hyper_value), epip_ori, step)
                del inputs
            self.set_train()

        if image:
            self.writer[mode].add_image("{}/epipolar".format(hyper_value), epip, step)
            self.writer[mode].add_image("{}/target".format(hyper_value), tgt, step)
            self.writer[mode].add_image("{}/flow".format(hyper_value), flow, step, dataformats='HWC')
            self.writer[mode].add_image("{}/mobile".format(hyper_value), mobile, step)
            self.writer[mode].add_image("{}/instances".format(hyper_value), seg_img, step)

    def epipolar_statics(self):
        self.set_train()

        num_quantile = 1000
        frames_ids_new = self.opt.frame_ids.copy()[1:]
        b, h, w = self.opt.batch_size, self.opt.height, self.opt.width

        scale_factor = get_scale_factor(b, h, w).to(self.device)

        ones = torch.ones((b, 1, h, w))  # (B, 1, H, W)
        pix_coords = create_coords(b, h, w)
        p1 = torch.cat([pix_coords, ones], 1).view(b, 3, -1)  # (B, 3, H*W)
        percentiles = torch.zeros((2, num_quantile, self.total_sample))
        percentage = torch.linspace(0, 1, num_quantile)

        # for batch_idx, inputs in enumerate(tqdm(self.train_dataset, total=self.total_sample)):
        for batch_idx, inputs in enumerate(tqdm(self.train_loader)):
            tgt_img = inputs[("color", 0, 0)].to(self.device)
            inv_K = inputs[("inv_K", 0)][:, :3, :3]

            for i in frames_ids_new:
                ref_img = inputs[("color", i, 0)].to(self.device)

                flow, features_enc = self.models["flownet"](tgt_img, ref_img, frame_id=i)
                axis_angle, translation = self.models["posenet"](tgt_img, ref_img)
                cam_T_cam = transformation_from_parameters(axis_angle, translation).detach().cpu()

                idx = 0 if i == -1 else 1
                flow_map = (scale_factor * flow[('flow', i, 0)]).detach().cpu()

                p2 = torch.cat([pix_coords + flow_map, ones], 1).view(b, 3, -1)  # (B, 3, H*W)
                epipolar = get_epipolar_new(p1, p2, inv_K, cam_T_cam).view(b, 1, h, w).abs()  # (B, H, W)
                # epipolar_weighted = epipolar / self.weight
                percentiles[idx, :, b * batch_idx:b * (batch_idx + 1)] = torch.quantile(
                    epipolar.view(b, -1), percentage, dim=1)  # (1000, B)

        percentiles = percentiles.numpy()
        np.save(join(self.opt.other_files_path, "eigen_zhou_percentiles.npy"), percentiles)
        thresholds = np.percentile(percentiles.reshape(-1), [80, 85, 88, 90, 92, 95, 98, 99])
        np.savetxt(join(self.opt.other_files_path, "eigen_zhou_thresholds"), thresholds)

        return thresholds
