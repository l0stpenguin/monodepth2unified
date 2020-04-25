# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json
from tqdm import tqdm

from utils import *
from kitti_utils import *
from layers import *

import datasets
import models


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.use_multi_gpu = torch.cuda.device_count() > 1

        self.scales = range(self.opt.num_scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["depth"] = models.DepthModel(
            num_layers=self.opt.num_layers, 
            num_scales=self.opt.num_scales,
            pretrained=(self.opt.weights_init=="pretrained"))
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose"] = models.PoseModel(
                    num_layers=self.opt.num_layers, 
                    pretrained=(self.opt.weights_init=="pretrained"))
            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = models.PoseCNN()

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.use_multi_gpu:
            self.models["depth"] = nn.DataParallel(self.models["depth"])
            self.models["pose"] = nn.DataParallel(self.models["pose"])

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

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
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            if self.step % self.opt.print_frequency == 0:
                self.log_time(batch_idx, duration, losses)

            if self.step % self.opt.log_frequency == 0:
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses, self.step)

            self.step += 1

        self.val()
        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        for f_i in self.opt.frame_ids:
            outputs.update(self.models["depth"](inputs["color_aug", f_i, 0], f_i))

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        # select what features the pose network takes as input
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = torch.cat(pose_inputs, 1)

                rotation, translation = self.models["pose"](pose_inputs)
                outputs[("rotation", f_i, 0)] = rotation
                outputs[("translation", f_i, 0)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", f_i, 0)] = transformation_from_parameters(
                    rotation, translation, 
                    rotation_mode=self.opt.rotation_mode,
                    invert=(f_i<0))

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        def merge_dict(dict1, dict2):
            for k, v in dict1.items():
                if k in dict2.keys():
                    dict2[k] += v
                else:
                    dict2[k] = v

        self.set_eval()
        val_losses = {}
        # try:
        #     inputs = self.val_iter.next()
        # except StopIteration:
        #     self.val_iter = iter(self.val_loader)
        #     inputs = self.val_iter.next()

        for batch_idx, inputs in tqdm(enumerate(self.val_loader), 
            desc='Val Epoch {}'.format(self.epoch), ncols=80, leave=True, total=len(self.val_loader)):
            with torch.no_grad():
                outputs, losses = self.process_batch(inputs)
                if batch_idx == 0:
                    val_inputs = inputs
                    val_outputs = outputs

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                merge_dict(losses, val_losses)
                del inputs, outputs, losses
        
        self.log("val", val_inputs, val_outputs, val_losses, self.epoch)
        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:

            for i, frame_id in enumerate(self.opt.frame_ids):

                disp = outputs[("disp", frame_id, scale)]
                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    source_scale = 0

                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

                outputs[("depth", frame_id, scale)] = depth

                if i == 0: continue

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", frame_id, 0)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    rotation = outputs[("rotation", frame_id, 0)]
                    translation = outputs[("translation", frame_id, 0)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        rotation, translation * mean_inv_depth[:, 0], 
                        rotation_mode=self.opt.rotation_mode,
                        invert=(frame_id<0))

                cam_points = self.backproject_depth[source_scale](
                    outputs[("depth", 0, scale)], inputs[("inv_K", source_scale)])
                pix_coords, computed_depth = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample_coords", frame_id, scale)] = pix_coords
                outputs[("computed_depth", frame_id, scale)] = computed_depth

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample_coords", frame_id, scale)],
                    padding_mode="border")#, align_corners=True)
                outputs[("projected_depth", frame_id, scale)] = F.grid_sample(
                    outputs[("depth", frame_id, source_scale)],
                    outputs[("sample_coords", frame_id, scale)],
                    padding_mode="border")#, align_corners=True)

                outputs[("valid_mask", frame_id, scale)] = (pix_coords.abs().max(dim=-1)[0] <= 1).float()
                if self.opt.occlusion_mask:
                    outputs[("valid_mask", frame_id, scale)] = outputs[("valid_mask", frame_id, scale)] * \
                        (computed_depth > 0).float().squeeze(1) * \
                        (outputs[("computed_depth", frame_id, scale)] <= outputs[("projected_depth", frame_id, scale)]).float().squeeze(1)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_geometry_loss(self, computed_depth, projected_depth):
        return (computed_depth - projected_depth).abs() / (computed_depth + projected_depth).abs()

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        total_reprojection_loss = 0
        total_geometry_loss = 0
        total_smooth_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []
            geometry_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            target = inputs[("color", 0, source_scale)]
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target) * outputs[("valid_mask", frame_id, scale)].unsqueeze(1))

                computed_depth = outputs[("computed_depth", frame_id, scale)]
                projected_depth = outputs[("projected_depth", frame_id, scale)]
                geometry_losses.append(
                    self.compute_geometry_loss(computed_depth, projected_depth) * outputs[("valid_mask", frame_id, scale)].unsqueeze(1))

            reprojection_losses = torch.cat(reprojection_losses, 1)
            geometry_losses = torch.cat(geometry_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target) * outputs[("valid_mask", frame_id, scale)].unsqueeze(1))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                geometry_loss = geometry_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
                geometry_loss, _ = torch.min(geometry_losses, dim=1, keepdim=True)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                # print(combined.size())
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs[("auto_mask", scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()
                geometry_loss = geometry_loss * outputs[("auto_mask", scale)].unsqueeze(1)

            reprojection_loss_to_optimize = to_optimise.mean()
            geometry_loss_to_optimize = geometry_loss.mean()

            smooth_loss = 0
            for frame_id in self.opt.frame_ids:
                disp = outputs[("disp", frame_id, scale)]
                color = inputs[("color", frame_id, scale)]
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss += get_smooth_loss(norm_disp, color)

            loss += reprojection_loss_to_optimize / (2 ** scale)
            loss += self.opt.geometry_consistency * geometry_loss_to_optimize / (2 ** scale)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            total_reprojection_loss += reprojection_loss_to_optimize / (2 ** scale)
            total_geometry_loss += geometry_loss_to_optimize / (2 ** scale)
            total_smooth_loss += smooth_loss / (2 ** scale)
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.opt.num_scales
        total_reprojection_loss /= self.opt.num_scales
        total_geometry_loss /= self.opt.num_scales
        total_smooth_loss /= self.opt.num_scales
        losses["loss"] = total_loss
        losses["reprojection_loss"] = total_reprojection_loss
        losses["geometry_loss"] = total_geometry_loss
        losses["smooth_loss"] = total_smooth_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80) #
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.4f}, p: {:.4f}, g: {:.4f}, s: {:.4f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, 
            losses["loss"].cpu().data, losses["reprojection_loss"].cpu().data, losses["geometry_loss"].cpu().data,
            losses["smooth_loss"].cpu().data, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses, log_step):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}_{}".format(mode, l), v, log_step)

        for j in range(min(2, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "{}_color/f{}_s{}_b{}".format(mode, frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, log_step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "{}_color_pred/f{}_s{}_b{}".format(mode, frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, log_step)

                    writer.add_image(
                        "{}_disp/f{}_s{}_b{}".format(mode, frame_id, s, j),
                        tensor2array(outputs[("disp", frame_id, s)][j], max_value=None, colormap='magma'),
                        log_step)
                    writer.add_image(
                        "{}_depth/f{}_s{}_b{}".format(mode, frame_id, s, j),
                        tensor2array(outputs[("depth", frame_id, s)][j], max_value=None),
                        log_step)
                    if frame_id != 0:
                        writer.add_image(
                            "{}_valid_mask/f{}_s{}_b{}".format(mode, frame_id, s, j),
                            tensor2array(outputs[("valid_mask", frame_id, s)][j], max_value=1, colormap='bone'),
                            log_step)

                if not self.opt.disable_automasking:
                    writer.add_image(
                        "{}_auto_mask/s{}_b{}".format(mode, s, j),
                        outputs[("auto_mask", s)][j][None, ...], log_step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            model_save = model.module if self.use_multi_gpu else model
            to_save = model_save.state_dict()
            # save the sizes - these are needed at prediction time
            to_save['height'] = self.opt.height
            to_save['width'] = self.opt.width
            to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
