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


class MotionTrainer:
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

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.models["mask"] = models.MaskModel(
            num_layers=self.opt.num_layers,
            num_scales=self.opt.num_scales,
            pretrained=(self.opt.weights_init=="pretrained"))
        self.models["mask"].to(self.device)
        self.parameters_to_train += list(self.models["mask"].parameters())

        if self.use_multi_gpu:
            self.models["mask"] = nn.DataParallel(self.models["mask"])
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.5)

        self.start_epoch = 0
        self.start_step = 0
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.dataset = datasets.MotionDataset
        
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = ".png" if self.opt.png else ".jpg"

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, 
            self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, 
            pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, 
            self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, 
            pin_memory=True, drop_last=True)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

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
        self.epoch = self.start_epoch
        self.step = self.start_step
        self.start_time = time.time()
        for self.epoch in range(self.start_epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        print(">> current learning rate: {}".format(self.model_lr_scheduler.get_lr()))
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
                self.log("train", inputs, outputs, losses, self.step)

            self.step += 1

        self.val()
        self.model_lr_scheduler.step()


    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        image_feats = {f_i: inputs[("color_aug", f_i, 0)] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if f_i < 0:
                image_inputs = [image_feats[f_i], image_feats[0]]
            else:
                image_inputs = [image_feats[0], image_feats[f_i]]

            image_inputs = torch.cat(image_inputs, 1)
            mask = self.models["mask"](image_inputs, f_i)
            outputs.update(mask)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses


    def val(self):
        """Validate the model
        """
        def merge_dict(dict1, dict2):
            for k, v in dict1.items():
                if k in dict2.keys():
                    dict2[k] += v
                else:
                    dict2[k] = v

        self.set_eval()
        val_losses = {}

        for batch_idx, inputs in tqdm(enumerate(self.val_loader), 
            desc='Val Epoch {}'.format(self.epoch), ncols=80, leave=True, total=len(self.val_loader)):
            with torch.no_grad():
                outputs, losses = self.process_batch(inputs)
                if batch_idx == 0:
                    val_inputs = inputs
                    val_outputs = outputs

                merge_dict(losses, val_losses)
                del inputs, outputs, losses
        
        self.log("val", val_inputs, val_outputs, val_losses, self.epoch)
        self.set_train()


    def log(self, mode, inputs, outputs, losses, log_step):
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}_{}".format(mode, l), v, log_step)

        for j in range(min(2, self.opt.batch_size)):  # write a maxmimum of four images
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "{}_color/f{}_b{}".format(mode, frame_id, j),
                    inputs[("color", frame_id, 0)][j].data, 
                    log_step)
                if frame_id == 0:
                    writer.add_image(
                        "{}_target/f{}_b{}".format(mode, frame_id, j),
                        mask2image(inputs["mask_gt"][j]),
                        log_step)
                if frame_id != 0:
                    writer.add_image(
                        "{}_pred/f{}_b{}".format(mode, frame_id, j),
                        mask2image(outputs[("mask", frame_id, 0)][j]),
                        log_step)


    def log_time(self, batch_idx, duration, losses):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            (self.num_total_steps - self.start_step) / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:3.1f} | loss: {:.4f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss"].cpu().data, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))


    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0
        for frame_id in self.opt.frame_ids[1:]:
            # print(type(outputs[("mask", frame_id, 0)]))
            # print(type(inputs["mask_gt"]))
            ce_loss = nn.functional.cross_entropy(
                outputs[("mask", frame_id, 0)], inputs["mask_gt"])
            total_loss += ce_loss.clone()
        
        total_loss /= (len(self.opt.frame_ids) - 1)
        losses["loss"] = total_loss
        return losses


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
        opt_dict = {
            'adam': self.model_optimizer.state_dict(),
            'epoch': self.epoch+1,
            'step': self.step
        }
        torch.save(opt_dict, save_path)

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
            self.model_optimizer.load_state_dict(optimizer_dict["adam"])
            self.start_epoch = optimizer_dict["epoch"]
            self.start_step = optimizer_dict["step"]
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")