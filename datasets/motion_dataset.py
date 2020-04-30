# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import torch
import skimage.transform
import numpy as np
import PIL.Image as pil
import imageio

from .mono_dataset import MonoDataset
from torchvision import transforms
import random


class MotionDataset(MonoDataset):
    def __init__(self, num_class, *args, **kwargs):
        super(MotionDataset, self).__init__(*args, **kwargs)

        # self.K = np.array([[0.773, 0, 0.560, 0],
        #                    [0, 2.552, 0.598, 0],
        #                    [0,     0,     1, 0],
        #                    [0,     0,     0, 1]], dtype=np.float32)
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        self.num_class = num_class
        self.side_map = {"r": "right", "l": "left"}

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if "cs_motion" in folder:
            width, height = color.size
            color = color.crop([0, 203, width, height-203])

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        if "kitti_motion" in folder:
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
        else:
            seq_index = int(frame_index.split("_")[0])
            img_index = int(frame_index.split("_")[1])
            f_str = "{:06d}_{:06d}{}".format(seq_index, img_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, self.side_map[side], f_str)
        return image_path

    def get_mask(self, folder, frame_index, do_flip):
        """
        raw annotations for motion objects:
            Background: 255 and 0           -> class 0
            Static: 150                     -> class 1
            Probably Moving: 151 and 152    -> class 2
        """
        f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "label", f_str)
        raw_mask = imageio.imread(image_path).astype(np.float32)
        if "cs_motion" in folder:
            raw_mask = raw_mask[203:-203, :]
        raw_mask = skimage.transform.resize(
            raw_mask, (self.height, self.width), order=0, preserve_range=True, mode='constant')
        mask_gt = np.zeros_like(raw_mask)
        if self.num_class == 3:
            mask_gt[np.where(raw_mask==150)] = 1
            mask_gt[np.where(raw_mask==151)] = 2
            mask_gt[np.where(raw_mask==152)] = 2
        else:
            mask_gt[np.where(raw_mask==150)] = 1
            mask_gt[np.where(raw_mask==151)] = 1
            mask_gt[np.where(raw_mask==152)] = 1

        if do_flip:
            mask_gt = np.fliplr(mask_gt)

        return mask_gt

    def __getitem__(self, index):
        inputs = {}
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        data_source = line[0]
        folder = line[1]
        if self.is_train:
            folder = os.path.join(data_source, "train", folder)
        else:
            folder = os.path.join(data_source, "val", folder)

        if len(line) == 4:
            if "kitti_motion" in folder:
                frame_index = int(line[2])
            else:
                frame_index = line[2]
            side = line[3]
        else:
            print(line)
            exit()
            frame_index = 0
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "left", "l": "right"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                if "kitti_motion" in folder:
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                else:
                    seq_index = int(frame_index.split("_")[0])
                    img_index = int(frame_index.split("_")[1])
                    neighbor_frame_index = "{}_{}".format(seq_index, img_index + i)
                    inputs[("color", i, -1)] = self.get_color(folder, neighbor_frame_index, side, do_flip)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        mask_gt = self.get_mask(folder, frame_index, do_flip)
        inputs["mask_gt"] = torch.from_numpy(mask_gt.astype(np.float32)).long()

        return inputs