# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import KITTIOdomDataset
import models

def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
        "eval_split should be either odom_9 or odom_10"

    sequence_id = int(opt.eval_split.split("_")[1])

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))

    dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, 1], 4, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    pose_model_path = os.path.join(opt.load_weights_folder, "pose.pth")

    if opt.pose_model_type == "posecnn":
        pose_model = models.PoseCNN()
    else:
        pose_model = models.PoseModel(num_layers=opt.num_layers, pretrained=False)
    pose_model.load_state_dict(torch.load(pose_model_path))
    pose_model = pose_model.cuda()
    pose_model.eval()

    pred_poses = []
    global_pose = np.identity(4)
    pred_global_poses = [global_pose[0:3,:].reshape(1,12)]
    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

            rotation, translation = pose_model(all_color_aug)

            local_pose = transformation_from_parameters(
                    rotation, translation, 
                    rotation_mode=opt.rotation_mode,
                    invert=False).cpu().numpy()
            global_pose = global_pose @ np.linalg.inv(local_pose)
            pred_poses.append(local_pose)
            pred_global_poses.append(global_pose[0:3,:].reshape(1,12))

    pred_poses = np.concatenate(pred_poses)
    global_poses = np.concatenate(pred_global_poses)

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
