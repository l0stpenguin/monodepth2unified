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

import json
import tqdm
from path import Path
from eval_utils import *

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
                               [0, 1], 4, is_train=False, img_ext=".png" if opt.png else ".jpg")
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    pose_model_path = os.path.join(opt.load_weights_folder, "pose.pth")
    result_dir = os.path.join(opt.load_weights_folder, "pose")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print("-> Results dir: {}".format(result_dir))

    if opt.pose_model_type == "posecnn":
        pose_model = models.PoseCNN()
    else:
        pose_model = models.PoseModel(num_layers=opt.num_layers, pretrained=False)
    pose_model.load_state_dict(torch.load(pose_model_path), strict=False)
    pose_model = pose_model.cuda()
    pose_model.eval()

    global_pose = np.identity(4)
    pred_global_poses = [global_pose[0:3,:].reshape(1,12)]
    pred_local_poses = []
    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

            rotation, translation = pose_model(all_color_aug)

            local_pose = transformation_from_parameters(
                    rotation, translation, 
                    rotation_mode=opt.rotation_mode,
                    invert=False).cpu().numpy()
            global_pose = global_pose @ np.linalg.inv(local_pose)
            pred_local_poses.append(local_pose)
            pred_global_poses.append(global_pose[0:3,:].reshape(1,12))

    local_poses = np.concatenate(pred_local_poses)
    global_poses = np.concatenate(pred_global_poses, axis=0)

    print("-> Computing local error: ")
    compute_global_error(global_poses, opt.data_path, sequence_id, result_dir)
    print("-> Computing global error: ")
    compute_local_error(local_poses, opt.data_path, sequence_id, result_dir)


def compute_local_error(local_poses, data_path, sequence_id, result_dir):
    gt_poses_path = os.path.join(data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(local_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))
    with open(os.path.join(result_dir, "local_error.log"), "w") as f:
        f.writelines("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))
    save_path = os.path.join(result_dir, "local_poses_{:02d}.npy".format(sequence_id))
    np.save(save_path, local_poses)
    print("-> Local poses saved to", save_path)


def globally_scale_poses(pred, gt):
    pred = pred.reshape((-1, 3, 4))
    gt = gt.reshape((-1, 3, 4))
    scale_factor = np.sum(pred[:, :, -1] * gt[:, :, -1]) / np.sum(pred[:, :, -1] ** 2)
    pred[:, :, -1] = pred[:, :, -1] * scale_factor
    return pred.reshape((-1, 12))


def compute_global_error(global_poses, data_path, sequence_id, result_dir):
    raw_poses_path = os.path.join(result_dir, "global_poses_{:02d}.txt".format(sequence_id))
    np.savetxt(raw_poses_path, global_poses, delimiter=" ", fmt="%1.8e")
    print("-> Global poses saved to", raw_poses_path)

    gt_poses_path = os.path.join(data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_poses = np.genfromtxt(gt_poses_path)
    scaled_poses = globally_scale_poses(global_poses, gt_poses)
    scaled_poses_path = os.path.join(result_dir, "scaled_poses_{:02d}.txt".format(sequence_id))
    np.savetxt(scaled_poses_path, scaled_poses, delimiter=" ", fmt="%1.8e")
    print("-> Scaled poses saved to", scaled_poses_path)

    odom_eval = kittiEvalOdom(scaled_poses_path, gt_poses_path, sequence_id)
    odom_eval.eval(result_dir)


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()

    saved_opt_path = Path(opt.load_weights_folder).abspath().dirname()/"opt.json"
    saved_model_path = opt.load_weights_folder
    data_path = opt.data_path
    eval_split = opt.eval_split
    with open(saved_opt_path, "r") as f:
        opt.__dict__ = json.load(f).copy()
    opt.load_weights_folder = saved_model_path
    opt.eval_split = eval_split
    opt.data_path = data_path

    evaluate(opt)
