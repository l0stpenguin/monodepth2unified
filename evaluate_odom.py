import torch
from imageio import imread, imsave
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from scipy.ndimage.interpolation import zoom
from .eval_utils import compute_err

import models
import cv2
import os

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--posenet", required=True, choices=['PoseNet', 'PoseResNet', 'FixPoseNet'])
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat', 'axisangle'], type=str)
parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
parser.add_argument("--gt-pose", default=None, type=str, help="groundtruth poses")
parser.add_argument("--pred-pose", default=None, type=str)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h,w,_ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255 - 0.5)/0.5).to(device)
    return tensor_img

@torch.no_grad()
def inference(args):
    # args = parser.parse_args()

    weights_pose = torch.load(args.pretrained_posenet)
    pose_net = getattr(models, args.posenet)().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()
    
    image_dir = Path(os.path.join(args.dataset_dir, args.sequence, "image_2/"))
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()
    print('{} files to test'.format(len(test_files)))

    global_pose = np.identity(4)
    poses=[global_pose[0:3,:].reshape(1,12)]

    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args)

    print('rotation mode:', args.rotation_mode)
    for iter in tqdm(range(n - 1)):
        tensor_img2 = load_tensor_image(test_files[iter+1], args)
        pose = pose_net(tensor_img1, tensor_img2)
        if args.rotation_mode in ['euler', 'quat']:
            pose_mat = pose_vec2mat(pose, args.rotation_mode)  # [B,3,4]
        else:
            translation_vector = pose[:, None, :3]  # [B, 3, 1]
            axisangle_vector = pose[:, None, 3:]
            pose_mat = transformation_from_parameters(axisangle_vector, translation_vector)
        pose_mat = pose_mat.squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
        global_pose = global_pose @ np.linalg.inv(pose_mat)

        poses.append(global_pose[0:3,:].reshape(1,12))

        # update 
        tensor_img1 = tensor_img2

    poses = np.concatenate(poses, axis=0)
    filename = Path(os.path.join(args.output_dir, args.sequence + ".txt"))
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')

    gt = np.genfromtxt(args.gt_pose)
    scaled_poses = scale_prediction(poses, gt)
    filename_scaled = Path(os.path.join(args.output_dir, 'scaled_' + args.sequence + ".txt"))
    np.savetxt(filename_scaled, scaled_poses, delimiter=' ', fmt='%1.8e')

    return filename, filename_scaled

def scale_prediction(pred, gt):
    pred = pred.reshape((-1, 3, 4))
    gt = gt.reshape((-1, 3, 4))
    scale_factor = np.sum(pred[:, :, -1] * gt[:, :, -1]) / np.sum(pred[:, :, -1] ** 2)
    pred[:, :, -1] = pred[:, :, -1] * scale_factor
    return pred.reshape((-1, 12))

def main():
    args = parser.parse_args()

    if args.pred_pose is None:
        print("estimating poses first ...")
        poses_path, scaled_poses_path = inference(args)
    else:
        print("loading poses from the existed file ...")
        pred = np.genfromtxt(args.pred_pose)
        gt = np.genfromtxt(args.gt_pose)
        scaled_poses = scale_prediction(pred, gt)
        scaled_poses_path = os.path.join(args.output_dir, 'scaled_' + args.sequence + ".txt")
        np.savetxt(scaled_poses_path, scaled_poses, delimiter=' ', fmt='%1.8e')
    print("computing errors ...")
    compute_err(scaled_poses_path, args.gt_pose, args.output_dir, args.sequence)

if __name__ == '__main__':
    main()
