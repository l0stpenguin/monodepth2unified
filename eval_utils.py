#!/usr/bin/env python
import numpy as np
import sys
from matplotlib import pyplot as plt

import h5py
import os, os.path
import cv2
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pred-pose", default=None, type=str)
parser.add_argument("--gt-pose", default=None, type=str)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--seq-id", type=str)


class kittiEvalOdom():
    # ----------------------------------------------------------------------
    # poses: N,4,4
    # pose: 4,4
    # ----------------------------------------------------------------------
    def __init__(self, pred_pose, gt_pose, seq_id):
        self.lengths= [100,200,300,400,500,600,700,800]
        self.num_lengths = len(self.lengths)
        self.pred_pose = pred_pose
        self.gt_pose = gt_pose
        self.seq_id = seq_id

    def loadPoses(self, file_name):
        # ----------------------------------------------------------------------
        # Each line in the file should follow one of the following structures
        # (1) idx pose(3x4 matrix in terms of 12 numbers)
        # (2) pose(3x4 matrix in terms of 12 numbers)
        # ----------------------------------------------------------------------
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ")]
            withIdx = int(len(line_split)==13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4+col+ withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            poses[frame_idx] = P
        return poses

    def trajectoryDistances(self, poses):
        # ----------------------------------------------------------------------
        # poses: dictionary: [frame_idx: pose]
        # ----------------------------------------------------------------------
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0,3] - P2[0,3]
            dy = P1[1,3] - P2[1,3]
            dz = P1[2,3] - P2[2,3]
            dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))	
        return dist

    def rotationError(self, pose_error):
        a = pose_error[0,0]
        b = pose_error[1,1]
        c = pose_error[2,2]
        d = 0.5*(a+b+c-1.0)
        return np.arccos(max(min(d,1.0),-1.0))

    def translationError(self, pose_error):
        dx = pose_error[0,3]
        dy = pose_error[1,3]
        dz = pose_error[2,3]
        return np.sqrt(dx**2+dy**2+dz**2)

    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calcSequenceErrors(self, poses_gt, poses_result):
        err = []
        dist = self.trajectoryDistances(poses_gt)
        self.step_size = 10
        
        for first_frame in range(9, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_)

                # ----------------------------------------------------------------------
                # Continue if sequence not long enough
                # ----------------------------------------------------------------------
                if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
                    continue

                # ----------------------------------------------------------------------
                # compute rotational and translational errors
                # ----------------------------------------------------------------------
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)

                # ----------------------------------------------------------------------
                # compute speed 
                # ----------------------------------------------------------------------
                num_frames = last_frame - first_frame + 1.0
                speed = len_/(0.1*num_frames)

                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err
        
    def saveSequenceErrors(self, err, file_name):
        fp = open(file_name,'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write+"\n")
        fp.close()

    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err 

    def plotPath(self, seq, poses_gt, poses_result):
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20
        plot_num =-1
            
        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xz = []
            # for pose in poses_dict[key]:
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0,3], pose[2,3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:,0], pos_xz[:,1], label = key)	
            
        plt.legend(loc = "upper right", prop={'size': fontsize_})
        plt.xticks(fontsize = fontsize_) 
        plt.yticks(fontsize = fontsize_) 
        plt.xlabel('x (m)',fontsize = fontsize_)
        plt.ylabel('z (m)',fontsize = fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_{:02}".format(seq)
        plt.savefig(self.plot_path_dir +  "/" + png_title + ".pdf",bbox_inches='tight', pad_inches=0)
        # plt.show()

    def plotError(self, avg_segment_errs):
        # ----------------------------------------------------------------------
        # avg_segment_errs: dict [100: err, 200: err...]
        # ----------------------------------------------------------------------
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            plot_y.append(avg_segment_errs[len_][0])
        fig = plt.figure()
        plt.plot(plot_x, plot_y)
        plt.show()

    def computeSegmentErr(self, seq_errs):
        # ----------------------------------------------------------------------
        # This function calculates average errors for different segment.
        # ----------------------------------------------------------------------

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []
        # ----------------------------------------------------------------------
        # Get errors
        # ----------------------------------------------------------------------
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])
        # ----------------------------------------------------------------------
        # Compute average
        # ----------------------------------------------------------------------
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:,0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:,1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def eval(self, result_dir):
        seq_id = int(self.seq_id)
        file_name = '{:02}.txt'.format(seq_id)

        poses_result = self.loadPoses(self.pred_pose)
        poses_gt = self.loadPoses(self.gt_pose)
        self.result_file_name = result_dir+file_name

        # ----------------------------------------------------------------------
        # compute sequence errors
        # ----------------------------------------------------------------------
        seq_err = self.calcSequenceErrors(poses_gt, poses_result)

        # ----------------------------------------------------------------------
        # compute overall error
        # ----------------------------------------------------------------------
        ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
        print("Sequence: " + str(seq_id))
        print("Average translational RMSE (%): ", ave_t_err*100)
        print("Average rotational error (deg/100m): ", ave_r_err/np.pi * 180 *100)
        with open(os.path.join(result_dir, 'pose_error.log'), 'w') as f:
            f.writelines("Sequence: " + str(seq_id) + "\n")
            f.writelines("Average translational RMSE (%): " + str(ave_t_err * 100) + "\n")
            f.writelines("Average rotational error (deg/100m): " + str(ave_r_err/np.pi * 180 *100) + "\n")

def compute_global_error(pred_pose, gt_pose, odom_result_dir, seq_id):
    odom_eval = kittiEvalOdom(pred_pose, gt_pose, seq_id)
    odom_eval.eval(odom_result_dir)

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs

# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse

def compute_local_error(pred_poses, gt_pose, data_path, sequence_id, odom_result_dir):
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
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

    save_path = os.path.join(odom_result_dir, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)

if __name__ == "__main__":
    args = parser.parse_args()
    compute_global_error(args.pred_pose, args.gt_pose, args.output_dir, args.seq_id)

