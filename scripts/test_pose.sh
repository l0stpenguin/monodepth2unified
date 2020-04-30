python evaluate_pose.py --data_path /home/share/lmDISK/datasets/kitti/odometry \
--load_weights_folder ckpt/tiny_g0.001/models/weights_19 \
--eval_split odom_9
echo "----------------------------------------"
python evaluate_pose.py --load_weights_folder ckpt/tiny_g0.001/models/weights_49 --eval_split odom_9
echo "----------------------------------------"