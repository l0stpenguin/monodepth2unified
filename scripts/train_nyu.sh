CUDA_VISIBLE_DEVICES=6 python train.py \
--data_path /home/share/datasets/nyu_v2/nyu_v2_raw \
--log_dir /home/gaof/workspace/sfm/checkpoints_nyu \
--split nyu_toy \
--num_layers 18 \
--dataset nyu \
--scales 0 \
--min_depth 0.01 \
--max_depth 10 \
--frame_ids 0 -5 5 \
--log_frequency 50 \
--height 192 --width 256 \
--avg_reprojection \
--allow_backward \
--model_name monodepth2-am-ab-o2
# --avg_reprojection \
# --disable_automasking \

# --pose_model_type shared \
# --allow_backward \