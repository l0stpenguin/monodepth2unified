CUDA_VISIBLE_DEVICES=4 python train.py \
--data_path /home/share/datasets/kitti/raw_data_sync \
--model_name kitti_192_640 \
--log_dir checkpoints \
--split eigen_zhou \
--png \
--scales 0 \
--min_depth 1e-3 \
--max_depth 80 \
--log_frequency 50