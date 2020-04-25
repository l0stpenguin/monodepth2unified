GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
--log_dir ckpt --split tiny_kitti \
--batch_size 4 \
--num_scales 4 \
--png --log_frequency 200 --print_frequency 200 \
--model_name tiny_4scale