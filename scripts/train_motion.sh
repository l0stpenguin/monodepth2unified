GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python motion_train.py \
--log_dir ckpt_motion --split fusion_motion \
--batch_size 4 \
--num_epochs 30 \
--num_scales 1 \
--png --log_frequency 200 --print_frequency 200 \
--load_weights_folder ckpt_motion/motion_step5/models/weights_5 \
--model_name motion_step5