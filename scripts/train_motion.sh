GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python motion_train.py \
--log_dir ckpt_motion --split fusion_motion \
--batch_size 4 \
--num_epochs 30 \
--num_scales 4 \
--png --log_frequency 200 --print_frequency 200 \
--prediction_level object \
--model_name motion_object_4scale