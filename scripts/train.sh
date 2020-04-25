GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
--log_dir ckpt \
--batch_size 12 \
--num_scales 1 \
--png --log_frequency 200 --print_frequency 200 \
--model_name mdp_original