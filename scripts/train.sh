GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
--log_dir ckpt \
--batch_size 12 \
--num_scales 1 \
--png --log_frequency 200 --print_frequency 200 \
--brightness 0.5 \
--occlusion_mode explicit \
--occlusion_penalty 0.01 \
--geometry_consistency 0.01 \
--model_name mdp_p0.5_o0.01_g0.01