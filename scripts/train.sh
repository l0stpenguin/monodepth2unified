GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
--log_dir ckpt \
--batch_size 12 \
--num_scales 1 \
--png --log_frequency 200 --print_frequency 1000 \
--geometry_consistency 0.1 \
--pose_model_type posecnn \
--rotation_mode euler \
--model_name mdp_g0.1_posecnn_euler