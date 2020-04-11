GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
--log_dir ckpt \
--png --log_frequency 200 \
--geometry_consistency 1e-2 \
--pose_model_type posecnn \
--rotation_mode euler \
--model_name mdp_g0.01_posecnn_euler