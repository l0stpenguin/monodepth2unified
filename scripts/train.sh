GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
--log_dir ckpt --split tiny_kitti \
--batch_size 4 \
--num_epochs 50 \
--num_scales 1 \
--occlusion_mode explicit \
--occlusion_penalty 1e-3 \
--png --log_frequency 200 --print_frequency 200 \
--load_weights_folder ckpt/tiny_o0.001/models/weights_20 \
--model_name tiny_o0.001
# --depth_model_type dispnet \
# --pose_model_type posecnn \
# --geometry_consistency 1e-3 \
# --png --log_frequency 200 --print_frequency 200 \
# --model_name tiny_g0.001