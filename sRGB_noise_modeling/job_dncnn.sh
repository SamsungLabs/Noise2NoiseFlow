
# Train the denoiser using samples from our pretrained noise model
python train_dncnn.py --sidd_path '../data/SIDD_Medium_Srgb/Data' --model DnCNN_NM \
    --noise_model_path our_model \
    --exp_name our_model_denoiser --num_workers 5 --nm_load_epoch 30
