# Commands for training our model and baselines from scratch. 
# For each model you can find a summary of thier expected NLL, expected kl_d, # of params.

## Isotropic Gaussian (3.703, 0.091,50)
python3 -m train_noise_model --logdir isotropic_gaussian --arch "CL"  --sidd_path '../data/SIDD_Medium_Srgb/Data' \
    --width 4 --epochs 2000 --lr 1e-4 --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 \
    --patch_height 32 --patch_sampling uniform  --n_channels 3 --lu_decomp --epochs_full_valid 10 --do_sample --no_resume

## Diagonal Gaussian (3.678, 0.079, 150)
python3 -m train_noise_model --logdir diagonal_gaussian --arch "CL2"  --sidd_path '../data/SIDD_Medium_Srgb/Data' \
    --width 4 --epochs 2000 --lr 1e-4 --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 \
    --patch_height 32 --patch_sampling uniform  --n_channels 3 --lu_decomp --epochs_full_valid 10 --do_sample --no_resume


## Full Gaussian (3.608, 0.085, 525)
python3 -m train_noise_model --logdir full_gaussian --arch "cond_c1x1"  --sidd_path '../data/SIDD_Medium_Srgb/Data' \
    --width 4 --epochs 2000 --lr 1e-4 --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 \
    --patch_height 32 --patch_sampling uniform  --n_channels 3 --lu_decomp --epochs_full_valid 10 --do_sample --no_resume


## Heteroscedastic Gaussian (3.642, 0.088, 72)
python3 -m train_noise_model --logdir heteroscedastic_gaussian --arch "SDN|CL"  --sidd_path '../data/SIDD_Medium_Srgb/Data' \
    --width 4 --epochs 2000 --lr 1e-4 --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 \
    --patch_height 32 --patch_sampling uniform  --n_channels 3 --lu_decomp --epochs_full_valid 10 --do_sample --no_resume


## Noise flow (3.311, 0.198, 2330)
python3 -m train_noise_model --logdir noise_flow --arch "AC|AC|AC|AC|Gain2|AC|AC|AC|AC|SDN"  --sidd_path '../data/SIDD_Medium_Srgb/Data' \
    --width 4 --epochs 2000 --lr 1e-4 --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 \
    --patch_height 32 --patch_sampling uniform  --n_channels 3 --lu_decomp --epochs_full_valid 10 --do_sample --no_resume


## Our model (3.072, 0.044, 6160)
python3 -m train_noise_model --logdir our_model --arch "CL2|CAC|CAC|CL2|CAC|CAC|CL2|CAC|CAC|CL2|CAC|CAC"  --sidd_path '../data/SIDD_Medium_Srgb/Data' \
    --width 4 --epochs 2000 --lr 1e-4 --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 \
    --patch_height 32 --patch_sampling uniform  --n_channels 3 --lu_decomp --epochs_full_valid 10 --do_sample --no_resume



# Command for using our pretrained model and sampling
python3 -m train_noise_model --logdir our_model --arch "CL2|CAC|CAC|CL2|CAC|CAC|CL2|CAC|CAC|CL2|CAC|CAC"  \
    --sidd_path '../data/SIDD_Medium_Srgb/Data'     --width 4 --epochs 2000 --lr 1e-4 --n_batch_train 138 --n_batch_test 138 \
     --n_patches_per_image 2898     --patch_height 32 --patch_sampling uniform  --n_channels 3 --lu_decomp \
     --epochs_full_valid 10 --do_sample
