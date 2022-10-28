# For a full list of input arguments and their usages, see 'arg_parser.py'

# --arch: The architecture of the noise model used. In our framework, they all have been implemented as flow transformations. 
# Options for arch include "sdn (signal dependent layer)", "unc (combination of an affine coupling followed by an invertible Conv1x1)", "gain (Gain layer)", "lt (Linear Transformation)".
# --sidd_path: Path to either SIDD_Full (For Noise2NoiseFlow and Noise2Noise) or SIDD_Meduim (Noise Flow). Note that You will have to change the default values to the path where you download SIDD.
# --iso and --cam: used to run the model on a specific camera sensor or iso level.
# --n_patches_per_image: Number of patches to extract from each image.
# --patch_height: Patch size (patches are square shaped).
# --patch_sampling: Either uniform or random or dncnn.
# --epochs_full_valid: How often the model performs an evaluation in training.
# --lu_decomp: Whether to use LU Decomposition in invertible Conv1x1 or not.
# --logdir: The name of the model
# --lmbda: The value of the lambda variable in the loss formulation
# --no_resume: Activates no_resume mode that ignores previous checkpoints in the 'logdir' dircetory if any.
# --do_sample: whether ro evaluate sampling performance in each evaluation epochs.
# --denoiser: dncnn or unet (default is dncnn).


# Example: training the Noise2NoiseFlow model (with Noise Flow as the noise model (S-Ax4-G-Ax4-CAM) and DnCNN as the denoiser).
python train_noise2noiseflow.py --arch "sdn|unc|unc|unc|unc|gain|unc|unc|unc|unc"      --sidd_path '../_Data/SIDD_Full' \
	--epochs 2000  --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 --patch_height 32 --patch_sampling uniform  \
	--n_channels 4 --epochs_full_valid 10 --lu_decomp --logdir n2nf --lmbda 262144 --no_resume


# Example: training the Noise2NoiseFlow model (with Heteroscedastic Gaussian Model (NLF) as the noise model (S-Ax4-G-Ax4-CAM) and UNet as the denoiser) with sampling at each evaluation epoch.
python train_noise2noiseflow.py --arch "sdn"      --sidd_path '../_Data/SIDD_Full' \
	--epochs 2000  --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 --patch_height 32 --patch_sampling uniform  \
	--n_channels 4 --epochs_full_valid 10 --lu_decomp --logdir n2nf_with_sample_unet --lmbda 262144 --denoiser unet --no_resume --do_sample


# Example: Resuming the trained Noise2NoiseFlow model (with Noise Flow as the noise model (S-Ax4-G-Ax4-CAM) and DnCNN as the denoiser) with sampling at each evaluation epoch.
python train_noise2noiseflow.py --arch "sdn"      --sidd_path '../_Data/SIDD_Full' \
	--epochs 2000  --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 --patch_height 32 --patch_sampling uniform  \
	--n_channels 4 --epochs_full_valid 10 --lu_decomp --logdir pretrained_model --lmbda 262144 --do_sample


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Example: training the Noise Flow model (S-Ax4-G-Ax4-CAM).
python train_noise_flow.py --arch "sdn|unc|unc|unc|unc|gain|unc|unc|unc|unc" \
    --sidd_path '../data/SIDD_Medium_Raw/Data' \
    --epochs 2000  --n_batch_train 138 --n_batch_test 138  --n_patches_per_image 2898 \
    --patch_height 32 --patch_sampling uniform  --n_channels 4 --epochs_full_valid 10 \
    --no_resume --logdir noiseflow --do_sample --lu_decomp


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Example: training a Noise2Noise model.

# --swap_images: whether to give both (I_1, I_2) and (I_2, I_1) as training pairs to the model or just (I_1, I_2). For a better performance, always set it to True.

python train_dncnn_noise2noise.py --sidd_path '../_Data/SIDD_Full/' --swap_images --logdir test
