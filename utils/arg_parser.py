"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str,
                        default='./logdir/', help="Location to save logs")
    parser.add_argument("--sidd_path", type=str,
                        default='./data/SIDD_Medium_Raw/Data', help="Location of the SIDD dataset")
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int,
                        default=-1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=128, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=100, help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--lu_decomp', action='store_true', default=False)
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--n_bits_x", type=int, default=10,
                        help="Number of bits of x")
    parser.add_argument("--do_sample", action='store_true',
                        help="To sample noisy images from the test set.")
    # Ablation
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=1,
                        help="Type of flow. 0=reverse, 1=1x1conv")
    # for SIDD
    parser.add_argument("--dataset_type", type=str, choices=['full', 'medium'],
                        help="Full or medium")
    parser.add_argument("--patch_height", type=int,
                        help="Patch height, width will be the same")
    parser.add_argument("--patch_sampling", type=str,
                        help="Patch sampling method form full images (uniform | random)")
    parser.add_argument("--n_tr_inst", type=int,
                        help="Number of training scene instances")
    parser.add_argument("--n_ts_inst", type=int,
                        help="Number of testing scene instances")
    parser.add_argument("--n_patches_per_image", type=int,
                        help="Max. number of patches sampled from each image")
    parser.add_argument("--start_tr_im_idx", type=int,
                        help="Starting image index for training")
    parser.add_argument("--end_tr_im_idx", type=int,
                        help="Ending image index for training")
    parser.add_argument("--start_ts_im_idx", type=int,
                        help="Starting image index for testing")
    parser.add_argument("--end_ts_im_idx", type=int,
                        help="Ending image index for testing")
    parser.add_argument("--camera", type=str,
                        help="To choose training scenes from one camera")  # to lower image loading frequency
    parser.add_argument("--iso", type=int,
                        help="To choose training scenes from one camera")  # to lower image loading frequency
    parser.add_argument("--arch", type=str, default='',  required=True,
                        help="Defines a mixture architecture of bijectors")
    parser.add_argument("--n_train_threads", type=int,
                        help="Number of training/testing threads")
    parser.add_argument("--n_channels", type=int, default=4,
                        help="Number of image channles")
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument("--lmbda", type=int, default=1, help="value for lambda in Noise2NoiseFlow loss term")
    parser.add_argument("--denoiser", type=str, default='dncnn',
                        help="Denoiser architecture type, choose between dncnn/unet.")

    parser.add_argument("--alpha", type=float, default=4, help="Alpha parameter in recorruption")
    parser.add_argument("--sigma", type=float, default=1/256, help="std of the zero mean noise vector z for recorruption")
    parser.add_argument("--pretrained_denoiser", default=True)
    
    hps = parser.parse_args()  # So error if typo
    return hps
