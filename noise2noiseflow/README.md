Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images
===

This directory provides the code for training and testing the Noise2NoiseFlow model introduced in the following paper:

[**Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images**](https://yorkucvil.github.io/Noise2NoiseFlow/)

# Required libraries

Required libraries and their version can be found in `camera_noise_modeling/requirements.txt`

To install run: `pip install -r requirements.txt`

# Required dataset

[Smartphone Image Denoising Dataset (SIDD)](https://www.eecs.yorku.ca/~kamel/sidd/)

In this work, we use SIDD-Full Dataset for training and SIDD-Medium for evaluation. Note that we use the raw images.

**NOTE**: Please change the references to SIDD-Full path (`sidd_path`) and SIDD-Medium-Raw path (`sidd_medium_path`) in both the argument default values and inside the training scripts to the path where you download them. For SIDD-Full, it is intended to point to a directory containing a directory for each of the image scenes, and for SIDD-Medium-Raw it should point to a directory with another directory inside it named `Data` that contains a folder for each of the SIDD image scenes. The dataloaders have been written to work well with the latest versions of SIDD that one can download from the website. In case it changes in the future, you can always adapt it to your own choice of directory formats by modifying `sidd_full_filenames_tuple` and `sidd_medium_filenames_tuple` functions in the `data_loader/sidd_utils.py` file.

You can use your own dataset to train our model. You just need to replace the dataloader to adapt to your custom dataset and modify the training script.

# Running the model

Find the commands needed for training, testing, sampling, and reproducing some of the results from the paper in `commands.sh`. You will need to download the SIDD-Full and SIDD-Medium datasets and change the dataset paths in the code.

The file `commands.sh` provides an overview of the script arguments. For a full list of arguments, please see `utils/arg_parser.py`.

# Pretrained Model

We have provided a model trained on SIDD for 160 epochs under the `experiments/paper/pretrained_model` folder. You can use the commands provided in the `commands.sh` file to resume the training or write your own script to use the model for noise modeling/sampling/denoising.

We have also provided two pretrained denoisers for _DnCNN_ and _UNet_ respectively trained with noisy-noisy pairs from SIDD-Full under the `denoisers` directory that are used as the starting point of the denoiser for training the main _"Noise2NoiseFlow"_ model in case you use the `--pretrained_denoiser` flag as input argument.

# Paper

Ali Maleky, Shayan Kousha, [Michael S. Brown](https://www.eecs.yorku.ca/~mbrown/), and [Marcus A. Brubaker](https://www.eecs.yorku.ca/~mab/). Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images. In _CVPR_, 2022.

# Citation

If you use our model in your research, we kindly ask that you cite the paper as follows:

    @inproceedings{maleky2022noise,
        title={{Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images}},
        author={Maleky, Ali and Kousha, Shayan and Brown, Michael S. and Brubaker, Marcus A.},
        booktitle={CVPR},
        year={2022}
    }

# Contact

Ali Maleky ([ali.maleky7997@gmail.com](mailto:ali.maleky7997@gmail.com))