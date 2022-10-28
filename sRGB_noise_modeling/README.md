Camera noise modeling and synthesis in the sRGB domain
===

This repository provides the code for training and testing the sRGB noise model introduced in the following paper:

[**Modeling sRGB Camera Noise with Normalizing Flows**](https://yorkucvil.github.io/sRGBNoise/)

It also provides code for training and testing DnCNN using samples from our model and our baseline models.

# Required libraries

Required libraries and their version can be found in `camera_noise_modeling/requirements.txt`

To install run: `pip install -r requirements.txt`

# Required dataset

[Smartphone Image Denoising Dataset (SIDD)](https://www.eecs.yorku.ca/~kamel/sidd/)

In this work, we use SIDD-Medium Dataset [SIDD_Medium_sRGB](eecs.yorku.ca/~kamel/sidd/dataset.php)

# Running the model

Find the commands needed for training, testing, sampling, and reproducing some of the results from the paper in `noise_model.sh`

`job_dncnn.sh` has an example command for training the DnCNN model using samples generated from our model. 

# Pretrained

Find our pretrainied model under the experiments directory.

# Paper

Shayan Kousha, Ali Maleky, [Marcus A. Brubaker](https://www.eecs.yorku.ca/~mab/), and [Michael S. Brown](https://www.eecs.yorku.ca/~mbrown/). Modeling sRGB Camera Noise with Normalizing Flows. In _CVPR_, 2022.

# Citation

If you use our sRGB noise model in your research, we kindly ask that you cite the paper as follows:

    @inproceedings{Kousha2022ModelingsRGB,
        title={{Modeling sRGB Camera Noise with Normalizing Flows}},
        author={Shayan Kousha and Ali Maleky and Michael Brown and Marcus A. Brubaker},
        year={2022},
        booktitle=CVPR}
    }

# Contact

Shayan Kousha ([shayanko@eecs.yorku.ca](mailto:shayanko@eecs.yorku.ca))
