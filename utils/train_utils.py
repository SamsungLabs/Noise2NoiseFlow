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
import torch
import torch.nn as nn
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def output_psnr_mse(img_orig, img_out, data_range):
    squared_error = torch.square(img_orig - img_out)
    mse = torch.mean(squared_error)
    psnr = 10 * torch.log10(data_range ** 2/ mse)
    return psnr

def mean_psnr(denoised, imclean, data_range=1.0):
    n_blk = imclean.shape[0]
    mean_psnr = 0
    psnrs = np.ndarray([n_blk])
    for b in range(n_blk):
        ref_block = imclean[b]
        res_block = denoised[b]
        psnr = output_psnr_mse(ref_block, res_block, data_range)
        mean_psnr += psnr
        psnrs[b] = psnr
    return mean_psnr / n_blk, psnrs

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def kldiv_simple(real_noise, sampled_noise):
    bw = 0.2 / 64	
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
    cnt_regr = 1	
    real_hist = get_histogram(real_noise, bin_edges=bin_edges, cnt_regr=cnt_regr)
    sampled_hist = get_histogram(sampled_noise, bin_edges=bin_edges, cnt_regr=cnt_regr)

    kld = kl_div_forward(real_hist, sampled_hist)

    return kld

def get_histogram(data, bin_edges=None, cnt_regr=1):
    n = np.prod(data.shape)	
    hist, _ = np.histogram(data, bin_edges)	
    return (hist + cnt_regr)/(n + cnt_regr * len(hist))

def kl_div_forward(p, q):
    assert (~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))).all()	
    idx = (p > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(p * np.log(p / q))

def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += structural_similarity(Iclean[i].transpose((1, 2, 0)), Img[i].transpose((1, 2, 0)), data_range=data_range,  sigma=0.8, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
    return (SSIM/Img.shape[0])

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)