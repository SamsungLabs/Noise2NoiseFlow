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
from torch import nn

from model.noise_flow import NoiseFlow
from model.dncnn import DnCNN
from utils.train_utils import weights_init_kaiming, weights_init_orthogonal
from model.unet import UNet
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

class Noise2NoiseFlow(nn.Module):
    def __init__(self, x_shape, arch, flow_permutation, param_inits, lu_decomp, denoiser_model='dncnn', dncnn_num_layers=9, lmbda=262144):
        super(Noise2NoiseFlow, self).__init__()

        self.noise_flow = NoiseFlow(x_shape, arch, flow_permutation, param_inits, lu_decomp)
        if denoiser_model == 'dncnn':
            self.denoiser = DnCNN(x_shape[0], dncnn_num_layers)
            # TODO: self.dncnn should be named self.denoiser by definition, but I changed it here since i needed it to be backward compatible for loading previous models for sampling.
            # self.denoiser.apply(weights_init_kaiming)
            self.denoiser.apply(weights_init_orthogonal)
        elif denoiser_model == 'unet':
            self.denoiser = UNet(in_channels=4, out_channels=4)

        self.denoiser_loss = nn.MSELoss(reduction='mean')
        self.lmbda = lmbda

    def denoise(self, noisy, clip=True):
        denoised = self.denoiser(noisy)
        if clip:
            denoised = torch.clamp(denoised, 0., 1.)

        return denoised

    def forward_u(self, noisy, **kwargs):
        denoised = self.denoise(noisy)
        kwargs.update({'clean' : denoised})
        noise = noisy - denoised

        z, objective = self.noise_flow.forward(noise, **kwargs)

        return z, objective, denoised

    def symmetric_loss(self, noisy1, noisy2, **kwargs):
        denoised1 = self.denoise(noisy1)
        denoised2 = self.denoise(noisy2)
        
        noise1 = noisy1 - denoised2
        noise2 = noisy2 - denoised1

        kwargs.update({'clean' : denoised2})
        nll1, _ = self.noise_flow.loss(noise1, **kwargs)

        kwargs.update({'clean' : denoised1})
        nll2, _ = self.noise_flow.loss(noise2, **kwargs)

        nll = (nll1 + nll2) / 2
        return nll

    def symmetric_loss_with_mse(self, noisy1, noisy2, **kwargs):
        denoised1 = self.denoise(noisy1, clip=False)
        denoised2 = self.denoise(noisy2, clip=False)

        mse_loss1 = self.denoiser_loss(denoised1, noisy2)
        mse_loss2 = self.denoiser_loss(denoised2, noisy1)

        denoised1 = torch.clamp(denoised1, 0., 1.)
        denoised2 = torch.clamp(denoised2, 0., 1.)
        
        noise1 = noisy1 - denoised2
        noise2 = noisy2 - denoised1

        kwargs.update({'clean' : denoised2})
        nll1, _ = self.noise_flow.loss(noise1, **kwargs)

        kwargs.update({'clean' : denoised1})
        nll2, _ = self.noise_flow.loss(noise2, **kwargs)

        nll = (nll1 + nll2) / 2
        mse_loss = (mse_loss1 + mse_loss2) / 2

        return nll, mse_loss



    def _loss_u(self, noisy1, noisy2, **kwargs):
        denoised1 = self.denoise(noisy1, clip=False)

        mse_loss = self.denoiser_loss(denoised1, noisy2)

        denoised1 = torch.clamp(denoised1, 0., 1.)

        noise = noisy1 - denoised1
        kwargs.update({'clean' : denoised1})
        nll, _ = self.noise_flow.loss(noise, **kwargs)

        return nll, mse_loss

    def loss_u(self, noisy1, noisy2, **kwargs):
        # return self.symmetric_loss(noisy1, noisy2, **kwargs), 0, 0

        # nll, mse = self._loss_u(noisy1, noisy2, **kwargs)
        nll, mse = self.symmetric_loss_with_mse(noisy1, noisy2, **kwargs)

        return nll + self.lmbda * mse, nll.item(), mse.item()
        # return nll, nll.item(), mse.item()

    def forward_s(self, noise, **kwargs):
        return self.noise_flow.forward(noise, **kwargs)

    def _loss_s(self, x, **kwargs):
        return self.noise_flow._loss(x, **kwargs)

    def loss_s(self, x, **kwargs):
        return self.noise_flow.loss(x, **kwargs)

    def mse_loss(self, noisy, clean, **kwargs):
        denoised = self.denoise(noisy, clip=False)
        mse_loss = self.denoiser_loss(denoised, clean)
        psnr = batch_PSNR(denoised, clean, 1.)
        return mse_loss.item(), psnr

    def sample(self, eps_std=None, **kwargs):
        return self.noise_flow.sample(eps_std, **kwargs)
