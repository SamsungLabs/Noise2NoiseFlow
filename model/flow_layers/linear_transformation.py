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
import numpy as np

class ConditionalLinear(nn.Module):
    def __init__(self, device='cpu', name='linear_transformation'):
        super(ConditionalLinear, self).__init__()
        self.name = name

        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)
        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'

        self.log_scale = nn.Parameter(torch.zeros(25), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(25), requires_grad=True)

    def _inverse(self, z, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        iso = gain_one_hot.nonzero()[:, 1]
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam = cam_one_hot.nonzero()[:, 1]
        iso_cam = iso * 5 + cam
        iso_cam = torch.arange(0, 25).cuda() == iso_cam.unsqueeze(1)

        log_scale = self.log_scale.unsqueeze(0).repeat_interleave(z.shape[0], dim=0)[iso_cam]
        bias = self.bias.unsqueeze(0).repeat_interleave(z.shape[0], dim=0)[iso_cam]

        x = (z - bias.reshape((-1, 1, 1, 1))) / torch.exp(log_scale.reshape((-1, 1, 1, 1)))
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        iso = gain_one_hot.nonzero()[:, 1]
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam = cam_one_hot.nonzero()[:, 1]
        iso_cam = iso * 5 + cam
        iso_cam = torch.arange(0, 25).cuda() == iso_cam.unsqueeze(1)

        log_scale = self.log_scale.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)[iso_cam]
        bias = self.bias.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)[iso_cam]
        
        z = x * torch.exp(log_scale.reshape((-1, 1, 1, 1))) + bias.reshape((-1, 1, 1, 1))
        log_abs_det_J_inv = log_scale * np.prod(x.shape[1:])

        return z, log_abs_det_J_inv

class ConditionalLinearExp2(nn.Module):
    def __init__(self, device='cpu', name='linear_transformation_exp2'):
        super(ConditionalLinearExp2, self).__init__()
        self.name = name

        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)
        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'

        self.log_scale = nn.Parameter(torch.zeros(25, 3), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(25, 3), requires_grad=True)

    def _inverse(self, z, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        iso = gain_one_hot.nonzero()[:, 1]
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam = cam_one_hot.nonzero()[:, 1]
        iso_cam = iso * 5 + cam
        iso_cam = torch.arange(0, 25).cuda() == iso_cam.unsqueeze(1)

        log_scale = self.log_scale.unsqueeze(0).repeat_interleave(z.shape[0], dim=0)[iso_cam]
        bias = self.bias.unsqueeze(0).repeat_interleave(z.shape[0], dim=0)[iso_cam]

        x = (z - bias.reshape((-1, z.shape[1], 1, 1))) / torch.exp(log_scale.reshape((-1, z.shape[1], 1, 1)))
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        iso = gain_one_hot.nonzero()[:, 1]
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam = cam_one_hot.nonzero()[:, 1]
        iso_cam = iso * 5 + cam
        iso_cam = torch.arange(0, 25).cuda() == iso_cam.unsqueeze(1)

        log_scale = self.log_scale.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)[iso_cam]
        bias = self.bias.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)[iso_cam]
        z = x * torch.exp(log_scale.reshape((-1, x.shape[1], 1, 1))) + bias.reshape((-1, x.shape[1], 1, 1))
        log_abs_det_J_inv = torch.sum(log_scale * np.prod(x.shape[2:]), dim=1)

        return z, log_abs_det_J_inv

class Gamma(nn.Module):
    def __init__(self, device='cpu', name='gamma'):
        super(Gamma, self).__init__()
        self.name = name
        self.gamma = nn.Parameter(torch.tensor(2.2), requires_grad=True)
        self.constant = 0.00005

    def _inverse(self, z, **kwargs):
        z = z.clamp(min=0)
        # print(torch.min(z), torch.max(z))
        x = z**(1/self.gamma)
        x -= self.constant
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        x += self.constant
        z = x**self.gamma
        log_abs_det_J_inv = torch.sum(torch.log(self.gamma) + (self.gamma-1)*torch.log(x), dim=[1, 2, 3])

        if 'writer' in kwargs.keys():
            kwargs['writer'].add_scalar('model/' + self.name + '_train', self.gamma, kwargs['step'])

        return z, log_abs_det_J_inv

