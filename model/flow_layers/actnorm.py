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
from torch.nn import functional as F, init
import numpy as np

class ActNorm(nn.Module):
    def __init__(self, features, name='actnorm', device='cpu'):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.
        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        if not is_positive_int(features):
            raise TypeError('Number of features must be a positive integer.')
        super().__init__()
        self.name = name
        self.initialized = False
        self.device=device
        self.log_scale = nn.Parameter(torch.zeros(features, device=device))
        self.shift = nn.Parameter(torch.zeros(features, device=device))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def _broadcastable_scale_shift(self, inputs):
        if inputs.dim() == 4:
            return self.scale.view(1, -1, 1, 1), self.shift.view(1, -1, 1, 1)
        else:
            return self.scale.view(1, -1), self.shift.view(1, -1)

    def _forward_and_log_det_jacobian(self, inputs, **kwargs):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Expecting inputs to be a 2D or a 4D tensor.')

        if self.training and not self.initialized:
            self._initialize(inputs)

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = scale * inputs + shift

        if inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = h * w * torch.sum(self.log_scale) * torch.ones(batch_size, device=self.device)
        else:
            batch_size,_ = inputs.shape
            logabsdet = torch.sum(self.log_scale) * torch.ones(batch_size, device=self.device)

        return outputs, logabsdet

    def _inverse(self, inputs, **kwargs):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Expecting inputs to be a 2D or a 4D tensor.')

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = (inputs - shift) / scale

        if inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = - h * w * torch.sum(self.log_scale) * torch.ones(batch_size, device=self.device)
        else:
            batch_size, _ = inputs.shape
            logabsdet = - torch.sum(self.log_scale) * torch.ones(batch_size, device=self.device)

        return outputs

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)

        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu

        self.initialized = True

def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0