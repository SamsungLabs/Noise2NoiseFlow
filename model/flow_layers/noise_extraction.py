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
import torch.nn.functional as F

class NoiseExtraction(nn.Module):
    def __init__(self, device='cpu', name='noise_extraction'):
        super(NoiseExtraction, self).__init__()
        self.name = name
        self.device = device

    def _inverse(self, z, **kwargs):
        x = z + kwargs['clean']
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        z = x - kwargs['clean']
        ldj = torch.zeros(x.shape[0], device=self.device)
        return z, ldj
