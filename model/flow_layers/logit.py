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

class Logit(nn.Module):
    def __init__(self, temperature=1, eps=1e-6, device='cpu', name='logit'):
        super(Logit, self).__init__()
        self.name = name
        self.eps = eps
        self.register_buffer('temperature', torch.tensor([temperature], device=device))

    def _inverse(self, z, **kwargs):
        z = self.temperature * z
        x = torch.sigmoid(z)
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        """
        ldj explanation:
        log(x/(1-x)) = z => x/(1-x) = exp(z) (property 1) => (1-x)/x = 1/exp(x) => 1/x - 1 = 1/exp(z)
        => 1/x = 1 + 1/exp(z) (property 2)

        softplus(-z) + softplus(z) = log(1 + exp(-z) + log(1 + exp(z)))
                                   = log(1 + 1/exp(z)) + log(1 + exp(z))
                                   = log(1/x) + log(1 + x/(1-x))
                                   = -log(x) + log((1-x+x)/(1-x))
                                   = -log(x) + log(1/(1-x))
                                   = -log(x) - log(1-x)
        """
        z = (1 / self.temperature) * (torch.logit(x, eps=self.eps))
        ldj = torch.sum( - (torch.log(self.temperature) - F.softplus(-self.temperature * z) - F.softplus(self.temperature * z)), dim=[1, 2, 3])
        return z, ldj
