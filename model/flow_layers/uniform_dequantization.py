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

class UniformDequantization(nn.Module):
    def __init__(self, num_bits=8, device='cpu', name='uniform_dequantization'):
        super(UniformDequantization, self).__init__()
        self.num_bits = num_bits
        self.quantization_bins = 2**num_bits
        self.register_buffer(
            'ldj_per_dim',
            - num_bits * torch.log(torch.tensor(2, device=device, dtype=torch.float))
        )
        self.name = name

    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = shape[1:].numel()
        ldj = self.ldj_per_dim * num_dims
        return ldj.repeat(batch_size)

    def _inverse(self, z, **kwargs):
        z = self.quantization_bins * z
        return z.floor().clamp(min=0, max=self.quantization_bins-1)

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        u = torch.rand(x.shape, device=self.ldj_per_dim.device, dtype=self.ldj_per_dim.dtype)
        z = (x.type(u.dtype) + u) / self.quantization_bins
        ldj = self._ldj(z.shape)
        return z, ldj
