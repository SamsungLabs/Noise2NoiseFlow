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
import numpy as np
import scipy.linalg

class Conv2d1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=True, name='Conv2d1x1'):
        super().__init__()
        self.name = name
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        pixels = list(input.size())[-1]
        if not self.LU:

            #thops.pixels(input)
            dlogdet = (torch.slogdet(self.weight)[1]) * pixels*pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = cpd_sum(self.log_s) * pixels*pixels
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.cpu().double()).float()
                u = torch.inverse(u.cpu().double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.cpu().inverse())).cuda()
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            return z, dlogdet
        else:
            z = F.conv2d(input, weight)
            return z, dlogdet

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        return self.forward(x)

    def _inverse(self, z, **kwargs):
        return self.forward(z, reverse=True)[0]

class ConditionalConv2d1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=True, name='Conv2d1x1', device='cpu'):
        super(ConditionalConv2d1x1, self).__init__()
        self.name = name
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight",nn.Parameter(torch.Tensor(np.expand_dims(torch.Tensor(w_init), axis=0).repeat(25, axis=0))))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np.expand_dims(np_l, axis=0).repeat(25, axis=0).astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np.expand_dims(np_log_s, axis=0).repeat(25, axis=0).astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np.expand_dims(np_u, axis=0).repeat(25, axis=0).astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)
        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'

    def get_weight(self, input, reverse, iso_cam=None):
        w_shape = self.w_shape
        pixels = list(input.size())[-1]
        if not self.LU:

            #thops.pixels(input)
            dlogdet = (torch.slogdet(self.weight[iso_cam])[1]) * pixels*pixels
            if not reverse:
                weight = self.weight[iso_cam].view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight[iso_cam].double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l[iso_cam]
            log_s = self.log_s[iso_cam].squeeze(0)
            l = l * self.l_mask + self.eye
            u = self.u[iso_cam]
            u = u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(log_s))
            dlogdet = cpd_sum(log_s) * pixels*pixels
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.cpu().double()).float()
                u = torch.inverse(u.cpu().double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.cpu().inverse())).cuda()
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, reverse=False, iso_cam=None):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse, iso_cam)
        if not reverse:
            z = F.conv2d(input, weight)
            return z, dlogdet
        else:
            z = F.conv2d(input, weight)
            return z, dlogdet

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[0, 1, 2, 3])
        iso = gain_one_hot.nonzero()
        cam_one_hot = self.cam_vals == (torch.mean(kwargs['cam'], dim=[0, 1, 2, 3]) * 10).round() / (10)
        cam = cam_one_hot.nonzero()
        iso_cam = iso * 5 + cam
        iso_cam = torch.arange(0, 25).cuda() == iso_cam.squeeze(1)

        return self.forward(x, iso_cam=iso_cam)

    def _inverse(self, z, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[0, 1, 2, 3])
        iso = gain_one_hot.nonzero()
        cam_one_hot = self.cam_vals == (torch.mean(kwargs['cam'], dim=[0, 1, 2, 3]) * 10).round() / (10)
        cam = cam_one_hot.nonzero()
        iso_cam = iso * 5 + cam
        iso_cam = torch.arange(0, 25).cuda() == iso_cam.squeeze(1)
        return self.forward(z, reverse=True, iso_cam=iso_cam)[0]

def cpd_sum(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor