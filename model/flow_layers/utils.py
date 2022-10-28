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

class SdnModelScale(nn.Module):
    def __init__(self, param_inits=False):
        super(SdnModelScale, self).__init__()
        self.c_i, self.beta1_i, self.beta2_i, self.gain_params_i, self.cam_params_i = param_inits

        # cam params
        self.n_param_per_cam = 3  # for scaling beta1, beta2, and gain
        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)  # 'IP', 'GP', 'S6', 'N6', 'G4'
        self.cam_params = nn.Parameter(torch.tensor(self.cam_params_i), requires_grad=True)

        # gain params
        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32)
        self.gain_params = nn.Parameter(torch.tensor(self.gain_params_i), requires_grad=True)  # -5.0 / c_i

        self.beta1 = nn.Parameter(torch.tensor(self.beta1_i), requires_grad=True) # -5.0 / c_i
        self.beta2 = nn.Parameter(torch.tensor(self.beta2_i), requires_grad=True) # 0.0

    def forward(self, clean, iso, cam):
        cam_idx = torch.where(self.cam_vals.to(clean.device) == cam[0, 0, 0, 0])
        cam_one_hot = F.one_hot(cam_idx[0], self.cam_vals.shape[0])
        one_cam_params = torch.sum(cam_one_hot * self.cam_params, dim=1)
        one_cam_params = torch.exp(self.c_i * one_cam_params)
        
        iso_idx = torch.where(self.iso_vals.to(clean.device) == iso[0, 0, 0, 0])
        gain_one_hot = F.one_hot(iso_idx[0], self.iso_vals.shape[0])
        g = torch.sum(gain_one_hot * self.gain_params)
        gain = torch.exp(self.c_i * g * one_cam_params[2]) * iso

        
        beta1 = torch.exp(self.c_i * self.beta1 * one_cam_params[0])
        beta2 = torch.exp(self.c_i * self.beta2 * one_cam_params[1])

        scale = beta1 * clean / gain + beta2
        assert torch.min(scale) >= 0
        return torch.sqrt(scale)

        # scale = torch.sqrt(beta1 * clean / gain + beta2)
    
        # return scale

class SdnModelLogScaleExp2(nn.Module):
    def __init__(self, gain_scale, param_inits=False, name='sdn_scale', device='cpu'):
        super(SdnModelLogScaleExp2, self).__init__()
        self.c_i = param_inits['c_i']
        self.beta1_i = param_inits['beta1_i']
        self.beta2_i = param_inits['beta2_i']
        self.name = name

        self._gain_scale = gain_scale(param_inits, name='sdn_layer_gain_scale', device=device)

        self.beta1 = nn.Parameter(torch.tensor(self.beta1_i, device=device), requires_grad=True) # -5.0 / c_i
        self.beta2 = nn.Parameter(torch.tensor(self.beta2_i, device=device), requires_grad=True) # 0.0

    def forward(self, clean_img, iso, cam, writer=None, step=None):
        gain_scale, one_cam_params = self._gain_scale(clean_img, iso, cam, writer, step)

        beta1 = torch.exp(self.c_i * self.beta1 * one_cam_params[:, :, :, :, 0])
        beta2 = torch.exp(self.c_i * self.beta2 * one_cam_params[:, :, :, :, 1])
        scale = beta1 * clean_img / gain_scale + beta2

        # if writer:
        #     writer.add_scalar('model/' + self.name + '_beta1', self.beta1, step)
        #     writer.add_scalar('model/' + self.name + '_beta2', self.beta1, step)
        #     writer.add_scalar('model/' + self.name + '_sdn_scale_mean', torch.mean(scale), step)

        assert torch.min(scale) >= 0
        return 0.5 * torch.log(scale)

class GainScale(nn.Module):
    def __init__(self, param_inits, name="gain_scale", device='cpu'):
        super(GainScale, self).__init__()
        self.c_i = param_inits['c_i']
        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'
        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)
        self.gain_params = nn.Parameter(torch.tensor(param_inits['gain_params_i'], dtype=torch.float32, device=device), requires_grad=True)
        self.cam_params = nn.Parameter(torch.tensor(param_inits['cam_params_i'], dtype=torch.float32, device=device), requires_grad=True)
        self.name = name

    def forward(self, clean_img, iso, cam, writer=None, step=None):
        cam_one_hot = self.cam_vals == cam.unsqueeze(4)
        one_cam_params = torch.sum(cam_one_hot.unsqueeze(4) * self.cam_params, dim=-1)
        one_cam_params = torch.exp(self.c_i * one_cam_params)

        gain_one_hot = self.iso_vals == iso.unsqueeze(4)
        g = torch.sum(gain_one_hot * self.gain_params, axis=4) 
        scale = torch.exp(self.c_i * g *  one_cam_params[:, :, :, :, 2]) * iso

        # if writer:
        #     writer.add_scalar('model/' + self.name + '_gain_params_mean', torch.mean(self.gain_params), step)
        #     writer.add_scalar('model/' + self.name + '_cam_params_mean', torch.mean(self.cam_params), step)
        #     writer.add_scalar('model/' + self.name + '_gain_scale', scale, step)

        return scale, one_cam_params