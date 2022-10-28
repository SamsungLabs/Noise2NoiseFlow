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


# --- Real-NVP ---
class AffineCoupling(nn.Module):
    def __init__(self, x_shape, shift_and_log_scale, name="real_nvp", device='cuda'):
        super(AffineCoupling, self).__init__()
        self.x_shape = x_shape
        self.ic, self.i0, self.i1 = x_shape
        self._shift_and_log_scale = shift_and_log_scale(num_in=self.ic // 2, num_out=2*(self.ic  - self.ic // 2), device=device)
        self.name = name

    def _inverse(self, z, **kwargs):
        z0 = z[:, :self.ic // 2, :, :]
        z1 = z[:, self.ic // 2:, :, :]
        shift, log_scale = self._shift_and_log_scale(z0)
        x1 = z1
        x1 = (z1 - shift) * torch.exp(-log_scale)
        x = torch.cat([z0, x1], dim=1)
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        x0 = x[:, :self.ic // 2, :, :]
        x1 = x[:, self.ic // 2:, :, :]

        writer = kwargs['writer'] if 'writer' in kwargs.keys() else None
        step = kwargs['step'] if 'step' in kwargs.keys() else None

        shift, log_scale = self._shift_and_log_scale(x0, writer, step)

        if 'writer' in kwargs.keys():
            writer.add_scalar('model/' + self.name + '_log_scale_mean', torch.mean(log_scale), step)
            writer.add_scalar('model/' + self.name + '_log_scale_min', torch.min(log_scale), step)
            writer.add_scalar('model/' + self.name + '_log_scale_max', torch.max(log_scale), step)

        z1 = x1 * torch.exp(log_scale) + shift
        z = torch.cat([x0, z1], dim=1)
        log_abs_det_J_inv = torch.sum(log_scale, dim=[1, 2, 3])
        return z, log_abs_det_J_inv

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, x_shape, shift_and_log_scale, encoder, name="conditional_coupling", device='cpu'):
        super(ConditionalAffineCoupling, self).__init__()
        self.x_shape = x_shape
        self.ic, self.i0, self.i1 = x_shape
        num_out = 2 *  (self.ic  - self.ic // 2)
        self._shift_and_log_scale = shift_and_log_scale(num_in=self.ic // 2 + self.ic, num_out=num_out, device=device)
        self.name = name

        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'
        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)

        self._encoder = encoder(10, 1)

    def _inverse(self, z, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        gain_one_hot = torch.where(gain_one_hot, 1., 0.)
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam_one_hot = torch.where(cam_one_hot, 1., 0.)
        embedding = self._encoder(torch.cat((gain_one_hot, cam_one_hot), dim=1))
        embedding = embedding.reshape((-1, 1, 1, 1))

        z0 = z[:, :self.ic // 2, :, :]
        z1 = z[:, self.ic // 2:, :, :]
        shift, log_scale = self._shift_and_log_scale(torch.cat((z0, kwargs['clean']), dim=1))
        log_scale *= embedding
        x1 = z1
        x1 = (z1 - shift) * torch.exp(-log_scale)
        x = torch.cat([z0, x1], dim=1)
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        gain_one_hot = torch.where(gain_one_hot, 1., 0.)
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam_one_hot = torch.where(cam_one_hot, 1., 0.)
        embedding = self._encoder(torch.cat((gain_one_hot, cam_one_hot), dim=1))
        embedding = embedding.reshape((-1, 1, 1, 1))

        x0 = x[:, :self.ic // 2, :, :]
        x1 = x[:, self.ic // 2:, :, :]
        shift, log_scale = self._shift_and_log_scale(torch.cat((x0, kwargs['clean']), dim=1))
        log_scale *= embedding
        z1 = x1 * torch.exp(log_scale) + shift
        z = torch.cat([x0, z1], dim=1)
        log_abs_det_J_inv = torch.sum(log_scale, dim=[1, 2, 3])
        return z, log_abs_det_J_inv

class ConditionalAffine(nn.Module):
    def __init__(self, x_shape, shift_and_log_scale, encoder, name="conditional_coupling", device='cpu', only_clean=False):
        super(ConditionalAffine, self).__init__()
        self.x_shape = x_shape
        self.ic, self.i0, self.i1 = x_shape
        num_out = 2 * self.ic
        self._shift_and_log_scale = shift_and_log_scale(num_in=self.ic, num_out=num_out, device=device)
        self.name = name
        self.only_clean = only_clean

        if not self.only_clean:
            self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'
            self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)

            self._encoder = encoder(10, 1)

    def _inverse(self, z, **kwargs):
        if not self.only_clean:
            gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
            gain_one_hot = torch.where(gain_one_hot, 1., 0.)
            cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
            cam_one_hot = torch.where(cam_one_hot, 1., 0.)
            embedding = self._encoder(torch.cat((gain_one_hot, cam_one_hot), dim=1))
            embedding = embedding.reshape((-1, 1, 1, 1))

        shift, log_scale = self._shift_and_log_scale(kwargs['clean'])

        if not self.only_clean:
            log_scale *= embedding
 
        x = (z - shift) * torch.exp(-log_scale)
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        if not self.only_clean:
            gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
            gain_one_hot = torch.where(gain_one_hot, 1., 0.)
            cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
            cam_one_hot = torch.where(cam_one_hot, 1., 0.)
            embedding = self._encoder(torch.cat((gain_one_hot, cam_one_hot), dim=1))
            embedding = embedding.reshape((-1, 1, 1, 1))

        shift, log_scale = self._shift_and_log_scale(kwargs['clean'])
        if not self.only_clean:
            log_scale *= embedding
        z = x * torch.exp(log_scale) + shift
        log_abs_det_J_inv = torch.sum(log_scale, dim=[1, 2, 3])
        return z, log_abs_det_J_inv

class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(self,
                 features,
                 context_features,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 zero_initialization=True):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList([
            nn.Linear(features, features)
            for _ in range(2)
        ])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(
                torch.cat(
                    (temps, self.context_layer(context)),
                    dim=1
                ),
                dim=1
            )
        return inputs + temps

class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(in_features + context_features, hidden_features)
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList([
            ResidualBlock(
                features=hidden_features,
                context_features=context_features,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ) for _ in range(num_blocks)
        ])
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(
                torch.cat((inputs, context), dim=1)
            )
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs

class ShiftAndLogScale(nn.Module):
    def __init__(self, num_in, num_out, width=4, shift_only=False, activation=nn.ReLU(), device='cpu'):
        super(ShiftAndLogScale, self).__init__()
        self.width = width
        self.shift_only = shift_only
        self.num_in = num_in
        self.num_output = num_out
        self.scale = nn.Parameter(torch.full((1,), 1e-4, device=device), requires_grad=True)

        self.conv2d_1 = nn.Conv2d(in_channels=self.num_in, out_channels=self.width, kernel_size=3, padding=1)
        nn.init.normal_(self.conv2d_1.weight, mean=0.0, std=self.width / 512 * 0.05)
        self.conv2d_1.bias.data.fill_(0.0)

        self.conv2d_2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, padding=0)
        nn.init.normal_(self.conv2d_2.weight, mean=0.0, std=self.width / 512 * 0.05)
        self.conv2d_2.bias.data.fill_(0.0)

        self.net = nn.Sequential(
            self.conv2d_1,
            nn.BatchNorm2d(num_features=self.width),
            activation,
            self.conv2d_2,
            nn.BatchNorm2d(num_features=self.width),
            activation
        )

        self.padding = nn.ConstantPad3d((1, 1, 1, 1, 0, 1), 0.)
        self.conv2d_3 = nn.Conv2d(in_channels=self.width+1, out_channels=self.num_output, kernel_size=3, padding=0)
        self.conv2d_3.weight.data.fill_(0.0)
        self.conv2d_3.bias.data.fill_(0.0)
        self.logs = nn.Parameter(torch.zeros([1, self.num_output, 1, 1], device=device), requires_grad=True)


    def forward(self, x, writer=None, step=None):
        x = self.net(x)

        x = self.padding(x)
        x[:, 4, :1, :] = 1.0
        x[:, 4, -1:, :] = 1.0
        x[:, 4, :, :1] = 1.0
        x[:, 4, :, -1:] = 1.0
        x = self.conv2d_3(x)
        x *= torch.exp(self.logs * 3)

        if self.shift_only:
            return x, torch.zeros(x.shape)

        shift, log_scale = torch.split(x, int(x.shape[1]/2), dim=1)
        log_scale = self.scale * torch.tanh(log_scale)

        return shift, log_scale
