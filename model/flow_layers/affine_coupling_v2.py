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


# --- Real-NVP ---
class AffineCoupling(nn.Module):
    def __init__(self, x_shape, shift_and_log_scale, name="real_nvp"):
        super(AffineCoupling, self).__init__()
        self.x_shape = x_shape
        self.ic, self.i0, self.i1 = x_shape
        self._shift_and_log_scale = shift_and_log_scale(x_shape=x_shape)
        self.scale = nn.Parameter(torch.full((1,), 1e-4), requires_grad=True)
        self.name = name

    def _inverse(self, z, **kwargs):
        z0 = z[:, :self.ic // 2, :, :]
        z1 = z[:, self.ic // 2:, :, :]
        shift, log_scale = self._shift_and_log_scale(z0)
        log_scale = self.scale * torch.tanh(log_scale)
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
        log_scale = self.scale * torch.tanh(log_scale)

        if 'writer' in kwargs.keys():
            writer.add_scalar('model/' + self.name + '_log_scale_mean', torch.mean(log_scale), step)
            writer.add_scalar('model/' + self.name + '_log_scale_min', torch.min(log_scale), step)
            writer.add_scalar('model/' + self.name + '_log_scale_max', torch.max(log_scale), step)

            writer.add_scalar('model/' + self.name + '_rescaling_scale_mean', torch.mean(self.scale), step)
            writer.add_scalar('model/' + self.name + '_rescaling_scale_min', torch.min(self.scale), step)
            writer.add_scalar('model/' + self.name + '_rescaling_scale_max', torch.max(self.scale), step)

        z1 = x1 * torch.exp(log_scale) + shift
        z = torch.cat([x0, z1], dim=1)
        log_abs_det_J_inv = torch.sum(log_scale, dim=[1, 2, 3])
        return z, log_abs_det_J_inv

class ShiftAndLogScale(nn.Module):
    def __init__(self, x_shape, width=4, shift_only=False, activation=nn.ReLU()):
        super(ShiftAndLogScale, self).__init__()
        self.width = width
        self.shift_only = shift_only
        self.activation = activation
        self.n_channels = x_shape[0]
        self.num_output = (1 if self.shift_only else 2) *  (self.n_channels  - self.n_channels // 2)
        self.num_in = self.n_channels  // 2

        # TODO squential
        self.conv2d_1 = Conv2d(self.num_in, self.width, name='conv2d_1')
        self.batch_norm_1 = BatchNorm(n_channels=self.width, name='bn_nvp_conv_1')
        self.conv2d_2 = Conv2d(self.width, self.width, filter_size=[1, 1], pad="VALID", name='conv2d_2')
        self.batch_norm_2 = BatchNorm(n_channels=self.width, name='bn_nvp_conv_2')
        self.conv2d_zeros = Conv2dZero( self.width, self.num_output, name='conv2d_zeros')

    def forward(self, x, writer=None, step=None):
        x = self.conv2d_1(x, writer, step)
        x = self.batch_norm_1(x, writer, step)
        x = self.activation(x)

        x = self.conv2d_2(x, writer, step)
        x = self.batch_norm_2(x, writer, step)
        x = self.activation(x)

        x = self.conv2d_zeros(x, writer, step)
        if self.shift_only:
            return x, torch.zeros(x.shape)
        shift, log_scale = torch.split(x, 2, dim=1)

        return shift, log_scale

class Conv2d(nn.Module):
    def __init__(self, n_in, width, filter_size=[3, 3], stride=[1, 1], pad="SAME", name="conv2d"):
        super(Conv2d, self).__init__()
        self.width = width
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.n_in = n_in
        self.filter_shape = [self.width, self.n_in] + self.filter_size
        self.name = name

       
        self.w = nn.Parameter(nn.init.normal_(torch.empty(self.filter_shape), mean=0.0, std=self.width / 512 * 0.05), requires_grad=True)
        self.b = nn.Parameter(torch.zeros([self.width]), requires_grad=True)

    def forward(self, x, writer=None, step=None):
        x = F.conv2d(x, self.w, bias=self.b, stride=self.stride, padding=1 if self.pad == 'SAME' else 0)

        # if writer:
        #     if 'conv2d_1' == self.name:
        #         prefix = 'model/real_nvp_conv_template/conv2d_1/'
        #     else:
        #         prefix = 'model/real_nvp_conv_template/conv2d_2/'
        #     writer.add_scalar(prefix + self.name + '_w_mean', torch.mean(self.w).item(), step)
        #     writer.add_scalar(prefix + self.name + '_b_mean', torch.mean(self.b).item(), step)

        return x

class Conv2dZero(nn.Module):
    def __init__(self, n_in, width, filter_size=[3, 3], stride=[1, 1], pad="SAME", logscale_factor=3, edge_bias=True, name='conv2dzero'):
        super(Conv2dZero, self).__init__()
        self.width = width
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.edge_bias = edge_bias
        self.logscale_factor = logscale_factor
        self.n_in = n_in + 1 if self.edge_bias and self.pad == "SAME" else n_in
        self.filter_shape = [self.width, self.n_in] + self.filter_size
        self.name = name

        self.w = nn.Parameter(torch.zeros(self.filter_shape), requires_grad=True)
        self.b = nn.Parameter(torch.zeros([self.width]), requires_grad=True)
        self.logs = nn.Parameter(torch.zeros([1, self.width, 1, 1]), requires_grad=True)

    def forward(self, x, writer=None, step=None):
        pad = self.pad
        if self.edge_bias and self.pad == "SAME":
            x = add_edge_padding(x, self.filter_size)
            pad = 'VALID'
        x = F.conv2d(x, self.w, bias=self.b, stride=self.stride, padding=1 if pad == 'SAME' else 0)
        x *= torch.exp(self.logs * self.logscale_factor)

        # if writer:
        #     writer.add_scalar('model/real_nvp_conv_template/conv2d_zeros/' + self.name + '_w_mean', torch.mean(self.w).item(), step)
        #     writer.add_scalar('model/real_nvp_conv_template/conv2d_zeros/' + self.name + '_b_mean', torch.mean(self.b).item(), step)
        #     writer.add_scalar('model/real_nvp_conv_template/conv2d_zeros/' + self.name + '_logs_mean', torch.mean(self.logs).item(), step)
        return x

class BatchNorm(nn.Module):
    def __init__(self, n_channels, eps=1e-4, decay=0.1, name='batch_norm'):
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.decay = decay
        # TODO train_m and train_v need to be shared?
        self.train_m = torch.zeros([n_channels], requires_grad=False)
        self.train_v = torch.ones([n_channels], requires_grad=False)
        self.name = name

    def forward(self, x, writer=None, step=None):
        x_shape = x.shape
        assert len(x_shape) == 2 or len(x_shape) == 4
        if self.training:
            if self.train_m.device != x.device:
                self.train_m = self.train_m.to(x.device)
                self.train_v = self.train_v.to(x.device)
            # statistics of current minibatch
            if len(x_shape) == 2:
                m = torch.mean(x, dim=[0])
                v = torch.var(x, dim=[0], unbiased=False)
            elif len(x_shape) == 4:
                m = torch.mean(x, dim=[0, 2, 3])
                v = torch.var(x, dim=[0, 2, 3], unbiased=False)
            self.train_m -= self.decay * (self.train_m.to(x.device) - m)
            self.train_v -= self.decay * (self.train_v.to(x.device) - v)
            # normalize using current minibatch statistics
            x_hat = (x - m.reshape((1, -1, 1, 1))) / torch.sqrt(v + self.eps).reshape((1, -1, 1, 1))
            # if writer:
            #     if 'bn_nvp_conv_2' == self.name:
            #         prefix = 'model/real_nvp_conv_template/cond_1/bn_nvp_conv_2/'
            #     else:
            #         prefix = 'model/real_nvp_conv_template/cond/bn_nvp_conv_1/'
            #     writer.add_scalar(prefix + self.name + '_mean_mean', torch.mean(m).item(), step)
            #     writer.add_scalar(prefix + self.name + '_var_mean', torch.mean(v).item(), step)
        else:
            x_hat = (x - self.train_m.to(x.device).reshape((1, -1, 1, 1))) / torch.sqrt(self.train_v.to(x.device) + self.eps).reshape((1, -1, 1, 1))
        return x_hat

def add_edge_padding(x, filter_size):
    """Slow way to add edge padding"""
    assert filter_size[0] % 2 == 1
    if filter_size[0] == 1 and filter_size[2] == 1:
        return x
    a = (filter_size[0] - 1) // 2  # vertical padding size
    b = (filter_size[1] - 1) // 2  # horizontal padding size
    x = F.pad(x, (b, b, a, a, 0, 0, 0, 0))
    pad = torch.zeros((1,) + (1,) + x.shape[2:4], dtype=torch.float32, device=x.device)
    pad[:, 0, :a, :] = 1.
    pad[:, 0, -a:, :] = 1.
    pad[:, 0, :, :b] = 1.
    pad[:, 0, :, -b:] = 1.
    pad = torch.repeat_interleave(pad, x.shape[0], dim=0)
    x = torch.cat((x, pad), axis=1)
    return x
