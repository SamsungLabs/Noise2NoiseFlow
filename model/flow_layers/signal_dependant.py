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
from model.flow_layers.neural_spline import (unconstrained_rational_quadratic_spline, \
                                            rational_quadratic_spline, sum_except_batch)


class SignalDependant(nn.Module):
    def __init__(self, scale, param_inits=False, name='sdn'):
        super(SignalDependant, self).__init__()
        self.name = name
        self.param_inits = param_inits
        self._scale = scale(self.param_inits)

    def _inverse(self, z, **kwargs):
        scale = self._scale(kwargs['clean'], kwargs['iso'], kwargs['cam'])
        x = z * scale
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        scale = self._scale(kwargs['clean'], kwargs['iso'], kwargs['cam'])

        if 'writer' in kwargs.keys():
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_mean', torch.mean(scale), kwargs['step'])
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_min', torch.min(scale), kwargs['step'])
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_max', torch.max(scale), kwargs['step'])

        z = x / scale
        log_abs_det_J_inv = - torch.sum(torch.log(scale), dim=[1, 2, 3])
        return z, log_abs_det_J_inv

class SignalDependantExp2(nn.Module):
    def __init__(self, log_scale, gain_scale, param_inits=False, device='cpu', name='sdn'):
        super(SignalDependantExp2, self).__init__()
        self.name = name
        self.param_inits = param_inits
        self._log_scale = log_scale(gain_scale, self.param_inits, device=device, name='sdn_layer_gain_scale')

    def _inverse(self, z, **kwargs):
        log_scale = self._log_scale(kwargs['clean'], kwargs['iso'], kwargs['cam'])
        x = z * torch.exp(log_scale)
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        writer = kwargs['writer'] if 'writer' in kwargs.keys() else None
        step = kwargs['step'] if 'step' in kwargs.keys() else None
        
        log_scale = self._log_scale(kwargs['clean'], kwargs['iso'], kwargs['cam'], writer, step)

        if 'writer' in kwargs.keys():
            writer.add_scalar('model/' + self.name + '_log_scale_mean', torch.mean(log_scale), step)
            writer.add_scalar('model/' + self.name + '_log_scale_min', torch.min(log_scale), step)
            writer.add_scalar('model/' + self.name + '_log_scale_max', torch.max(log_scale), step)

        z = x / torch.exp(log_scale)
        log_abs_det_J_inv = - torch.sum(log_scale, dim=[1, 2, 3])
        return z, log_abs_det_J_inv


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

class SignalDependantNS(nn.Module):
    def __init__(
        self,
        transform_net,
        x_shape,
        param_inits=False,
        num_bins=10,
        tails="linear",
        tail_bound=1.0,
        name='sdn',
        device='cpu',
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
        ):
        super(SignalDependantNS, self).__init__()
        self.name = name
        self.ic, self.i0, self.i1 = x_shape
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        
        # self._transform_net = transform_net(
        #     x_shape=x_shape,
        #     width=16,
        #     num_in=x_shape[0],
        #     num_output=x_shape[0] * self._transform_dim_multiplier(),
        #     device=device
        # )

        self._transform_net = transform_net(
            x_shape[0],
            x_shape[0] * self._transform_dim_multiplier()
        )

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _inverse(self, z, **kwargs):
        b, c, h, w = z.shape
        transform_params = self._transform_net(kwargs['clean'])
        transform_params = transform_params.reshape(b, c, -1, h, w).permute(
                0, 1, 3, 4, 2
            )
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self._transform_net, 'width'):
            unnormalized_widths /= np.sqrt(self._transform_net.width)
            unnormalized_heights /= np.sqrt(self._transform_net.width)
        elif hasattr(self._transform_net, 'hidden_channels'):
            unnormalized_widths /= np.sqrt(self._transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self._transform_net.hidden_channels)
        else:
            warnings.warn('Inputs to the softmax are not scaled down: initialization might be bad.')

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        x, logabsdet = spline_fn(
            inputs=z,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=True,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        logabsdet = sum_except_batch(logabsdet)

        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        b, c, h, w = x.shape
        transform_params = self._transform_net(kwargs['clean'])
        transform_params = transform_params.reshape(b, c, -1, h, w).permute(
                0, 1, 3, 4, 2
            )
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self._transform_net, 'width'):
            unnormalized_widths /= np.sqrt(self._transform_net.width)
            unnormalized_heights /= np.sqrt(self._transform_net.width)
        elif hasattr(self._transform_net, 'hidden_channels'):
            unnormalized_widths /= np.sqrt(self._transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self._transform_net.hidden_channels)
        else:
            warnings.warn('Inputs to the softmax are not scaled down: initialization might be bad.')

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        z, logabsdet = spline_fn(
            inputs=x,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=False,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        logabsdet = sum_except_batch(logabsdet)

        return z, logabsdet