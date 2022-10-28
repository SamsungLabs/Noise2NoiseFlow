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
DEFAULT_MIN_BIN_WIDTH = 1e-2
DEFAULT_MIN_BIN_HEIGHT = 1e-2
DEFAULT_MIN_DERIVATIVE = 1e-2

class NeuralSpline(nn.Module):
    def __init__(
        self,
        x_shape,
        transform_net,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
        num_bins=10,
        tails="linear",
        tail_bound=1.0,
        name="neural_spline",
        device='cpu'
        ):
        super(NeuralSpline, self).__init__()

        self.ic, self.i0, self.i1 = x_shape
        self.name = name
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self._transform_net = transform_net(
            x_shape[0]  // 2,
            (self.ic - self.ic // 2) * self._transform_dim_multiplier()
        )

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _inverse(self, z, **kwargs):
        identity_split = z[:, :self.ic // 2, ...]
        transform_split = z[:, self.ic // 2:, ...]

        b, c, h, w = transform_split.shape
        transform_params = self._transform_net(identity_split)
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

        transform_split, logabsdet = spline_fn(
            inputs=transform_split,
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

        outputs = torch.cat([identity_split, transform_split], dim=1)
        return outputs

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        identity_split = x[:, :self.ic // 2, ...]
        transform_split = x[:, self.ic // 2:, ...]

        b, c, h, w = transform_split.shape
        transform_params = self._transform_net(identity_split)
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


        transform_split, logabsdet = spline_fn(
            inputs=transform_split,
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

        outputs = torch.cat([identity_split, transform_split], dim=1)
        return outputs, logabsdet

class ConditionalNeuralSpline(nn.Module):
    def __init__(
        self,
        x_shape,
        transform_net,
        encoder,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
        num_bins=10,
        tails="linear",
        tail_bound=1.0,
        name="conditional_neural_spline",
        device='cpu'
        ):
        super(ConditionalNeuralSpline, self).__init__()

        self.ic, self.i0, self.i1 = x_shape
        self.name = name
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self._transform_net = transform_net(
            x_shape[0]  // 2 + x_shape[0],
            (self.ic - self.ic // 2) * self._transform_dim_multiplier()
        )

        self._encoder = encoder(10, 1)

        self.cam_vals = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)  # 'IP', 'GP', 'S6', 'N6', 'G4'
        self.iso_vals = torch.tensor([100, 400, 800, 1600, 3200], dtype=torch.float32, device=device)

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _inverse(self, z, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        gain_one_hot = torch.where(gain_one_hot, 1., 0.)
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam_one_hot = torch.where(cam_one_hot, 1., 0.)
        embedding = self._encoder(torch.cat((gain_one_hot, cam_one_hot), dim=1))
        embedding = embedding.reshape((-1, 1, 1, 1, 1))

        identity_split = z[:, :self.ic // 2, ...]
        transform_split = z[:, self.ic // 2:, ...]

        b, c, h, w = transform_split.shape
        transform_params = self._transform_net(torch.cat((identity_split, kwargs['clean']), dim=1))
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

        unnormalized_widths *= torch.exp(embedding)
        unnormalized_heights *= torch.exp(embedding)
    
        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        transform_split, logabsdet = spline_fn(
            inputs=transform_split,
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

        outputs = torch.cat([identity_split, transform_split], dim=1)
        return outputs

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        gain_one_hot = self.iso_vals == torch.mean(kwargs['iso'], dim=[1, 2, 3]).unsqueeze(1)
        gain_one_hot = torch.where(gain_one_hot, 1., 0.)
        cam_one_hot = self.cam_vals == torch.mean(kwargs['cam'], dim=[1, 2, 3]).unsqueeze(1)
        cam_one_hot = torch.where(cam_one_hot, 1., 0.)
        embedding = self._encoder(torch.cat((gain_one_hot, cam_one_hot), dim=1))
        embedding = embedding.reshape((-1, 1, 1, 1, 1))

        identity_split = x[:, :self.ic // 2, ...]
        transform_split = x[:, self.ic // 2:, ...]

        b, c, h, w = transform_split.shape
        transform_params = self._transform_net(torch.cat((identity_split, kwargs['clean']), dim=1))
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

        unnormalized_widths *= torch.exp(embedding)
        unnormalized_heights *= torch.exp(embedding)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}


        transform_split, logabsdet = spline_fn(
            inputs=transform_split,
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

        outputs = torch.cat([identity_split, transform_split], dim=1)
        return outputs, logabsdet

class TransformNet(nn.Module):
    def __init__(self, x_shape, num_in, num_output, width=4, activation=nn.ReLU(), device='cpu'):
        super(TransformNet, self).__init__()
        self.width = width
        self.activation = activation
        self.n_channels = x_shape[0]
        self.num_output = num_output
        self.num_in = num_in

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
        return x

class ConvResidualBlock(nn.Module):
    def __init__(self,
                 channels,
                 context_channels=None,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 zero_initialization=True):
        super().__init__()
        self.activation = activation

        if context_channels is not None:
            self.context_layer = nn.Conv2d(
                in_channels=context_channels,
                out_channels=channels,
                kernel_size=1,
                padding=0
            )
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm2d(channels, eps=1e-3)
                for _ in range(2)
            ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            for _ in range(2)
        ])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        if context is not None:
            temps = F.glu(
                torch.cat(
                    (temps, self.context_layer(context)),
                    dim=1
                ),
                dim=1
            )
        return inputs + temps


class ConvResidualNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 context_channels=None,
                 num_blocks=2,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False
                 ):
        super().__init__()
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        if context_channels is not None:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels + context_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0
            )
        else:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0
            )
        self.blocks = nn.ModuleList([
            ConvResidualBlock(
                channels=hidden_channels,
                context_channels=context_channels,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ) for _ in range(num_blocks)
        ])
        self.final_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(
                torch.cat((inputs, context), dim=1)
            )
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs

def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        (
            outputs[inside_interval_mask],
            logabsdet[inside_interval_mask],
        ) = rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            d_type=outputs.dtype
        )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
    d_type=torch.float
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # root = (- b + torch.sqrt(discriminant)) / (2 * a)
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs.to(d_type), logabsdet.to(d_type)
    
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

def is_nonnegative_int(x):
    return isinstance(x, int) and x >= 0
