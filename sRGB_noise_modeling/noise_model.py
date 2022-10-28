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
import math

import sys
sys.path.append('../')
from model.flow_layers.conv2d1x1 import Conv2d1x1, ConditionalConv2d1x1
from model.flow_layers.affine_coupling import (ConditionalAffineCoupling, AffineCoupling, ShiftAndLogScale, ResidualNet,
                                              ConditionalAffine)
from model.flow_layers.neural_spline import ConditionalNeuralSpline, NeuralSpline, TransformNet, ConvResidualNet
from model.flow_layers.signal_dependant import SignalDependantExp2, SignalDependantNS
from model.flow_layers.gain import Gain, GainExp2
from model.flow_layers.uniform_dequantization import UniformDequantization
from model.flow_layers.logit import Logit
from model.flow_layers.noise_extraction import NoiseExtraction
from model.flow_layers.utils import SdnModelLogScaleExp2, GainScale
from model.flow_layers.linear_transformation import ConditionalLinear, ConditionalLinearExp2, Gamma
from model.flow_layers.actnorm import ActNorm
from model.flow_layers.squeeze import SqueezeLayer, squeeze2d

class NoiseModel(nn.Module):
    def __init__(self, x_shape, arch, flow_permutation, param_inits, lu_decomp, device, is_raw=True):
        super(NoiseModel, self).__init__()
        self.arch = arch
        self.flow_permutation = flow_permutation
        self.param_inits = param_inits
        self.decomp = lu_decomp
        self.device = device
        self.is_raw = is_raw
        self.current_shape = x_shape
        self.squeeze_level = 1
        self.model = nn.ModuleList(self.noise_model_arch())
        self.base_dist = StandardNormal(self.current_shape)
        if not self.is_raw:
            self.data_pre_processing = nn.ModuleList(self.pre_processing_flow())

    def pre_processing_flow(self):
        bijectors = [
            UniformDequantization(
                name='und_clean_img_pre_processing',
                device=self.device
            ),
            # Logit(
            #     name='lgt_clean_img_pre_processing',
            #     device=self.device
            # )
            # Gamma(
            #     name='gamma_clean_img_pre_processing',
            #     device=self.device
            # )
        ]

        return bijectors

    def noise_model_arch(self,):
        arch_lyrs = self.arch.split('|')  # e.g., AC|sdn|AC|gain|AC
        bijectors = []

        if not self.is_raw:
            print('|-UniformDequantization')
            bijectors.append(
                UniformDequantization(
                    name='und',
                    device=self.device
                )
            )

            # print('|-Gamma')
            # bijectors.append(
            #     Gamma(
            #         name='gamma',
            #         device=self.device
            #     )
            # )
            # print('|-Logit')
            # bijectors.append(
            #     Logit(
            #         name='lgt',
            #         device=self.device
            #     )
            # )

            print('|-NoiseExtraction')
            bijectors.append(
                NoiseExtraction(
                    name='ne',
                    device=self.device
                )
            )

        for i, lyr in enumerate(arch_lyrs):
            if lyr == 'CL':
                print('|-ConditionalLinear')
                bijectors.append(
                    ConditionalLinear(
                        name='CL_{}'.format(i),
                        device=self.device
                    )
                )
            elif lyr == 'CL2':
                print('|-ConditionalLinearExp2')
                bijectors.append(
                    ConditionalLinearExp2(
                        name='CL2_{}'.format(i),
                        device=self.device
                    )
                )
            elif lyr == 'C1x1':
                print('|-Conv2d1x1')
                bijectors.append(
                    Conv2d1x1(
                        num_channels=self.current_shape[0],
                        LU_decomposed=self.decomp,
                        name='Conv2d_1x1_{}'.format(i)
                    )
                )
            elif lyr == 'cond_c1x1':
                print('|-ConditionalConv2d1x1')
                bijectors.append(
                    ConditionalConv2d1x1(
                        num_channels=self.current_shape[0],
                        LU_decomposed=self.decomp,
                        name='Conv2d_1x1_{}'.format(i),
                        device=self.device
                    )
                )
            elif lyr == 'SQZ':
                print('|-Squeeze')
                bijectors.append(
                    SqueezeLayer(
                        factor=2,
                        name='SQZ_{}'.format(i),
                        level=self.squeeze_level
                    )
                )
                self.squeeze_level += 1
                self.current_shape = (self.current_shape[0] * 4,
                                 self.current_shape[1] // 2,
                                 self.current_shape[2] // 2)
            elif lyr == 'CA_Icg':
                print('|-ConditionalAffine')
                bijectors.append(
                    ConditionalAffine(
                        x_shape=self.current_shape,
                        shift_and_log_scale=ShiftAndLogScale,
                        name='ca_%d' % i,
                        encoder=lambda in_features, out_features: ResidualNet(
                            in_features=in_features,
                            out_features=out_features,
                            hidden_features=5,
                            num_blocks=3,
                            use_batch_norm=True,
                            dropout_probability=0.0
                        ),
                        device=self.device,
                        only_clean=False
                    )
                )
            elif lyr == 'CA_I':
                print('|-ConditionalAffine')
                bijectors.append(
                    ConditionalAffine(
                        x_shape=self.current_shape,
                        shift_and_log_scale=ShiftAndLogScale,
                        name='ca_%d' % i,
                        encoder=lambda in_features, out_features: ResidualNet(
                            in_features=in_features,
                            out_features=out_features,
                            hidden_features=5,
                            num_blocks=3,
                            use_batch_norm=True,
                            dropout_probability=0.0
                        ),
                        device=self.device,
                        only_clean=True
                    )
                )
            elif lyr == 'AC' or lyr =='CAC':
                if self.flow_permutation == 0:
                    # TODO impoliment permute
                    pass
                elif self.flow_permutation == 1:
                    print('|-Conv2d1x1')
                    bijectors.append(
                        Conv2d1x1(
                            num_channels=self.current_shape[0],
                            LU_decomposed=self.decomp,
                            name='Conv2d_1x1_{}'.format(i)
                        )
                    )
                elif self.flow_permutation == 2:
                    print('|-ConditionalConv2d1x1')
                    bijectors.append(
                        ConditionalConv2d1x1(
                            num_channels=self.current_shape[0],
                            LU_decomposed=self.decomp,
                            name='Conv2d_1x1_{}'.format(i),
                            device=self.device
                        )
                    )
                else:
                    print('|-No permutation specified. Not using any.')

                if lyr == 'AC':
                    print('|-AffineCoupling')
                    bijectors.append(
                        AffineCoupling(
                            x_shape=self.current_shape,
                            shift_and_log_scale=ShiftAndLogScale,
                            name='AC_%d' % i,
                            device=self.device
                        )
                    )
                else:
                    print('|-ConditionalAffineCoupling')
                    bijectors.append(
                        ConditionalAffineCoupling(
                            x_shape=self.current_shape,
                            shift_and_log_scale=ShiftAndLogScale,
                            name='CAC_%d' % i,
                            encoder=lambda in_features, out_features: ResidualNet(
                                in_features=in_features,
                                out_features=out_features,
                                hidden_features=5,
                                num_blocks=3,
                                use_batch_norm=True,
                                dropout_probability=0.0
                            ),
                            device=self.device

                        )
                    )
            elif lyr =='SC' or lyr == 'CSC':
                # print('|-actnorm')
                # bijectors.append(
                #     ActNorm(
                #         features=self.current_shape[0],
                #         name='actnorm_%d' % i,
                #         device=self.device
                #     )
                # )
                if self.flow_permutation == 0:
                    # TODO impoliment permute
                    pass
                elif self.flow_permutation == 1:
                    print('|-Conv2d1x1')
                    bijectors.append(
                        Conv2d1x1(
                            num_channels=self.current_shape[0],
                            LU_decomposed=self.decomp,
                            name='Conv2d_1x1_{}'.format(i)
                        )
                    )
                else:
                    print('|-No permutation specified. Not using any.')

                if lyr =='SC':
                    print('|-SplineCoupling')
                    bijectors.append(
                        NeuralSpline(
                            x_shape=self.current_shape,
                            # transform_net=TransformNet,
                            transform_net = lambda in_channels, out_channels: ConvResidualNet(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                hidden_channels=96,
                                num_blocks=3,
                                use_batch_norm=True,
                                dropout_probability=0.0
                            ),
                            num_bins=4,
                            tail_bound=3,
                            name='SC%d' % i,
                            device=self.device
                        )
                    )
                else:
                    print('|-ConditionalSplineCoupling')
                    bijectors.append(
                        ConditionalNeuralSpline(
                            x_shape=self.current_shape,
                            # transform_net=TransformNet,
                            transform_net = lambda in_channels, out_channels: ConvResidualNet(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                hidden_channels=96,
                                num_blocks=3,
                                use_batch_norm=True,
                                dropout_probability=0.0
                            ),
                            encoder=lambda in_features, out_features: ResidualNet(
                                in_features=in_features,
                                out_features=out_features,
                                hidden_features=5,
                                num_blocks=3,
                                use_batch_norm=True,
                                dropout_probability=0.0
                            ),
                            num_bins=4,
                            tail_bound=3,
                            name='conditional_sc_%d' % i,
                            device=self.device
                        )
                    )
            elif lyr == 'SDN':
                print('|-SignalDependant')
                bijectors.append(
                    SignalDependantExp2(
                        name='sdn_%d' % i,
                        log_scale=SdnModelLogScaleExp2,
                        param_inits=self.param_inits,
                        gain_scale=GainScale,
                        device=self.device
                    )
                )
            elif lyr == 'sdn_ns':
                print('|-SignalDependantNS')
                bijectors.append(
                    SignalDependantNS(
                        x_shape=self.current_shape,
                        name='sdn_ns_%d' % i,
                        num_bins=4,
                        tail_bound=3,
                        # transform_net=TransformNet,
                        transform_net = lambda in_channels, out_channels: ConvResidualNet(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            hidden_channels=96,
                            num_blocks=3,
                            use_batch_norm=True,
                            dropout_probability=0.0
                        ),
                        device=self.device
                    )
                )
            elif lyr == 'Gain2':
                print('|-GainExp2')
                bijectors.append(
                    GainExp2(
                        name='gain_exp2_%d' % i,
                        gain_scale=GainScale,
                        param_inits=self.param_inits,
                        device=self.device
                    )
                )
            elif lyr == 'gain':
                print('|-Gain')
                bijectors.append(
                    Gain(
                        name='gain_%d' % i,
                        device=self.device
                    )
                )

        return bijectors

    def forward(self, x, **kwargs):
        z = x
        clean = kwargs['clean']
        objective = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)

        for bijector in self.model:
            z, log_abs_det_J_inv = bijector._forward_and_log_det_jacobian(z, **kwargs)
            objective += log_abs_det_J_inv

            if isinstance(bijector, SqueezeLayer):
                clean2 = clean
                for i in range(bijector.level):
                    clean2 = squeeze2d(clean2)
                    kwargs['clean']= clean2

            if 'writer' in kwargs.keys():
                kwargs['writer'].add_scalar('model/' + bijector.name, torch.mean(log_abs_det_J_inv), kwargs['step'])
        return z, objective

    def _loss(self, x, **kwargs):
        z, objective = self.forward(x, **kwargs)
        # base measure
        log_z = self.base_dist.log_prob(z)
        # logp, _ = self.prior("prior", x)

        # log_z = logp(z)
        objective += log_z

        if 'writer' in kwargs.keys():
            kwargs['writer'].add_scalar('model/log_z', torch.mean(log_z), kwargs['step'])
            kwargs['writer'].add_scalar('model/z', torch.mean(z), kwargs['step'])
        nobj = - objective
        # std. dev. of z
        mu_z = torch.mean(z, dim=[1, 2, 3])
        var_z = torch.var(z, dim=[1, 2, 3])
        sd_z = torch.mean(torch.sqrt(var_z))

        return nobj, sd_z

    def loss(self, x, **kwargs):
        # batch_average = torch.mean(x, dim=0)
        # if 'writer' in kwargs.keys():
        #     kwargs['writer'].add_histogram('real_noise', batch_average, kwargs['step'])
        #     kwargs['writer'].add_scalar('real_noise_std', torch.std(batch_average), kwargs['step'])

        if not self.is_raw:
            for bijector in self.data_pre_processing:
                kwargs['clean'], _ = bijector._forward_and_log_det_jacobian(kwargs['clean'])

        nll, sd_z = self._loss(x=x, **kwargs)

        return torch.mean(nll), sd_z

    def inverse(self, z, **kwargs):
        x = z
        clean = kwargs['clean']
        for bijector in reversed(self.model):
            clean2 = clean
            for i in range(int(np.log2(32/x.shape[-1]))):
                clean2 = squeeze2d(clean2)
            kwargs['clean'] = clean2

            x = bijector._inverse(x, **kwargs)
            

        return x
    
    def sample(self, eps_std=None, **kwargs):
        if not self.is_raw:
            for bijector in self.data_pre_processing:
                kwargs['clean'], _ = bijector._forward_and_log_det_jacobian(kwargs['clean'])

        # _, sample = self.prior("prior", kwargs['clean'])
        # z = sample(eps_std)
        z = self.base_dist.sample(num_samples=kwargs['clean'].shape[0])
        x = self.inverse(z, **kwargs)
        batch_average = torch.mean(x, dim=0)
        if 'writer' in kwargs.keys():
            kwargs['writer'].add_histogram('sample_noise', batch_average, kwargs['step'])
            kwargs['writer'].add_scalar('sample_noise_std', torch.std(batch_average), kwargs['step'])

        return x

    def get_layer_names(self):
        layer_names = []
        for b in self.model:
            layer_names.append(b.name)
        return layer_names

class StandardNormal(nn.Module):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super(StandardNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x):
        log_base =  - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * x**2
        return sum_except_batch(log_base+log_inner)

    def sample(self, num_samples):
        return torch.randn(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype)

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)