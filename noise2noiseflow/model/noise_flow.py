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

from model.flow_layers.conv2d1x1 import Conv2d1x1
from model.flow_layers.affine_coupling import AffineCoupling, ShiftAndLogScale
from model.flow_layers.signal_dependant import SignalDependant
from model.flow_layers.gain import Gain
from model.flow_layers.utils import SdnModelScale
# from model.flow_layers.linear_transformation import LinearTransformation

class NoiseFlow(nn.Module):

    def __init__(self, x_shape, arch, flow_permutation, param_inits, lu_decomp):
        super(NoiseFlow, self).__init__()
        self.arch = arch
        self.flow_permutation = flow_permutation
        self.param_inits = param_inits
        self.decomp = lu_decomp
        self.model = nn.ModuleList(self.noise_flow_arch(x_shape))

    def noise_flow_arch(self, x_shape):
        arch_lyrs = self.arch.split('|')  # e.g., unc|sdn|unc|gain|unc
        bijectors = []
        for i, lyr in enumerate(arch_lyrs):
            is_last_layer = False

            if lyr == 'unc':
                if self.flow_permutation == 0:
                    pass
                elif self.flow_permutation == 1:
                    print('|-Conv2d1x1')
                    bijectors.append(
                        Conv2d1x1(
                            num_channels=x_shape[0],
                            LU_decomposed=self.decomp,
                            name='Conv2d_1x1_{}'.format(i)
                        )
                    )
                else:
                    print('|-No permutation specified. Not using any.')
                    # raise Exception("Flow permutation not understood")

                print('|-AffineCoupling')
                bijectors.append(
                    AffineCoupling(
                        x_shape=x_shape,
                        shift_and_log_scale=ShiftAndLogScale,
                        name='unc_%d' % i
                    )
                )
            # elif lyr == 'lt':
            #     print('|-LinearTransfomation')
            #     bijectors.append(
            #         LinearTransformation(
            #             name='lt_{}'.format(i),
            #             device='cuda'
            #         )
            #     )
            elif lyr == 'sdn':
                print('|-SignalDependant')
                bijectors.append(
                    SignalDependant(
                        name='sdn_%d' % i,
                        scale=SdnModelScale,
                        param_inits=self.param_inits
                    )
                )
            elif lyr == 'gain':
                print('|-Gain')
                bijectors.append(
                    Gain(name='gain_%d' % i)
                )

        return bijectors

    def forward(self, x, **kwargs):
        z = x
        objective = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
        for bijector in self.model:
            z, log_abs_det_J_inv = bijector._forward_and_log_det_jacobian(z, **kwargs)
            objective += log_abs_det_J_inv

            if 'writer' in kwargs.keys():
                kwargs['writer'].add_scalar('model/' + bijector.name, torch.mean(log_abs_det_J_inv), kwargs['step'])
        return z, objective

    def _loss(self, x, **kwargs):
        z, objective = self.forward(x, **kwargs)
        # base measure
        logp, _ = self.prior("prior", x)

        log_z = logp(z)
        objective += log_z

        if 'writer' in kwargs.keys():
            kwargs['writer'].add_scalar('model/log_z', torch.mean(log_z), kwargs['step'])
            kwargs['writer'].add_scalar('model/z', torch.mean(z), kwargs['step'])
        nobj = - objective
        # std. dev. of z
        mu_z = torch.mean(x, dim=[1, 2, 3])
        var_z = torch.var(x, dim=[1, 2, 3])
        sd_z = torch.mean(torch.sqrt(var_z))

        return nobj, sd_z

    def loss(self, x, **kwargs):
        batch_average = torch.mean(x, dim=0)
        # if 'writer' in kwargs.keys():
        #     kwargs['writer'].add_histogram('real_noise', batch_average, kwargs['step'])
        #     kwargs['writer'].add_scalar('real_noise_std', torch.std(batch_average), kwargs['step'])

        nll, sd_z = self._loss(x=x, **kwargs)
        nll_dim = torch.mean(nll) / np.prod(x.shape[1:])
        # nll_dim = torch.mean(nll)      # The above line should be uncommented

        return nll_dim, sd_z

    def inverse(self, z, **kwargs):
        x = z
        for bijector in reversed(self.model):
            x = bijector._inverse(x, **kwargs)
        return x
    
    def sample(self, eps_std=None, **kwargs):
        _, sample = self.prior("prior", kwargs['clean'])
        z = sample(eps_std)
        x = self.inverse(z, **kwargs)
        batch_average = torch.mean(x, dim=0)
        if 'writer' in kwargs.keys():
            kwargs['writer'].add_histogram('sample_noise', batch_average, kwargs['step'])
            kwargs['writer'].add_scalar('sample_noise_std', torch.std(batch_average), kwargs['step'])

        return x

    def prior(self, name, x):
        n_z = x.shape[1]
        h = torch.zeros([x.shape[0]] +  [2 * n_z] + list(x.shape[2:4]), device=x.device)
        pz = gaussian_diag(h[:, :n_z, :, :], h[:, n_z:, :, :])

        def logp(z1):
            objective = pz.logp(z1)
            return objective

        def sample(eps_std=None):
            if eps_std is not None:
                z = pz.sample2(pz.eps * torch.reshape(eps_std, [-1, 1, 1, 1]))
            else:
                z = pz.sample
            return z

        return logp, sample

def gaussian_diag(mean, logsd):
    class o(object):
        pass

    o.mean = mean
    o.logsd = logsd
    o.eps = torch.normal(torch.zeros(mean.shape, device=mean.device), torch.ones(mean.shape, device=mean.device))
    o.sample = mean + torch.exp(logsd) * o.eps
    o.sample2 = lambda eps: mean + torch.exp(logsd) * eps

    o.logps = lambda x: -0.5 * (np.log(2 * np.pi) + 2. * o.logsd + (x - o.mean) ** 2 / torch.exp(2. * o.logsd))
    o.logp = lambda x: torch.sum(o.logps(x), dim=[1, 2, 3])
    o.get_eps = lambda x: (x - mean) / torch.exp(logsd)
    return o
