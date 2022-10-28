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
import numpy as np
import time
import torch
import sys
import os
import logging
from numpy import save, load

class PatchStatsCalculator:

    def __init__(self, dataset, patch_height=256, n_channels=4, save_dir='.', file_postfix=''):
        self.dataset = dataset
        self.patch_height = patch_height
        self.n_channels = n_channels
        self.save_dir = save_dir
        self.file_postfix = file_postfix

        self.n_pat = 0
        self.threads = []
        self.stats = None

        self.init_pat_stats()

    def init_pat_stats(self):
        # initialize patch stats
        self.stats = dict({
            # patch-wise
            'in_mu': torch.zeros((self.n_channels, self.patch_height, self.patch_height)),
            'clean_mu': torch.zeros((self.n_channels, self.patch_height, self.patch_height)),
            'in_vr': torch.zeros((self.n_channels, self.patch_height, self.patch_height)),
            'clean_vr': torch.zeros((self.n_channels, self.patch_height, self.patch_height)),
            'in_sd': torch.zeros((self.n_channels, self.patch_height, self.patch_height)),
            'clean_sd': torch.zeros((self.n_channels, self.patch_height, self.patch_height)),
            'n_pat': 0,
            # scalars
            'sc_in_mu': 0, 'sc_clean_mu': 0, 'sc_in_vr': 0, 'sc_clean_vr': 0, 'sc_in_sd': 0, 'sc_clean_sd': 0, 'n_pix': 0
        })

    def calc_stats(self):
        t0 = time.time()

        self.calc_patch_stats()
        self.calc_scalar_stats()
        save(os.path.join(self.save_dir, 'pat_stats%s.npy') % self.file_postfix, self.stats)
        # logging.trace('calc. stats: time = %3.0f s ' % (time.time() - t0))
        return self.stats


    def calc_patch_stats(self):
        n_pat = 0  # number of patches
        
        for image in self.dataset:
            for idx in range(image['noise'].shape[0]):
                self.stats['in_vr'] = self.online_var_step(self.stats['in_vr'], self.stats['in_mu'], n_pat, image['noise'][idx])
                self.stats['clean_vr'] = self.online_var_step(self.stats['clean_vr'], self.stats['clean_mu'], n_pat, image['clean'][idx])
                self.stats['in_mu'] = self.online_mean_step(self.stats['in_mu'], n_pat, image['noise'][idx])
                self.stats['clean_mu'] = self.online_mean_step(self.stats['clean_mu'], n_pat, image['clean'][idx])
                n_pat += 1

        self.stats['n_pat'] = n_pat

    def calc_scalar_stats(self):
        # scalar mean and scalar sample variance
        k = self.stats['n_pat']
        g = self.patch_height * self.patch_height * self.n_channels
        self.stats['n_pix'] = k * g

        self.stats['sc_in_mu'] = torch.mean(self.stats['in_mu'])
        self.stats['sc_clean_mu'] = torch.mean(self.stats['clean_mu'])

        t_sum = torch.sum(self.stats['in_vr']) + torch.var(self.stats['in_mu']) * (k * (g - 1)) / (k - 1)
        self.stats['sc_in_vr'] = t_sum * (k - 1) / (k * g - 1)
        t_sum = torch.sum(self.stats['clean_vr']) + torch.var(self.stats['clean_mu']) * (k * (g - 1)) / (k - 1)
        self.stats['sc_clean_vr'] = t_sum * (k - 1) / (k * g - 1)

        if self.stats['sc_in_vr'] < sys.float_info.epsilon:
            self.stats['sc_in_vr'] = torch.tensor(sys.float_info.epsilon)
        if self.stats['sc_clean_vr'] < sys.float_info.epsilon:
            self.stats['sc_clean_vr'] = torch.tensor(sys.float_info.epsilon)

        # standard deviation (sd): take square root
        self.stats['in_sd'] = torch.sqrt(self.stats['in_vr'])
        self.stats['clean_sd'] = torch.sqrt(self.stats['clean_vr'])
        self.stats['sc_in_sd'] = torch.sqrt(self.stats['sc_in_vr'])
        self.stats['sc_clean_sd'] = torch.sqrt(self.stats['sc_clean_vr'])

    def calc_baselines(self, test_dataset):
        nll_gauss_lst = []
        nll_sdn_lst = []

        for image in test_dataset:
            x = image['noise']
            y = image['clean']

            vr_gauss = self.stats['sc_in_vr']
            nll_mb_gauss = 0.5 * (torch.log(2 * torch.tensor(np.pi)) + torch.log(vr_gauss) + (x) ** 2 / vr_gauss)
            nll_mb_gauss = torch.sum(nll_mb_gauss, axis=(1, 2, 3))
            nll_gauss_lst.append(np.array(nll_mb_gauss))

            if 'nlf0' in image.keys(): # sRGB vs Raw
                nlf0 = image['nlf0']
                nlf1 = image['nlf1']
                vr = y * nlf0 + nlf1
                nll_mb = 0.5 * (torch.log(2 * torch.tensor(np.pi)) + torch.log(vr) + (x) ** 2 / vr)
                nll_mb = torch.sum(nll_mb, axis=(1, 2, 3))
                nll_sdn_lst.append(np.array(nll_mb))

        nll_sdn = np.mean(nll_sdn_lst) if len(nll_sdn_lst) > 0 else np.nan
        nll_gauss = np.mean(nll_gauss_lst)
        save(os.path.join(self.save_dir, 'nll_bpd_gauss%s.npy') % self.file_postfix, (nll_gauss, 0))
        save(os.path.join(self.save_dir, 'nll_bpd_sdn%s.npy') % self.file_postfix, (nll_sdn, 0))
        
        return nll_gauss, nll_sdn

    @staticmethod
    def online_mean_step(cur_mean, cur_n, new_point):
        if cur_n >= 0:
            return cur_mean + (new_point - cur_mean) / (cur_n + 1)

    @staticmethod
    def online_var_step(cur_var, cur_mean, cur_n, new_point):
        if cur_n >= 1:
            return cur_var + ((new_point - cur_mean) ** 2 / (cur_n + 1)) - (cur_var / cur_n)
        else:
            return cur_var

    