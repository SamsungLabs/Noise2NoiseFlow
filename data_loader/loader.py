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
from data_loader.sidd_utils import (extract_iso_cam, load_one_tuple_images, load_one_tuple_srgb_images,\
                                    sidd_full_filenames_len, sidd_medium_filenames_tuple, get_sidd_filename_tuple,\
                                    extract_patches, extract_nlf)
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from os.path import exists
import os

def check_download_sidd(sidd_path):
    if not exists(sidd_path):
        print(sidd_path + ' does not exist')

class SIDDFullRawDataset(Dataset):
    """SIDD Full Raw dataset."""

    def __init__(self, sidd_full_path, train_or_test='train', cam=None, iso=None, num_patches_per_image=2898,
                 patch_size=(32, 32), patch_sampling='uniform', shuffle_patches=False, subtract_images=False,
                 transform=None):
        self.sidd_full_path = sidd_full_path
        self.train_or_test = train_or_test
        self.cam = cam
        self.iso = iso
        self.num_patches_per_image = num_patches_per_image
        self.patch_size = patch_size
        self.shuffle_patches = shuffle_patches
        self.patch_sampling = patch_sampling
        self.subtract_images = subtract_images
        self.transform = transform

        self.len = sidd_full_filenames_len(
            sidd_full_path=self.sidd_full_path,
            train_or_test=self.train_or_test,
            cam=self.cam,
            iso=self.iso)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name_tuple = get_sidd_filename_tuple(
            idx,
            sidd_full_path=self.sidd_full_path,
            train_or_test=self.train_or_test,
            cam=self.cam,
            iso=self.iso
        )
        img1, img2, nlf0, nlf1, iso, cam = load_one_tuple_images(file_name_tuple, subtract=self.subtract_images)
        img1_patches, img2_patches = extract_patches(
            (img1, img2),
            num_patches=self.num_patches_per_image,
            patch_size=self.patch_size,
            sampling=self.patch_sampling,
            shuffle=self.shuffle_patches
        )
        img1_patches = img1_patches.transpose((0, 3, 1, 2))
        img2_patches = img2_patches.transpose((0, 3, 1, 2))

        sample = {
            'noisy1': torch.from_numpy(img1_patches),
            'noisy2': torch.from_numpy(img2_patches),
            'nlf0': nlf0,
            'nlf1': nlf1,
            'iso': iso,
            'cam' : cam
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class SIDDMediumDataset(Dataset):
    """SIDD Raw/SRGB Medium dataset."""

    def __init__(self, sidd_medium_path, train_or_test='train', cam=None, iso=None, num_patches_per_image=2898,
                 patch_size=(32, 32), patch_sampling='uniform', shuffle_patches=False, subtract_images=True,
                 transform=None, is_raw=True, first_im_idx=10, last_im_idx=12, model=None, temp=None, device=None):
        self.device = device
        self.temp = temp
        self.sidd_medium_path = sidd_medium_path
        self.raw = is_raw
        self.train_or_test = train_or_test
        self.cam = cam
        self.iso = iso
        self.num_patches_per_image = num_patches_per_image
        self.patch_size = patch_size
        self.shuffle_patches = shuffle_patches
        self.patch_sampling = patch_sampling
        self.subtract_images = subtract_images
        self.transform = transform
        self.model = model
        self.file_names_tuple, self.cnt_inst = sidd_medium_filenames_tuple(
            sidd_path=self.sidd_medium_path,
            train_or_test=self.train_or_test,
            cam=self.cam, iso=self.iso,
            first_im_idx=first_im_idx, last_im_idx=last_im_idx
        )
        self.last_updated_row = 0
        self.input_key_name = 'noise' if self.raw else 'noisy'

        first_iter = True
        for i, file_name_tuple in enumerate(self.file_names_tuple):
            if self.raw:
                img1, img2, nlf0, nlf1, iso, cam = load_one_tuple_images(file_name_tuple, subtract=self.subtract_images)
            else:
                img1, img2, iso, cam = load_one_tuple_srgb_images(file_name_tuple)
                nlf0 = None
                nlf1 = None

            img1_patches, img2_patches = extract_patches(
                (img1, img2),
                num_patches=self.num_patches_per_image,
                patch_size=self.patch_size,
                sampling=self.patch_sampling,
                shuffle=self.shuffle_patches
            )
            img1_patches = img1_patches.transpose((0, 3, 1, 2))
            img2_patches = img2_patches.transpose((0, 3, 1, 2))

            if first_iter:
                first_iter = False
                array_size = len(self.file_names_tuple) + 3
                self.input = np.zeros((array_size * img1_patches.shape[0],) + img1_patches.shape[1:])
                self.clean = np.zeros((array_size * img1_patches.shape[0],) + img1_patches.shape[1:])
                self.iso = np.zeros((array_size * img1_patches.shape[0],) + img1_patches.shape[1:])
                self.cam = np.zeros((array_size * img1_patches.shape[0],) + img1_patches.shape[1:])
                self.nlf0 = np.zeros((array_size * img1_patches.shape[0],) + img1_patches.shape[1:])
                self.nlf1 = np.zeros((array_size * img1_patches.shape[0],) + img1_patches.shape[1:])
                self.pid = np.zeros(array_size * img1_patches.shape[0])

            self.input[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]] = img1_patches
            self.clean[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]] = img2_patches
            self.iso[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]] = np.full(img1_patches.shape, iso)
            self.cam[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]] = np.full(img1_patches.shape, cam)
            self.nlf0[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]] = np.full(img1_patches.shape, nlf0)
            self.nlf1[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]] = np.full(img1_patches.shape, nlf1)
            self.pid[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]] = np.arange(img1_patches.shape[0])

            if self.model:
                with torch.no_grad():
                    self.model.eval()
                    kwargs = {
                        'clean': torch.from_numpy(img2_patches).to(torch.float).to(self.device),
                        'eps_std': torch.tensor(self.temp, device=self.device),
                        'iso': torch.from_numpy(self.iso[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]]).to(torch.float).to(self.device),
                        'cam': torch.from_numpy(self.cam[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]]).to(torch.float).to(self.device)
                    }
                    input_image = self.model.sample(**kwargs)

                self.input[self.last_updated_row: self.last_updated_row + img1_patches.shape[0]] = input_image.cpu()

            self.last_updated_row += img1_patches.shape[0]

    def __len__(self):
        return self.last_updated_row

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            self.input_key_name: torch.from_numpy(self.input[idx]).to(torch.float),
            'clean': torch.from_numpy(self.clean[idx]).to(torch.float),
            'iso': torch.from_numpy(self.iso[idx]).to(torch.float),
            'cam': torch.from_numpy(self.cam[idx]).to(torch.float),
            'pid': torch.tensor(self.pid[idx])
        }

        if self.raw:
            sample.update({
                'nlf0': torch.from_numpy(self.nlf0[idx]).to(torch.float),
                'nlf1': torch.from_numpy(self.nlf1[idx]).to(torch.float)
            })

        if self.transform:
            sample = self.transform(sample)

        return sample

class SIDDFullRawDatasetWrapper:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.data_batch = self.dataset[0]
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_batch is None:
            if self.cursor >= self.dataset_size:
                self._reset()
                raise StopIteration
            self.data_batch = self.dataset[self.cursor]

        noisy1, noisy2, nlf0, nlf1, iso, cam = self.data_batch['noisy1'], self.data_batch['noisy2'], self.data_batch['nlf0'], self.data_batch['nlf1'], self.data_batch['iso'], self.data_batch['cam']

        end_idx = min(noisy1.shape[0], self.batch_size)
        noisy1_batch, noisy2_batch = noisy1[:end_idx, :, :, :], noisy2[:end_idx, :, :, :]
        
        if noisy1.shape[0] == end_idx:
            self.data_batch = None
            self.cursor += 1
        else:
            self.data_batch = {'noisy1' : noisy1[end_idx:, :, :, :], 'noisy2' : noisy2[end_idx:, :, :, :], 'nlf0': nlf0, 'nlf1': nlf1, 'iso': iso, 'cam' : cam}
        
        return {'noisy1' : noisy1_batch, 'noisy2' : noisy2_batch, 'nlf0': nlf0, 'nlf1': nlf1, 'iso': iso, 'cam' : cam}

    def _reset(self):
        self.data_batch = self.dataset[0]
        self.cursor = 0

class SIDDMediumDatasetWrapper:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.data_batch = self.dataset[0]
        self.cursor = 0
        self.raw = True if 'nlf0' in self.data_batch.keys() else False
        self.cnt_inst = dataset.cnt_inst

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_batch is None:
            if self.cursor >= self.dataset_size:
                self._reset()
                raise StopIteration
            self.data_batch = self.dataset[self.cursor]

        if self.raw:
            noise, gt, nlf0, nlf1, iso, cam = self.data_batch['noise'], self.data_batch['clean'], self.data_batch['nlf0'], self.data_batch['nlf1'], self.data_batch['iso'], self.data_batch['cam']
        else:
            noise, gt, iso, cam = self.data_batch['noise'], self.data_batch['clean'], self.data_batch['iso'], self.data_batch['cam']

        end_idx = min(noise.shape[0], self.batch_size)
        noise_batch, gt_batch = noise[:end_idx, :, :, :], gt[:end_idx, :, :, :]

        if noise.shape[0] == end_idx:
            self.data_batch = None
            self.cursor += 1
        else:
            if self.raw:
                self.data_batch = {'noise' : noise[end_idx:, :, :, :], 'clean' : gt[end_idx:, :, :, :], 'nlf0': nlf0, 'nlf1': nlf1, 'iso': iso, 'cam' : cam}
            else:
                self.data_batch = {'noise' : noise[end_idx:, :, :, :], 'clean' : gt[end_idx:, :, :, :], 'iso': iso, 'cam' : cam}

        if self.raw:
            return {'noise' : noise_batch, 'clean' : gt_batch, 'nlf0': nlf0, 'nlf1': nlf1, 'iso': iso, 'cam' : cam}
        else:
            return {'noise' : noise_batch, 'clean' : gt_batch, 'iso': iso, 'cam' : cam}

    def _reset(self):
        self.data_batch = self.dataset[0]
        self.cursor = 0

class SIDDFullRawDatasetV2(Dataset):
    """SIDD Full Raw dataset used for training DnCNN (Simpler version)."""

    def __init__(self, sidd_full_path, train_or_test='train', cam=None, iso=None, num_patches_per_image=2898, patch_size=(32, 32), patch_sampling='uniform', shuffle_patches=False, subtract_images=False, transform=None):
        self.sidd_full_path = sidd_full_path
        self.train_or_test = train_or_test
        self.cam = cam
        self.iso = iso
        self.num_patches_per_image = num_patches_per_image
        self.patch_size = patch_size
        self.shuffle_patches = shuffle_patches
        self.patch_sampling = patch_sampling
        self.subtract_images = subtract_images
        self.transform = transform

        self.len = sidd_full_filenames_len(sidd_full_path=self.sidd_full_path, train_or_test=self.train_or_test, cam=self.cam, iso=self.iso)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name_tuple = get_sidd_filename_tuple(idx, sidd_full_path=self.sidd_full_path, train_or_test=self.train_or_test, cam=self.cam, iso=self.iso)
        img1, img2, nlf0, nlf1, iso, cam = load_one_tuple_images(file_name_tuple, subtract=self.subtract_images)
        img1_patches, img2_patches = extract_patches((img1, img2), num_patches=self.num_patches_per_image, patch_size=self.patch_size, sampling=self.patch_sampling, shuffle=self.shuffle_patches)
        img1_patches = img1_patches.transpose((0, 3, 1, 2))
        img2_patches = img2_patches.transpose((0, 3, 1, 2))

        sample = {'noisy1' : torch.from_numpy(img1_patches), 'noisy2' : torch.from_numpy(img2_patches), 'nlf0': nlf0, 'nlf1': nlf1, 'iso': iso, 'cam' : cam}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SIDDMediumDatasetV2(Dataset):
    """SIDD Raw/SRGB Medium dataset used for evaluating DnCNN (Simpler version)."""

    def __init__(self, sidd_medium_path, train_or_test='train', cam=None, iso=None, num_patches_per_image=2898, patch_size=(32, 32), patch_sampling='uniform', shuffle_patches=False, subtract_images=True, transform=None):
        self.sidd_medium_path = sidd_medium_path
        self.raw = 'raw' in sidd_medium_path.lower()
        self.train_or_test = train_or_test
        self.cam = cam
        self.iso = iso
        self.num_patches_per_image = num_patches_per_image
        self.patch_size = patch_size
        self.shuffle_patches = shuffle_patches
        self.patch_sampling = patch_sampling
        self.subtract_images = subtract_images
        self.transform = transform

        self.file_names_tuple, self.cnt_inst = sidd_medium_filenames_tuple(sidd_path=self.sidd_medium_path, train_or_test=self.train_or_test, cam=self.cam, iso=self.iso)

    def __len__(self):
        return len(self.file_names_tuple)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name_tuple = self.file_names_tuple[idx]
        if self.raw:
            img1, img2, nlf0, nlf1, iso, cam = load_one_tuple_images(file_name_tuple, subtract=self.subtract_images)
            img1_patches, img2_patches = extract_patches((img1, img2), num_patches=self.num_patches_per_image, patch_size=self.patch_size, sampling=self.patch_sampling, shuffle=self.shuffle_patches)
            img1_patches = img1_patches.transpose((0, 3, 1, 2))
            img2_patches = img2_patches.transpose((0, 3, 1, 2))

            sample = {'noise' : torch.from_numpy(img1_patches), 'clean' : torch.from_numpy(img2_patches), 'nlf0': nlf0, 'nlf1': nlf1, 'iso': iso, 'cam' : cam}
        else:
            img1, img2, iso, cam = load_one_tuple_srgb_images(file_name_tuple)
            img1_patches, img2_patches = extract_patches((img1, img2), num_patches=self.num_patches_per_image, patch_size=self.patch_size, sampling=self.patch_sampling, shuffle=self.shuffle_patches)
            img1_patches = img1_patches.transpose((0, 3, 1, 2))
            img2_patches = img2_patches.transpose((0, 3, 1, 2))

            sample = {'noise' : torch.from_numpy(img1_patches).to(torch.float), 'clean' : torch.from_numpy(img2_patches).to(torch.float), 'iso': iso, 'cam' : cam}
        
        if self.transform:
            sample = self.transform(sample)

        return sample