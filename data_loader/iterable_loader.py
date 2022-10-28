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
from data_loader.sidd_utils import load_one_tuple_images, load_one_tuple_srgb_images, sidd_full_filenames_len, sidd_medium_filenames_tuple, get_sidd_filename_tuple, extract_patches, load_raw_np_images, divide_parts
from torch.utils.data import Dataset, IterableDataset, DataLoader
from data_loader.sidd_utils import calc_kldiv_mb
import torch
import numpy as np

class IterableSIDDFullRawDataset(IterableDataset):
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
        return self.len * self.num_patches_per_image

    def patch_generator(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            start = 0
            end = self.len

        else:
            image_div_parts = divide_parts(self.len, worker_info.num_workers)
            start, end = sum(image_div_parts[:worker_info.id]), sum(image_div_parts[:worker_info.id+1])

        for idx in range(start, end):
            file_name_tuple = get_sidd_filename_tuple(idx, sidd_full_path=self.sidd_full_path, train_or_test=self.train_or_test, cam=self.cam, iso=self.iso, numpy=False)

            # img1_patches, img2_patches, nlf0, nlf1, iso, cam = load_raw_np_images(file_name_tuple, subtract=self.subtract_images)

            img1, img2, nlf0, nlf1, iso, cam = load_one_tuple_images(file_name_tuple, subtract=self.subtract_images)
            img1_patches, img2_patches = extract_patches((img1, img2), num_patches=self.num_patches_per_image, patch_size=self.patch_size, sampling=self.patch_sampling, shuffle=self.shuffle_patches)
            img1_patches = img1_patches.transpose((0, 3, 1, 2))
            img2_patches = img2_patches.transpose((0, 3, 1, 2))

            for patch_idx in range(len(img1_patches)):
                sample = {'noisy1' : torch.from_numpy(img1_patches[patch_idx]), 'noisy2' : torch.from_numpy(img2_patches[patch_idx]), 'nlf0': torch.full(img1_patches[patch_idx].shape, nlf0).to(torch.float), 'nlf1': torch.full(img1_patches[patch_idx].shape, nlf1).to(torch.float), 'iso': torch.full(img1_patches[patch_idx].shape, iso).to(torch.float), 'cam' : torch.full(img1_patches[patch_idx].shape, cam).to(torch.float)}

                if self.transform:
                    sample = self.transform(sample)

                yield sample

    def __iter__(self):
        return self.patch_generator()


class IterableSIDDMediumDataset(IterableDataset):
    def __init__(self, sidd_medium_path, train_or_test='train', cam=None, iso=None, num_patches_per_image=2898,
                 patch_size=(32, 32), patch_sampling='uniform', shuffle_patches=False, subtract_images=True,
                 transform=None, is_raw=True, first_im_idx=10, last_im_idx=12, no_patching=False, no_patch_size=None,
                 model=None, temp=None, device=None, exclude_iso=None):
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
        self.no_patching = no_patching
        self.no_patch_size = no_patch_size
        self.model = model
        self.device = device
        self.temp = temp


        self.file_names_tuple, self.cnt_inst = sidd_medium_filenames_tuple(
            sidd_path=self.sidd_medium_path,
            train_or_test=self.train_or_test,
            cam=self.cam,
            iso=self.iso,
            first_im_idx=first_im_idx,
            last_im_idx=last_im_idx,
            exclude_iso=exclude_iso
        )

    def __len__(self):
        return len(self.file_names_tuple) * self.num_patches_per_image

    def patch_generator(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            start = 0
            end = len(self.file_names_tuple)

        else:
            image_div_parts = divide_parts(len(self.file_names_tuple), worker_info.num_workers)
            start, end = sum(image_div_parts[:worker_info.id]), sum(image_div_parts[:worker_info.id+1])

        for idx in range(start, end):
            file_name_tuple = self.file_names_tuple[idx]

            if self.raw:
                img1, img2, nlf0, nlf1, iso, cam = load_one_tuple_images(file_name_tuple, subtract=self.subtract_images)
                input_key_name = 'noise'
            else:
                img1, img2, iso, cam = load_one_tuple_srgb_images(file_name_tuple)
                input_key_name = 'noisy'


            if not self.no_patching:
                img1_patches, img2_patches = extract_patches((img1, img2), num_patches=self.num_patches_per_image, patch_size=self.patch_size, sampling=self.patch_sampling, shuffle=self.shuffle_patches)
                img1_patches = img1_patches.transpose((0, 3, 1, 2))
                img2_patches = img2_patches.transpose((0, 3, 1, 2))

                for patch_idx in range(len(img1_patches)):
                    sample = {
                        input_key_name : torch.from_numpy(img1_patches[patch_idx]).to(torch.float),
                        'clean' : torch.from_numpy(img2_patches[patch_idx]).to(torch.float),
                        'iso': torch.full(img1_patches[patch_idx].shape, iso).to(torch.float),
                        'cam': torch.full(img1_patches[patch_idx].shape, cam).to(torch.float),
                        'pid': torch.tensor(patch_idx)
                    }

                    if self.raw:
                        sample.update({'nlf0': torch.full(img1_patches[patch_idx].shape, nlf0).to(torch.float), 'nlf1': torch.full(img1_patches[patch_idx].shape, nlf1).to(torch.float)})

                    if self.transform:
                        sample = self.transform(sample)

                    yield sample
            else:
                img1 = np.squeeze(img1[:, :self.no_patch_size, :self.no_patch_size, :].transpose((0, 3, 1, 2)))
                img2 = np.squeeze(img2[:, :self.no_patch_size, :self.no_patch_size, :].transpose((0, 3, 1, 2)))
                sample = {
                    input_key_name : torch.from_numpy(img1).to(torch.float),
                    'clean' : torch.from_numpy(img2).to(torch.float),
                    'iso': torch.full([3, self.no_patch_size, self.no_patch_size], iso).to(torch.float),
                    'cam': torch.full([3, self.no_patch_size, self.no_patch_size], cam).to(torch.float),
                    'pid': torch.tensor(0)
                }

                if self.model:
                    with torch.no_grad():
                        self.model.eval()
                        kwargs = {
                            'clean': torch.unsqueeze(sample['clean'].to(self.device), dim=0),
                            'eps_std': torch.tensor(self.temp, device=self.device),
                            'iso': torch.unsqueeze(sample['iso'].to(self.device), dim=0),
                            'cam': torch.unsqueeze(sample['cam'].to(torch.float).to(self.device), dim=0)
                        }
                        noise = self.model.sample(**kwargs)
                    kwargs.update({
                        input_key_name: torch.unsqueeze(sample[input_key_name], dim=0),
                        'pid': torch.tensor([0])
                    })
                    kwargs['clean'] = kwargs['clean'].to('cpu')
                    kldiv_batch, cnt_batch = calc_kldiv_mb(
                        kwargs,
                        noise.data.to('cpu'),
                        '',
                        None,
                        2
                    )

                    sample[input_key_name] = torch.squeeze(noise.cpu())
                    # print(kldiv_batch/ cnt_batch)

                yield sample

    def __iter__(self):
        return self.patch_generator()
