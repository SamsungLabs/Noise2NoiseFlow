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
import h5py
import numpy as np
from scipy.io import loadmat, savemat
import gc
from sklearn.utils import shuffle
import glob
import os
import random
import cv2
import pickle
import matplotlib.pyplot as plt

RNG = np.random.RandomState(42)

def pack_raw(raw_im):
    """Packs Bayer image to 4 channels (h, w) --> (h/2, w/2, 4)."""
    # pack Bayer image to 4 channels
    im = np.expand_dims(raw_im, axis=2)
    img_shape = im.shape
    h = img_shape[0]
    w = img_shape[1]
    out = np.concatenate((im[0:h:2, 0:w:2, :],
                          im[0:h:2, 1:w:2, :],
                          im[1:h:2, 1:w:2, :],
                          im[1:h:2, 0:w:2, :]), axis=2)

    del raw_im
    gc.collect()

    return out

def unpack_raw(raw4ch):
    """Unpacks 4 channels to Bayer image (h/2, w/2, 4) --> (h, w)."""
    img_shape = raw4ch.shape
    h = img_shape[0]
    w = img_shape[1]
    # d = img_shape[2]
    bayer = np.zeros([h * 2, w * 2], dtype=np.float32)
    # bayer = raw4ch
    # bayer.reshape((h * 2, w * 2))
    bayer[0::2, 0::2] = raw4ch[:, :, 0]
    bayer[0::2, 1::2] = raw4ch[:, :, 1]
    bayer[1::2, 1::2] = raw4ch[:, :, 2]
    bayer[1::2, 0::2] = raw4ch[:, :, 3]
    return bayer

def get_nlf(metadata):
    nlf = metadata['UnknownTags'][7, 0][2][0][0:2]
    return nlf

def load_metadata(meta_path):
    """Loads metadata from file."""
    meta = loadmat(meta_path)
    meta = meta['metadata']
    return meta[0, 0]

def load_one_tuple_images(filepath_tuple, subtract=False):
    image1_path = filepath_tuple[0]  # index 0: input noisy image path
    image2_path = filepath_tuple[1]  # index 1: ground truth image path
    meta_path = filepath_tuple[2]  # index 2: metadata path

    with h5py.File(image1_path, 'r') as f:  # (use this for .mat files with -v7.3 format)
        raw = f[list(f.keys())[0]]  # use the first and only key
        # input_image = np.transpose(raw)  # TODO: transpose?
        input_image = np.expand_dims(pack_raw(raw), axis=0)
        input_image = np.nan_to_num(input_image)
        input_image = np.clip(input_image, 0.0, 1.0)

    with h5py.File(image2_path, 'r') as f:
        gt_raw = f[list(f.keys())[0]]  # use the first and only key
        # gt_image = np.transpose(gt_raw)  # TODO: transpose?
        gt_image = np.expand_dims(pack_raw(gt_raw), axis=0)
        gt_image = np.nan_to_num(gt_image)
        gt_image = np.clip(gt_image, 0.0, 1.0)

    var_image = []

    nlf0, nlf1 = extract_nlf(meta_path)
    iso, cam = extract_iso_cam(image1_path)

    # use noise layer instead of noise image TODO: just to be aware of this crucial step
    if subtract:
        input_image = input_image - gt_image

    (one, h, w, c) = input_image.shape

    return input_image, gt_image, nlf0, nlf1, iso, cam

def load_one_tuple_srgb_images(filepath_tuple):
    in_path = filepath_tuple[0]  # index 0: input noisy image path
    gt_path = filepath_tuple[1]  # index 1: ground truth image path
    
    input_image = cv2.imread(in_path).astype(int)
    gt_image = cv2.imread(gt_path).astype(int)
    iso, cam = extract_iso_cam(in_path)

    input_image = input_image[np.newaxis, ...]
    gt_image = gt_image[np.newaxis, ...]

    return input_image, gt_image, iso, cam

def extract_nlf(meta_path):
    metadata = load_metadata(meta_path)
    nlf0, nlf1 = get_nlf(metadata)
    # fix NLF
    nlf0 = 1e-6 if nlf0 <= 0 else nlf0
    nlf1 = 1e-6 if nlf1 <= 0 else nlf1

    return nlf0, nlf1

def extract_iso_cam(file_path):
    fparts = file_path.split('/')
    sdir = fparts[-3]
    if len(sdir) != 30:
        sdir = fparts[-2]  # if subdirectory does not exist
    iso = float(sdir[12:17])
    cam = float(['IP', 'GP', 'S6', 'N6', 'G4'].index(sdir[9:11]))

    return iso, cam
    
def sample_indices_uniform(h, w, ph, pw, shuf=False, n_pat_per_im=None):
    """Uniformly sample patch indices from (0, 0) up to (h, w) with patch height and width (ph, pw) """
    ii = []
    jj = []
    n_p = 0
    for i in np.arange(0, h - ph + 1, ph):
        for j in np.arange(0, w - pw + 1, pw):
            ii.append(i)
            jj.append(j)
            n_p += 1
            if (n_pat_per_im is not None) and (n_p == n_pat_per_im):
                break
        if (n_pat_per_im is not None) and (n_p == n_pat_per_im):
            break
    if shuf:
        ii, jj = shuffle(ii, jj)
    return ii, jj, n_p

def sample_indices_random(h, w, ph, pw, n_p):
    """Randomly sample n_p patch indices from (0, 0) up to (h, w) with patch height and width (ph, pw) """
    ii = []
    jj = []
    for k in np.arange(0, n_p):
        i = np.random.randint(0, h - ph + 1)
        j = np.random.randint(0, w - pw + 1)
        ii.append(i)
        jj.append(j)
    return ii, jj

def extract_patches(im_tuple, num_patches, patch_size, sampling='uniform', shuffle=False):
    image1, image2 = im_tuple
    H, W = image1.shape[1:3]
    patch_height, patch_width = patch_size
    if sampling == 'uniform':
        ii, jj, n_p = sample_indices_uniform(H, W, patch_height, patch_width, shuf=shuffle, n_pat_per_im=num_patches)
        assert n_p == num_patches
    elif sampling == 'random':
        ii, jj = sample_indices_random(H, W, patch_height, patch_width, num_patches)
    elif sampling == 'dncnn':
        stride = 64
        image1_patches, image2_patches = [], []
        for i in range(0, H - patch_height + 1, stride):
            for j in range(0, W - patch_width + 1, stride):
                img1_patch = image1[:, i:i + patch_height, j:j + patch_width, :]
                img2_patch = image2[:, i:i + patch_height, j:j + patch_width, :]
                image1_patches.append(img1_patch)
                image2_patches.append(img2_patch)
        
        return np.concatenate(image1_patches, axis=0), np.concatenate(image2_patches, axis=0)
    else:
        raise ValueError('Invalid sampling mode: {}'.format(sampling))
    
    image1_patches, image2_patches = [], []
    for (i, j) in zip(ii, jj):
        img1_patch = image1[:, i:i + patch_height, j:j + patch_width, :]
        img2_patch = image2[:, i:i + patch_height, j:j + patch_width, :]
        image1_patches.append(img1_patch)
        image2_patches.append(img2_patch)
    
    return np.concatenate(image1_patches, axis=0), np.concatenate(image2_patches, axis=0)

def get_sidd_filename_tuple(idx, sidd_full_path, train_or_test='train', numpy=False, cam=None, iso=None):
    if train_or_test == 'train':
        inst_idxs = [4, 11, 13, 17, 18, 20, 22, 23, 25, 27, 28, 29, 30, 34, 35, 39, 40, 42, 43, 44, 45, 47, 81, 86, 88,
                     90, 101, 102, 104, 105, 110, 111, 115, 116, 125, 126, 127, 129, 132, 135,
                     138, 140, 175, 177, 178, 179, 180, 181, 185, 186, 189, 192, 193, 194, 196, 197]
        # removed: 114, 134, 184, 136, 190, 188, 117, 137, 191
    else:
        inst_idxs = [54, 55, 57, 59, 60, 62, 63, 66, 150, 151, 152, 155, 159, 160, 161, 163, 164, 165, 166, 198,
                     199] # removed 154 since it raises PermissionDenied error. find a way and add it back

    counter = 0
    for id in inst_idxs:
        id_str = '%04d' % id
        subdir = glob.glob(os.path.join(sidd_full_path, id_str + '*'))[0]

        _, _, inst_cam, inst_iso, _, _, _ = subdir.split('/')[-1].split('_')
        inst_iso = int(inst_iso)

        if (cam is not None) and (inst_cam != cam):
            continue
        if (iso is not None) and (iso != 0) and (inst_iso != iso):
            continue

        if counter != idx:
            counter += 1
            continue
        else:
            noisy_dir = '{}_NOISY_RAW'.format(id_str)
            metadata_dir = '{}_METADATA_RAW'.format(id_str)

            num_noisy_images = len(os.listdir(os.path.join(subdir, noisy_dir)))
            noisy1_idx = random.randint(1, num_noisy_images-1)
            noisy2_idx = noisy1_idx + 1

            if numpy:
                noisy1_img_path = os.path.join(subdir, noisy_dir, '{}_NOISY_RAW_{}.npy'.format(id_str, '%03d' % noisy1_idx))
                noisy2_img_path = os.path.join(subdir, noisy_dir, '{}_NOISY_RAW_{}.npy'.format(id_str, '%03d' % noisy2_idx))
                metadata_path = os.path.join(subdir, metadata_dir, '{}_METADATA_RAW_{}.npy'.format(id_str, '%03d' % noisy1_idx))
            else:
                noisy1_img_path = os.path.join(subdir, noisy_dir, '{}_NOISY_RAW_{}.MAT'.format(id_str, '%03d' % noisy1_idx))
                noisy2_img_path = os.path.join(subdir, noisy_dir, '{}_NOISY_RAW_{}.MAT'.format(id_str, '%03d' % noisy2_idx))
                metadata_path = os.path.join(subdir, metadata_dir, '{}_METADATA_RAW_{}.MAT'.format(id_str, '%03d' % noisy1_idx))

            data_tuple = tuple((noisy1_img_path, noisy2_img_path, metadata_path))

            return data_tuple
    raise ValueError('index out of range. max length is {}'.format(counter))

def sidd_full_filenames_len(sidd_full_path, train_or_test='train', cam=None, iso=None):
    if train_or_test == 'train':
        inst_idxs = [4, 11, 13, 17, 18, 20, 22, 23, 25, 27, 28, 29, 30, 34, 35, 39, 40, 42, 43, 44, 45, 47, 81, 86, 88,
                     90, 101, 102, 104, 105, 110, 111, 115, 116, 125, 126, 127, 129, 132, 135,
                     138, 140, 175, 177, 178, 179, 180, 181, 185, 186, 189, 192, 193, 194, 196, 197]
        # removed: 114, 134, 184, 136, 190, 188, 117, 137, 191
    else:
        inst_idxs = [54, 55, 57, 59, 60, 62, 63, 66, 150, 151, 152, 155, 159, 160, 161, 163, 164, 165, 166, 198,
                     199] # removed 154 since it raises PermissionDenied error. find a way and add it back

    cntr = 0

    for id in inst_idxs:
        id_str = '%04d' % id
        subdir = glob.glob(os.path.join(sidd_full_path, id_str + '*'))[0]

        _, _, inst_cam, inst_iso, _, _, _ = subdir.split('/')[-1].split('_')
        inst_iso = int(inst_iso)

        if (cam is not None) and (inst_cam != cam):
            continue
        if (iso is not None) and (iso != 0) and (inst_iso != iso):
            continue

        cntr += 1

    return cntr

def sidd_full_filenames_tuple(sidd_full_path, train_or_test='train', numpy=False, cam=None, iso=None):
    if train_or_test == 'train':
        inst_idxs = [4, 11, 13, 17, 18, 20, 22, 23, 25, 27, 28, 29, 30, 34, 35, 39, 40, 42, 43, 44, 45, 47, 81, 86, 88,
                     90, 101, 102, 104, 105, 110, 111, 115, 116, 125, 126, 127, 129, 132, 135,
                     138, 140, 175, 177, 178, 179, 180, 181, 185, 186, 189, 192, 193, 194, 196, 197]
        # removed: 114, 134, 184, 136, 190, 188, 117, 137, 191
    else:
        inst_idxs = [54, 55, 57, 59, 60, 62, 63, 66, 150, 151, 152, 155, 159, 160, 161, 163, 164, 165, 166, 198,
                     199] # removed 154 since it raises PermissionDenied error. find a way and add it back

    fns = []

    for id in inst_idxs:
        id_str = '%04d' % id
        subdir = glob.glob(os.path.join(sidd_full_path, id_str + '*'))[0]

        _, _, inst_cam, inst_iso, _, _, _ = subdir.split('/')[-1].split('_')
        inst_iso = int(inst_iso)

        if (cam is not None) and (inst_cam != cam):
            continue
        if (iso is not None) and (iso != 0) and (inst_iso != iso):
            continue

        noisy_dir = '{}_NOISY_RAW'.format(id_str)
        metadata_dir = '{}_METADATA_RAW'.format(id_str)

        num_noisy_images = len(os.listdir(os.path.join(subdir, noisy_dir)))
        noisy1_idx = random.randint(1, num_noisy_images-1)
        noisy2_idx = noisy1_idx + 1

        if numpy:
            noisy1_img_path = os.path.join(subdir, noisy_dir, '{}_NOISY_RAW_{}.npy'.format(id_str, '%03d' % noisy1_idx))
            noisy2_img_path = os.path.join(subdir, noisy_dir, '{}_NOISY_RAW_{}.npy'.format(id_str, '%03d' % noisy2_idx))
            metadata_path = os.path.join(subdir, metadata_dir, '{}_METADATA_RAW_{}.npy'.format(id_str, '%03d' % noisy1_idx))
        else:
            noisy1_img_path = os.path.join(subdir, noisy_dir, '{}_NOISY_RAW_{}.MAT'.format(id_str, '%03d' % noisy1_idx))
            noisy2_img_path = os.path.join(subdir, noisy_dir, '{}_NOISY_RAW_{}.MAT'.format(id_str, '%03d' % noisy2_idx))
            metadata_path = os.path.join(subdir, metadata_dir, '{}_METADATA_RAW_{}.MAT'.format(id_str, '%03d' % noisy1_idx))

        data_tuple = tuple((noisy1_img_path, noisy2_img_path, metadata_path))
        fns.append(data_tuple)

    return fns

def sidd_medium_filenames_tuple(sidd_path, train_or_test='train', numpy=False, first_im_idx=10, last_im_idx=12, cam=None, iso=None, exclude_iso=None):
    """Returns filenames: list of tuples: (input noisy, ground truth, per-pixel variance, metadata), all .MAT
    """
    if train_or_test == 'train':
        inst_idxs = [4, 11, 13, 17, 18, 20, 22, 23, 25, 27, 28, 29, 30, 34, 35, 39, 40, 42, 43, 44, 45, 47, 81, 86, 88,
                     90, 101, 102, 104, 105, 110, 111, 115, 116, 125, 126, 127, 129, 132, 135,
                     138, 140, 175, 177, 178, 179, 180, 181, 185, 186, 189, 192, 193, 194, 196, 197]
        # removed: 114, 134, 184, 136, 190, 188, 117, 137, 191
    elif train_or_test == 'train_dncnn':
        inst_idxs = [1,   2,   3,   4,   5,   6,   7,   8,  10,  11,  12, 13,  14,  15,  16,  17,  18,
                     19,  20,  22,  23,  25, 27,  28,  29,  30,  32,  33,  34,  35,  38, 39,  40,  42,
                     43,  44,  45,  47,  48,  51, 52,  54,  55,  57,  59,  60,  62,  63, 66,  75,  77,
                     81,  86,  87,  88,  90, 94,  98,  101, 102, 104, 105, 110, 111, 113, 114, 115, 116,
                     117, 118, 122, 125, 126, 127, 129, 132, 133, 134, 135, 136, 137, 138, 140, 142,
                     147, 149, 150, 151, 152, 154, 155, 156, 159, 160, 161, 163, 164, 165, 166, 167,
                     169, 172, 175, 177, 178, 179, 180, 181, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194,
                     195, 196, 197, 198, 199]
    elif train_or_test == 'all':
        inst_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 27, 28, 29, 30,
                      32,  33,  34,  35, 36,  38, 39,  40,  42,
                     43,  44,  45,  47,  48,  50, 51, 52,  54,  55,  57,  59,  60,  62,  63, 64, 65, 66, 68,  69, 70, 72, 73, 75, 76, 77, 78,
                     80, 81, 83, 84,  86,  87,  88, 89, 90, 91, 92,  94, 96, 97, 98, 99,  101, 102, 104, 105, 106, 107, 108, 110, 111, 113, 114, 115, 116,
                     117, 118, 120, 121, 122, 123, 125, 126, 127, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142,
                     144, 145, 146, 147, 149, 150, 151, 152, 154, 155, 156,157, 159, 160, 161, 163, 164, 165, 166, 167, 168,
                     169, 172, 173, 175, 177, 178, 179, 180, 181, 182, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194,
                     195, 196, 197, 198, 199, 200]
    else:
        inst_idxs = [54, 55, 57, 59, 60, 62, 63, 66, 150, 151, 152, 154, 155, 159, 160, 161, 163, 164, 165, 166, 198,
                     199]
    # example: 0001_001_S6_00100_00060_3200_L
    cnt_inst = 0
    fns = []
    for id in inst_idxs:
        id_str = '%04d' % id
        subdir = os.path.split(glob.glob(os.path.join(sidd_path, id_str + '*'))[0])[-1]
        if (cam is not None) and (subdir[9:11] != cam):
            continue
        if (iso is not None) and (iso != 0) and (int(subdir[12:17]) != iso):
            continue
        if (exclude_iso is not None) and int(subdir[12:17]) in exclude_iso:
            continue
        n_files = len(glob.glob(os.path.join(sidd_path, subdir, id_str + '_GT_RAW', '*.MAT')))
        for i in range(first_im_idx, last_im_idx):
            if numpy:
                if 'SIDD_Medium_Srgb' in sidd_path:
                    a_tuple = tuple(
                        (
                            os.path.join(sidd_path, subdir, id_str + '_NOISY_SRGB_%03d.npy' % i),
                            os.path.join(sidd_path, subdir, id_str + '_GT_SRGB_%03d.npy' % i),
                        )
                    )
                elif 'SIDD_Medium' in sidd_path:
                    a_tuple = tuple(
                        (
                            os.path.join(sidd_path, subdir, id_str + '_NOISY_RAW_%03d.npy' % i),
                            os.path.join(sidd_path, subdir, id_str + '_GT_RAW_%03d.npy' % i),
                            os.path.join(sidd_path, subdir, id_str + '_METADATA_RAW_%03d.npy' % i)
                        )
                    )
                else:
                    raise ValueError('Invalid path for SIDD_Medium.')
            else:
                if 'SIDD_Medium_Srgb' in sidd_path:
                    a_tuple = tuple(
                        (
                            os.path.join(sidd_path, subdir, id_str + '_NOISY_SRGB_%03d.PNG' % i),
                            os.path.join(sidd_path, subdir, id_str + '_GT_SRGB_%03d.PNG' % i),
                        )
                    )
                elif 'SIDD_Medium' in sidd_path:
                    a_tuple = tuple(
                        (
                            os.path.join(sidd_path, subdir, id_str + '_NOISY_RAW_%03d.MAT' % i),
                            os.path.join(sidd_path, subdir, id_str + '_GT_RAW_%03d.MAT' % i),
                            os.path.join(sidd_path, subdir, id_str + '_METADATA_RAW_%03d.MAT' % i)
                        )
                    )
                else:
                    raise ValueError('Invalid path for SIDD_Medium.')
            fns.append(a_tuple)
        cnt_inst += 1

    return fns, cnt_inst

def load_srgb_np_images(filepath_tuple):
    in_path = filepath_tuple[0]  # index 0: input noisy image path
    gt_path = filepath_tuple[1]  # index 1: ground truth image path
    
    input_image = np.load(in_path)
    gt_image = np.load(gt_path)

    fparts = in_path.split('/')
    sdir = fparts[-3]
    if len(sdir) != 30:
        sdir = fparts[-2]  # if subdirectory does not exist
    iso = float(sdir[12:17])
    cam = float(['IP', 'GP', 'S6', 'N6', 'G4'].index(sdir[9:11]))

    # iso, cam = np.load(in_path.replace('NOISY', 'METADATA'))
    
    input_image = input_image - gt_image
    
    return input_image, gt_image, iso, cam

def load_raw_np_images(filepath_tuple, subtract=False):
    image1_path = filepath_tuple[0]  # index 0: input noisy image path
    image2_path = filepath_tuple[1]  # index 1: ground truth image path
    meta_path = filepath_tuple[2]  # index 2: metadata path

    input_image = np.load(image1_path)
    gt_image = np.load(image2_path)
    nlf0, nlf1, iso, cam = np.load(meta_path)

    # use noise layer instead of noise image TODO: just to be aware of this crucial step
    if subtract:
        input_image = input_image - gt_image

    return input_image, gt_image, nlf0, nlf1, iso, cam

def divide_parts(n, n_parts):
    """divide a number into a list of parts"""
    (div, rem) = divmod(n, n_parts)
    divs = [div] * n_parts
    if rem != 0:
        for r in range(rem):
            divs[r] += 1
    return divs

def calc_kldiv_mb(image_dict, x_samples, vis_dir, sc_sd, n_models=4, save_mat=False, save_noisy_img=False, is_raw=True):
    input_key_name = 'noise' if is_raw else 'noisy'
    subdir = "Data"	
    subdir = os.path.join(vis_dir, subdir)
    if not os.path.exists(subdir) and (save_mat or save_noisy_img):	
        os.makedirs(subdir, exist_ok=True)
    step = 5  # 30	
    klds_all = np.ndarray([n_models])	
    klds_all[:] = 0.0	
    cnt = 0	
    # thr = []	
    for i in range(0, image_dict[input_key_name].shape[0], step):
        if is_raw:
            klds = kldiv_patch_set(	
                image_dict[input_key_name][i],	
                image_dict['clean'][i],	
                image_dict['nlf0'][i] if 'nlf0' in image_dict.keys() else None,	
                image_dict['nlf1'][i] if 'nlf1' in image_dict.keys() else None,	
                image_dict['pid'][i],	
                x_samples[i],	
                sc_sd,	
                subdir,
            )
        else:
            klds = kldiv_patch_set_v2(	
                image_dict[input_key_name][i],	
                image_dict['clean'][i],	
                image_dict['nlf0'][i] if 'nlf0' in image_dict.keys() else None,	
                image_dict['nlf1'][i] if 'nlf1' in image_dict.keys() else None,	
                image_dict['pid'][i],	
                x_samples[i],	
                sc_sd,	
                subdir,
                save_mat,
                save_noisy_img
            )
        klds_all += klds	
        cnt += 1	
   	
    return klds_all, cnt
	
def kldiv_patch_set(real_noise, gt, nlf0, nlf1, pid, x_samples, sc_sd, subdir):	
    ng = np.random.normal(0, sc_sd, gt.shape)  # Gaussian	
    ns = x_samples  # NF-sampled	
    n = real_noise  # Real	
    xs = np.clip(gt + ns, 0.0, 1.0)	
    xg = np.clip(gt + ng, 0.0, 1.0)	
    x = np.clip(gt + n, 0.0, 1.0)	
    if nlf0 is None:	
        noise_pats_raw = (ng, ns, n)	
    else:	
        nlf_sd = np.sqrt(nlf0[0, 0, 0] * gt + nlf1[0, 0, 0])  # Camera NLF	
        nl = nlf_sd * np.random.normal(0, 1, gt.shape)  # Camera NLF	
        xl = np.clip(gt + nl.numpy(), 0.0, 1.0)	
        noise_pats_raw = (ng, nl, ns, n)	
    # save_mat = True # TODO: Was True before	
    # # save mat files	
    # if save_mat:	
    #     savemat(os.path.join(subdir, '%s_%04d.mat' % ('gt', pid)), {'x': gt})	
    #     savemat(os.path.join(subdir, '%s_%04d.mat' % ('ng', pid)), {'x': ng})	
    #     savemat(os.path.join(subdir, '%s_%04d.mat' % ('ns', pid)), {'x': ns})	
    #     savemat(os.path.join(subdir, '%s_%04d.mat' % ('n', pid)), {'x': n})	
    #     savemat(os.path.join(subdir, '%s_%04d.mat' % ('xg', pid)), {'x': xg})	
    #     savemat(os.path.join(subdir, '%s_%04d.mat' % ('xs', pid)), {'x': xs})	
    #     savemat(os.path.join(subdir, '%s_%04d.mat' % ('x', pid)), {'x': x})	
    #     if nlf0 is not None:	
    #         savemat(os.path.join(subdir, '%s_%04d.mat' % ('nl', pid)), {'x': nl})	
    #         savemat(os.path.join(subdir, '%s_%04d.mat' % ('xl', pid)), {'x': xl})	
    # histograms	
    bw = 0.2 / 64	
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)	
    cnt_regr = 1
    hists = [None] * len(noise_pats_raw)	
    klds = np.ndarray([len(noise_pats_raw)])	
    klds[:] = 0.0	
    for h in reversed(range(len(noise_pats_raw))):	
        # hists[h], bin_centers = get_histogram(noise_pats_raw[h], bin_edges=bin_edges)
        hists[h] = get_histogram(noise_pats_raw[h], bin_edges=bin_edges, cnt_regr=cnt_regr)	
        # noinspection PyTypeChecker	
        klds[h] = kl_div_forward(hists[-1], hists[h])	
    # savemat(os.path.join(subdir, '%s_%04d.mat' % ('kl_ng', pid)), {'x': klds[0]})	
    # savemat(os.path.join(subdir, '%s_%04d.mat' % ('kl_nl', pid)), {'x': klds[1]})	
    # savemat(os.path.join(subdir, '%s_%04d.mat' % ('kl_ns', pid)), {'x': klds[2]})	
    return klds

def kldiv_patch_set_v2(real_noise, clean, nlf0, nlf1, pid, x_samples, sc_sd, subdir, save_mat, save_noisy_img):	
    ns = x_samples  # NF-sampled	
    n = real_noise  # Real	
    # xs = np.clip(clean + ns, 0.0, 1.0)	
    # x = np.clip(clean + n, 0.0, 1.0)
    is_raw = False if nlf0 is None else True
    if nlf0 is None:
        noise_pats = (ns - clean, n - clean)	
    else:	
        ng = np.random.normal(0, sc_sd, clean.shape)  # Gaussian
        nlf_sd = np.sqrt(nlf0[0, 0, 0] * clean + nlf1[0, 0, 0])  # Camera NLF	
        nl = nlf_sd * np.random.normal(0, 1, clean.shape)  # Camera NLF
        # xg = np.clip(clean + ng, 0.0, 1.0)
        # xl = np.clip(clean + nl.numpy(), 0.0, 1.0)	
        noise_pats = (ng, nl, ns, n)	

    # save mat files	
    if save_mat:
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('clean', pid)), {'x': clean})	
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('ng', pid)), {'x': ng})	
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('ns', pid)), {'x': ns})	
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('n', pid)), {'x': n})	
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('xg', pid)), {'x': xg})	
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('xs', pid)), {'x': xs})	
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('x', pid)), {'x': x})	
        if nlf0 is not None:	
            savemat(os.path.join(subdir, '%s_%04d.mat' % ('nl', pid)), {'x': nl})	
            savemat(os.path.join(subdir, '%s_%04d.mat' % ('xl', pid)), {'x': xl})	
    # histograms
    if is_raw:
        bw = 0.2 / 64	
        bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
        left_edge=0.0
        right_edge=1.0
    else:
        bw = 4
        left_edge = -260.0
        right_edge = 261.0
        bin_edges = np.arange(left_edge, right_edge, bw)

    cnt_regr = 1
    hists = [None] * len(noise_pats)	
    klds = np.ndarray([len(noise_pats)])	
    klds[:] = 0.0
    for h in reversed(range(len(noise_pats))):
        hists[h] = get_histogram(noise_pats[h], bin_edges=bin_edges, cnt_regr=cnt_regr)
        # noinspection PyTypeChecker	
        klds[h] = kl_div_forward(hists[-1], hists[h])	
        if save_noisy_img:
            plt.figure(figsize=[10,8])
            plt.bar(bin_edges[:-1], hists[h], width = 1, color='#0504aa',alpha=0.7)
            plt.savefig(os.path.join(subdir, '{}_hist_{}.png'.format(pid, h)))
    # savemat(os.path.join(subdir, '%s_%04d.mat' % ('kl_ng', pid)), {'x': klds[0]})	
    # savemat(os.path.join(subdir, '%s_%04d.mat' % ('kl_nl', pid)), {'x': klds[1]})	
    # savemat(os.path.join(subdir, '%s_%04d.mat' % ('kl_ns', pid)), {'x': klds[2]})
    if save_noisy_img:
        cv2.imwrite(os.path.join(subdir, '{}_clean.png'.format(pid)), np.array(clean).transpose((1, 2, 0)))
        cv2.imwrite(os.path.join(subdir, '{}_noisy_real.png'.format(pid)), np.array(n).transpose((1, 2, 0)))
        cv2.imwrite(os.path.join(subdir, '{}_noisy_nf.png'.format(pid)), np.array(ns).transpose((1, 2, 0)))

    return klds	

def get_histogram(data, bin_edges=None, cnt_regr=1):
    n = np.prod(data.shape)	
    hist, _ = np.histogram(data, bin_edges)	
    return (hist + cnt_regr)/(n + cnt_regr * len(hist))

def kl_div_forward(p, q):
    assert (~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))).all()	
    idx = (p > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(p * np.log(p / q))
