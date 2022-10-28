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
import scipy.io as sio
import h5py
import cv2

def read_metadata(metadata_file_path):
    # metadata
    meta = sio.loadmat(metadata_file_path)
    meta = meta['metadata'][0, 0]
    # black_level = float(meta['black_level'][0, 0])
    # white_level = float(meta['white_level'][0, 0])
    bayer_pattern = get_bayer_pattern(meta)  # meta['bayer_pattern'].tolist()
    bayer_2by2 = (np.asarray(bayer_pattern) + 1).reshape((2, 2)).tolist()
    # nlf = meta['nlf']
    # shot_noise = nlf[0, 2]
    # read_noise = nlf[0, 3]
    wb = get_wb(meta)
    # cst1 = meta['cst1']
    cst1, cst2 = get_csts(meta)
    # cst2 = cst2.reshape([3, 3])  # use cst2 for rendering, TODO: interpolate between cst1 and cst2
    iso = get_iso(meta)
    cam = get_cam(meta)
    nlf0, nlf1 = get_nlf(meta)
    return meta, bayer_2by2, wb, cst2, iso, cam, nlf0, nlf1


def get_bayer_pattern(metadata):
    bayer_id = 33422
    bayer_tag_idx = 1
    try:
        unknown_tags = metadata['UnknownTags']
        if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
            bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
        else:
            raise Exception
    except:
        try:
            unknown_tags = metadata['SubIFDs'][0, 0]['UnknownTags'][0, 0]
            if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
                bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
            else:
                raise Exception
        except:
            try:
                unknown_tags = metadata['SubIFDs'][0, 1]['UnknownTags']
                if unknown_tags[1]['ID'][0][0][0] == bayer_id:
                    bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
                else:
                    raise Exception
            except:
                print('Bayer pattern not found. Assuming RGGB.')
                bayer_pattern = [1, 2, 2, 3]
    return bayer_pattern


def get_wb(metadata):
    return metadata['AsShotNeutral']


def get_csts(metadata):
    return metadata['ColorMatrix1'].reshape((3, 3)), metadata['ColorMatrix2'].reshape((3, 3))


def get_iso(metadata):
    try:
        iso = metadata['ISOSpeedRatings'][0][0]
    except:
        try:
            iso = metadata['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
        except:
            raise Exception('ISO not found.')
    return iso


def get_cam(metadata):
    model = metadata['Make'][0]
    # cam_ids = [0, 1, 3, 3, 4]  # IP, GP, S6, N6, G4
    cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
    return cam_dict[model]


def get_nlf(metadata):
    nlf = metadata['UnknownTags'][7, 0][2][0][0:2]
    return nlf


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
    return out


def load_raw_image_packed(im_file):
    """Loads and returns a normalized packed raw-RGB image from .mat file (im_file) with dimensions (1, ?, ?, 4)"""
    with h5py.File(im_file, 'r') as f:  # (use this for .mat files with -v7.3 format)
        raw = f[list(f.keys())[0]]  # use the first and only key
        # input_image = np.transpose(raw)  # transpose?
        raw = np.expand_dims(pack_raw(raw), axis=0)
        raw = np.nan_to_num(raw)
        raw = np.clip(raw, 0.0, 1.0)
    return raw


def raw_to_srgb(image, bayer_2by2, wb, cst2):
    image = np.expand_dims(image, 0)
    image = np.squeeze(image)[1:-1, 1:-1, :]
    image = unpack_raw(image)
    srgb = process_sidd_image(image, bayer_2by2, wb, cst2)
    return srgb


def process_sidd_image(image, bayer_pattern, wb, cst, *, save_file_rgb=None):
    """Simple processing pipeline"""

    image = flip_bayer(image, bayer_pattern)

    image = stack_rggb_channels(image)

    rgb2xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )

    rgb2cam = np.matmul(cst, rgb2xyz)

    cam2rgb = np.linalg.inv(rgb2cam)

    cam2rgb = cam2rgb / np.sum(cam2rgb, axis=-1, keepdims=True)
    image_srgb = process(image, 1 / wb[0][0], 1 / wb[0][1], 1 / wb[0][2], cam2rgb)

    image_srgb = image_srgb * 255.0
    image_srgb = image_srgb.astype(np.uint8)

    image_srgb = swap_channels(image_srgb)

    if save_file_rgb:
        # Save
        cv2.imwrite(save_file_rgb, image_srgb)

    return image_srgb


def flip_bayer(image, bayer_pattern):
    if (bayer_pattern == [[1, 2], [2, 3]]):
        pass
    elif (bayer_pattern == [[2, 1], [3, 2]]):
        image = np.fliplr(image)
    elif (bayer_pattern == [[2, 3], [1, 2]]):
        image = np.flipud(image)
    elif (bayer_pattern == [[3, 2], [2, 1]]):
        image = np.fliplr(image)
        image = np.flipud(image)
    else:
        import pdb
        pdb.set_trace()
        print('Unknown Bayer pattern.')
    return image


def stack_rggb_channels(raw_image):
    """Stack the four RGGB channels of a Bayer raw image along a third dimension"""
    height, width = raw_image.shape
    channels = []
    for yy in range(2):
        for xx in range(2):
            raw_image_c = raw_image[yy:height:2, xx:width:2].copy()
            channels.append(raw_image_c)
    channels = np.stack(channels, axis=-1)
    return channels


def process(bayer_images, red_gains, green_gains, blue_gains, cam2rgbs):
    bayer_images = apply_gains(bayer_images, red_gains, green_gains, blue_gains)
    bayer_images = np.clip(bayer_images, 0.0, 1.0)
    images = demosaic_CV2(bayer_images)
    images = apply_ccm(images, cam2rgbs)
    images = np.clip(images, 0.0, 1.0)
    images = gamma_compression(images)
    return images


def apply_gains(bayer_image, red_gains, green_gains, blue_gains):
    gains = np.stack([red_gains, green_gains, green_gains, blue_gains], axis=-1)
    gains = gains[np.newaxis, np.newaxis, :]
    return bayer_image * gains


def demosaic_CV2(rggb_channels_stack):
    # using opencv demosaic
    bayer = RGGB2Bayer(rggb_channels_stack)
    dem = cv2.cvtColor(np.clip(bayer * 16383, 0, 16383).astype(dtype=np.uint16), cv2.COLOR_BayerBG2RGB_EA)
    dem = dem.astype(dtype=np.float32) / 16383
    return dem


def RGGB2Bayer(im):
    # convert RGGB stacked image to one channel Bayer
    bayer = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    bayer[0::2, 0::2] = im[:, :, 0]
    bayer[0::2, 1::2] = im[:, :, 1]
    bayer[1::2, 0::2] = im[:, :, 2]
    bayer[1::2, 1::2] = im[:, :, 3]
    return bayer


def apply_ccm(image, ccm):
    images = image[:, :, np.newaxis, :]
    ccms = ccm[np.newaxis, np.newaxis, :, :]
    return np.sum(images * ccms, axis=-1)


def gamma_compression(images, gamma=2.2):
    return np.maximum(images, 1e-8) ** (1.0 / gamma)


def swap_channels(image):
    """Swap the order of channels: RGB --> BGR"""
    h, w, c = image.shape
    image1 = np.zeros(image.shape)
    for i in range(c):
        image1[:, :, i] = image[:, :, c - i - 1]
    return image1


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
