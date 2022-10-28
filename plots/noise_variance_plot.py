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
from torch.utils.data import DataLoader
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import json
import torch
import time
import os
from scipy.signal import savgol_filter

from data_loader.iterable_loader import IterableSIDDMediumDataset
from train_noise_flow import init_params
from data_loader.utils import hps_loader
from model.noise_flow import NoiseFlow

parser = argparse.ArgumentParser(description="plot")
parser.add_argument("--sidd_path", type=str, default='./data/SIDD_Medium_Srgb/Data', help="Location of the SIDD dataset")
parser.add_argument("--n_batch_test", type=int, default=1, help="Minibatch size")
parser.add_argument("--n_patches_per_image", type=int, default=2898)
parser.add_argument("--patch_sampling", type=str, default='uniform')
parser.add_argument("--camera", type=str)
parser.add_argument("--iso", type=int)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument('--is_raw', action='store_true', default=False)
args = parser.parse_args()

cam_mapping = ['IP', 'GP', 'S6', 'N6', 'G4']
cam_fullname = {
    'IP': 'iPhone 7',
    'GP': 'Pixel',
    'S6': 'Galaxy S6',
    'N6': 'Nexus 6',
    'G4': 'G4'
}
def load_checkpoint(model, checkpoint_dir):
	checkpoint = torch.load(checkpoint_dir)
	model.load_state_dict(checkpoint['state_dict'])
	return model, checkpoint['epoch_num']


def main_multiple_models(iso):
    assert args.camera is not None
    models = {
        'Real Samples': None,
        'Isotropic Gaussian': 'lt_testt',
        'Noise Flow': 'Srgb_all_date_real_gain_correct_order',
        'Our Model': 'srgb_all_data_8_cac_4_lt2',
    }
    var_summary_model = {}
    color_channel = 'Red'
    index = {
        'Red': 0,
        'Green': 1,
        'Blue': 2
    }
    for key in models.keys():
        args.model_path = models[key]
        var_summary = main(iso, plot=False, color_channel='Red')
        var_summary_model[key] = var_summary[args.camera][index[color_channel]]

    colors = ['purple', 'orange', 'green', 'red']
    x = [_ for _ in range (256)]
    window_length = 31
    polyorder = 3
    for model, color in zip(var_summary_model.keys(), colors):
        y = var_summary_model[model]
        yhat = savgol_filter(y, window_length, polyorder)
        plt.scatter(x, y, s=3, color=color)
        plt.plot(x, yhat, color=color, label=model, linewidth = 2 if model == 'Real Samples' else 1, zorder= 10 if model == 'Real Samples' else 1)

    plt.xlabel('Clean Image Intensity', fontsize=14)
    plt.ylabel('Noise Variance', fontsize=14)
    plt.ylim(-3, 250)
    plt.legend(prop={"size":12})
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.savefig('plots/motivation/multi_models_{}_{}_{}.pdf'.format(color_channel, args.camera, iso if iso else '-1'), pad_inches=0, bbox_inches='tight')
    plt.clf()

def main(iso, plot=True, color_channel=None):
    nf = None
    no_patch_size = 1700
    n_channels = 3 if not args.is_raw else 4
    exclude_iso = None
    if args.model_path:
        exclude_iso = [50, 200, 320, 500, 640, 1000, 6400, 10000, 2000]
        model_path = os.path.abspath(os.path.join('experiments', 'sidd', args.model_path))
        hps = hps_loader(os.path.join(model_path, 'hps.txt'))
        hps.param_inits = init_params()
        x_shape = [n_channels, no_patch_size, no_patch_size]
        nf = NoiseFlow(x_shape, hps.arch, hps.flow_permutation, hps.param_inits, hps.lu_decomp, hps.device, hps.raw)
        nf.to(hps.device)

        logdir = model_path + '/saved_models'
        models = sorted(os.listdir(hps.model_save_dir))
        last_epoch = str(max([int(i.split("_")[1]) for i in models[1:]]))
        saved_model_file_name = 'epoch_{}_nf_model_net.pth'.format(30 if 'srgb_all_data_8_cac_4_lt2' in hps.model_save_dir else last_epoch)
        saved_model_file_path = os.path.join(hps.model_save_dir, saved_model_file_name)

        nf, epoch = load_checkpoint(nf, saved_model_file_path)
        print('NF epoch is {}'.format(epoch))

    test_dataset = IterableSIDDMediumDataset(
        sidd_medium_path=args.sidd_path if not args.is_raw else './data/SIDD_Medium_Raw/Data',
        train_or_test='all',
        cam=args.camera,
        iso=iso,
        patch_size=(32, 32),
        is_raw=args.is_raw,
        num_patches_per_image=args.n_patches_per_image,
        last_im_idx=11,
        no_patching=True,
        no_patch_size=no_patch_size,
        model=nf,
        temp=hps.temp if nf else None,
        device=hps.device if nf else None,
        exclude_iso=exclude_iso
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.n_batch_test,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    args.n_tr_inst = test_dataset.cnt_inst
    print('# test scene instances (cam = {}, iso = {}) = {}'.format(
        str(args.camera), str(iso), args.n_tr_inst))

    if iso is None:
        iso = -1
    noise_summary = {
        'IP': [None] * 256,
        'S6': [None] * 256,
        'GP': [None] * 256,
        'N6': [None] * 256,
        'G4': [None] * 256
    }

    for k in noise_summary.keys():
        for i in range(256):
            if not args.is_raw:
                noise_summary[k][i] = [[], [], []]
            else: 
                noise_summary[k][i] = [[], [], [], []]

    total_time = time.time()
    total_idx_time = 0
    total_zip_time = 0
    total_extend_time = 0
    for n_batch, batch_images in enumerate(test_dataloader):
        clean = batch_images['clean']
        noisy = batch_images['noise']
        cam = cam_mapping[int(np.mean(np.array(batch_images['cam'])))]
        noise = noisy - clean
        batch_zip_time = 0
        for i in range(0, 256, 1):
            idx_time = time.time()
            if args.is_raw:
                j = i / 256
                r_idx = ((j < clean[:, 0, :, :]) == (clean[:, 0, :, :] < j + 1/256)).nonzero()
                total_idx_time += time.time() - idx_time
                g_idx = ((j < clean[:, 1, :, :]) == (clean[:, 1, :, :] < j + 1/256)).nonzero()
                g_idx_2 = ((j < clean[:, 2, :, :]) == (clean[:, 2, :, :] < j + 1/256)).nonzero()
                b_idx = ((j < clean[:, 3, :, :]) == (clean[:, 3, :, :] < j + 1/256)).nonzero()
            else:
                r_idx = (clean[:, 0, :, :] == i).nonzero()
                total_idx_time += time.time() - idx_time
                g_idx = (clean[:, 1, :, :] == i).nonzero()
                g_idx_2 = None
                b_idx = (clean[:, 2, :, :] == i).nonzero()

            if r_idx.shape[0] != 0 and (color_channel is None or color_channel == 'Red'):
                zip_time = time.time()
                aa = r_idx[:, 0]
                bb = r_idx[:, 1]
                cc = r_idx[:, 2]
                total_zip_time += time.time() - zip_time
                batch_zip_time += time.time() - zip_time
                extend_time = time.time()
                noise_summary[cam][i][0].extend(np.array(noise[aa, 0, bb, cc])) 
                total_extend_time += time.time() - extend_time

            if g_idx.shape[0] != 0 and (color_channel is None or color_channel == 'Green'):
                aa = g_idx[:, 0]
                bb = g_idx[:, 1]
                cc = g_idx[:, 2]
                noise_summary[cam][i][1].extend(np.array(noise[aa, 1, bb, cc]))

            if g_idx_2 is not None and g_idx_2.shape[0] != 0 and (color_channel is None or color_channel == 'Green'):
                aa = g_idx_2[:, 0]
                bb = g_idx_2[:, 1]
                cc = g_idx_2[:, 2]
                noise_summary[cam][i][2].extend(np.array(noise[aa, 2, bb, cc]))

            if b_idx.shape[0] != 0 and (color_channel is None or color_channel == 'Blue'):
                aa = b_idx[:, 0]
                bb = b_idx[:, 1]
                cc = b_idx[:, 2]
                channel = 2 if g_idx_2 is None else 3
                noise_summary[cam][i][channel].extend(np.array(noise[aa, channel, bb, cc])) 

    total_time = time.time() - total_time

    # with open('./plots/motivation/{}_noise_dara.json'.format(iso), 'w') as fp:
    #     json.dump(eval(str(noise_summary)), fp)

    # with open('./plots/motivation/{}_noise_dara.json'.format(iso), 'r') as fp:
    #     noise_summary = json.load(fp)

    var_time = time.time()
    if args.is_raw:
        var_summary = {
            'IP': [[], [], [], []],
            'S6': [[], [], [], []],
            'GP': [[], [], [], []],
            'N6': [[], [], [], []],
            'G4': [[], [], [], []]
        }
    else:
        var_summary = {
            'IP': [[], [], []],
            'S6': [[], [], []],
            'GP': [[], [], []],
            'N6': [[], [], []],
            'G4': [[], [], []]
        }
    for cam in cam_mapping:
        for c in range(n_channels):
            for i in range(256):
                noise = noise_summary[cam][i][c]
                if len(noise) != 0:
                    var = np.var(noise)
                else:
                    var = 0

                var_summary[cam][c].append(var)

    var_time = time.time() - var_time

    # with open('./plots/motivation/{}_var.json'.format(iso), 'w') as fp:
    #     json.dump(eval(str(var_summary)), fp)

    # with open('./plots/motivation/{}_var.json'.format(iso), 'r') as fp:
    #     var_summary = json.load(fp)


    plt_time = time.time()
    if plot:
        if args.is_raw:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, n_channels)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, n_channels)
        fig.set_figwidth(20)
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        x = [_ for _ in range (256)]
        if args.is_raw:
            x = [i / 256 for i in x]
        limits = {
            '100': [200, 200, 300],
            '400': [500, 500, 500],
            '800': [1000, 700, 1000],
            '1600': [1500, 1000, 2000],
            '3200': [3000, 2000, 3000],
            '-1': [1300, 1300, 1300]
        }
        window_length = 31
        polyorder = 3
        for cam, color in zip(cam_mapping, colors):
            y = var_summary[cam][0]
            yhat = savgol_filter(y, window_length, polyorder)
            ax1.scatter(x, y, s=3, color=color)
            ax1.plot(x, yhat, color=color, label=cam_fullname[cam])

            y = var_summary[cam][1]
            yhat = savgol_filter(y, window_length, polyorder) 
            ax2.scatter(x, y, s=3, color=color)
            ax2.plot(x, yhat, color=color, label=cam_fullname[cam])

            y = var_summary[cam][2]
            yhat = savgol_filter(y, window_length, polyorder)
            ax3.scatter(x, y, s=3, color=color)
            ax3.plot(x, yhat, color=color, label=cam_fullname[cam])

            if args.is_raw:
                y = var_summary[cam][3]
                yhat = savgol_filter(y, window_length, polyorder)
                ax4.scatter(x, y, s=3, color=color)
                ax4.plot(x, yhat, color=color, label=cam_fullname[cam])


        if not args.is_raw:
            ax1.set_ylim(bottom=0, top=limits[str(iso)][0])
            ax2.set_ylim(bottom=0, top=limits[str(iso)][1])
            ax3.set_ylim(bottom=0, top=limits[str(iso)][2])
        ax1.legend(prop={"size":12})
        ax1.set_title('Red', fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax2.legend(prop={"size":12})
        ax2.set_title('Green', fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        if args.is_raw:
            ax3.legend(prop={"size":12})
            ax3.set_title('Green', fontsize=16)
            ax3.tick_params(axis='both', which='major', labelsize=12)
            ax4.legend(prop={"size":12})
            ax4.set_title('Blue', fontsize=16)
            ax4.tick_params(axis='both', which='major', labelsize=12)

        else:
            ax3.legend(prop={"size":12})
            ax3.set_title('Blue', fontsize=16)
            ax3.tick_params(axis='both', which='major', labelsize=12)

        ax1.set_xlabel('Clean Image Intensity', fontsize=14)
        ax1.set_ylabel('Noise Variance', fontsize=14)
        ax2.set_xlabel('Clean Image Intensity', fontsize=14)
        ax2.set_ylabel('Noise Variance', fontsize=14)
        ax3.set_xlabel('Clean Image Intensity', fontsize=14)
        ax3.set_ylabel('Noise Variance', fontsize=14)
        if args.is_raw:
            ax4.set_xlabel('Clean Image Intensity', fontsize=14)
            ax4.set_ylabel('Noise Variance', fontsize=14)

        plt.savefig('plots/motivation/{}_var_window_length_{}_order_{}_{}_{}.pdf'.format(
            iso,
            window_length,
            polyorder,
            args.model_path if args.model_path else 'real',
            args.is_raw
        ), pad_inches=0, bbox_inches='tight')
        plt.clf()
    plt_time = time.time() - plt_time
    print(total_time, var_time, plt_time)
    return var_summary

def plot_iso_cam(is_std):
    with open('./plots/motivation/cam_{}.json'.format('std' if is_std else 'mean'), 'r') as fp:
        plot_data = json.load(fp)

    plt.figure().set_figwidth(7)
    for cam in cam_mapping:
        x = [100, 400, 800, 1600, 3200]
        y = plot_data[cam]
        y = [np.nan if i == 0 else i for i in y]
        plt.plot(x, y, '*', linestyle='-', label=cam_fullname[cam])
        plt.xticks(x, x)

    plt.xlabel('Camera ISO', fontsize=14)
    plt.ylabel('Noise Standard Deviation' if is_std else 'Noise Mean', fontsize=14)
    plt.legend(prop={"size":12})
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.savefig('plots/motivation/cam_{}.pdf'.format('std' if is_std else 'mean'), pad_inches=0, bbox_inches='tight')

def plot_nll_kld(fig_type):
    models = [
        '/home/shayanko/repos/NoiseFlow_PyTorch/experiments/sidd/srgb_all_data_8_cac_4_lt2',
        '/home/shayanko/repos/NoiseFlow_PyTorch/experiments/sidd/lt_testt',
        '/home/shayanko/repos/NoiseFlow_PyTorch/experiments/sidd/lt2_testt2',
        '/home/shayanko/repos/NoiseFlow_PyTorch/experiments/sidd/cond_c1x1_testt',
        '/home/shayanko/repos/NoiseFlow_PyTorch/experiments/sidd/heteroscedastic_gaussian_conditioned',
        '/home/shayanko/repos/NoiseFlow_PyTorch/experiments/sidd/Srgb_all_date_real_gain_correct_order'
    ]
    model_names= [
        'Our Model',
        'Isotropic Gaussian',
        'Diagonal Gaussian',
        'Full Gaussian',
        'Heteroscedastic (NLF)',
        'Noise Flow'
    ]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']
    if fig_type == 'kld':
        mode = ['all_kld']
    else:
        mode = ['Test']
    for color, model, name in zip(colors, models, model_names):
        for m in mode:
            y = []
            with open(model + '/{}.txt'.format(m.lower())) as f:
                lines = f.readlines()
                if fig_type == 'kld':
                    lines = json.loads(lines[0])
                count = 0
                for line in lines:
                    if count >= 1:
                        epoch = int(line.split('\t')[0]) if fig_type == 'nll' else count
                        lim = 250 if fig_type == 'nll' else 200
                        if epoch > lim:
                            break
                        if fig_type == 'kld':
                            y.append(line)
                        else:
                            y.append(float(line.split('\t')[1])/(32*32*3))

                    count += 1

            x = [_ + 1 for _ in range(len(y))]
            x_test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            x_test.extend([_ for _ in range(10, lim+1, 10)])

            label = name
            plt.plot(x if m == 'Train' else x_test, y, color=color, linestyle='--' if m == 'Train' else '-', label=label)

    plt.ylabel(r'Marginal $D_{KL}$' if fig_type == 'kld' else'NLL per Dimension', fontsize=18)
    top = 1.0 if fig_type == 'kld' else 6.0
    bottom = .0 if fig_type == 'kld' else 2.8
    plt.ylim(bottom, top)
    # plt.axes().set_ylim(top=)
    plt.xlabel('Epoch', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(prop={"size":13})
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig('./plots/{}_fig.pdf'.format(fig_type), pad_inches=0, bbox_inches='tight')

def plot_iso_cam_epoch(is_cam):
    with open('./plots/iso_cam_epoch/iso_cam_epoch.json', 'r') as fp:
        plot_data = json.load(fp)

    colors = ['blue', 'orange', 'green', 'red', 'purple']
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x.extend([_ for _ in range(10, 201, 10)])

    if is_cam:
        plot_data = plot_data['CAM']
        keys = ['IP', 'GP', 'S6', 'N6', 'G4']
        labels = ['iPhone 7', 'Pixel', 'Galaxy S6', 'Nexus 6', 'G4']
        true_values = [9.201781, 5.294486, 18.826666, 10.316835, 10.458241]
    else:
        plot_data = plot_data['ISO']
        keys = ['100', '400', '800', '1600', '3200']
        labels = ['100', '400', '800', '1600', '3200']
        true_values = [5.6360874, 11.432604, 13.323198, 19.284262, 36.22865]

    
    for key, label, color, tv in zip(keys, labels, colors, true_values):
        y = plot_data[key][1:]

        plt.plot(x, y, color=color, linestyle='-', label=label)
        plt.plot(x, [tv] * len(y), color=color, linestyle='--')


    plt.ylabel('Noise Standard Deviation', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(prop={"size":13})
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig('./plots/{}_epoch.pdf'.format('cam' if is_cam else 'iso'), pad_inches=0, bbox_inches='tight')


if __name__ == "__main__":
    # for iso in [100, 400, 800, 1600, 3200]:
    # for iso in [3200]:
    #     main(iso)
    # main(None)

    main_multiple_models(100)

    # plot_iso_cam(False)
    # plot_iso_cam(True)


    # plot_nll_kld('nll')
    # plot_nll_kld('kld')

    # plot_iso_cam_epoch(True)
    # plot_iso_cam_epoch(False)

# python -m plots.noise_variance_plot