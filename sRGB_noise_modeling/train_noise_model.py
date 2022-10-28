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
import logging
import os
import shutil
import queue
import socket
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../')
from utils.arg_parser import arg_parser
from data_loader.loader import (check_download_sidd, SIDDMediumDataset)
from data_loader.iterable_loader import IterableSIDDMediumDataset
from data_loader.utils import calc_train_test_stats, get_its, ResultLogger, hps_logger
from utils.mylogger import add_logging_level
from utils.patch_stats_calculator import PatchStatsCalculator
from noise_model import NoiseModel
from data_loader.sidd_utils import calc_kldiv_mb

def save_checkpoint(model, optimizer, grad_scaler, epoch_num, checkpoint_dir):
    checkpoint = {
        'epoch_num' : epoch_num,
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'grad_scaler': grad_scaler.state_dict()
        }
    torch.save(checkpoint, checkpoint_dir)

def load_checkpoint(model, optimizer, grad_scaler, checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    grad_scaler.load_state_dict(checkpoint['grad_scaler'])
    return model, optimizer, grad_scaler, checkpoint['epoch_num']

def init_params():
    npcam = 3
    c_i = 1.0
    beta1_i = -5.0 / c_i
    beta2_i = 0.0
    gain_params_i = np.ndarray([5])
    gain_params_i[:] = -5.0 / c_i
    cam_params_i = np.ndarray([npcam, 5])
    cam_params_i[:, :] = 1.0

    return {
        "c_i": c_i,
        "beta1_i": beta1_i,
        "beta2_i": beta2_i,
        "gain_params_i": gain_params_i,
        "cam_params_i": cam_params_i
    }

def main(hps):
    check_download_sidd(hps.sidd_path)

    total_time = time.time()
    host = socket.gethostname()
    torch.random.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    # set up a custom logger    
    add_logging_level('TRACE', 100)
    logging.getLogger(__name__).setLevel("TRACE")
    logging.basicConfig(level=logging.TRACE)

    hps.n_bins = 2. ** hps.n_bits_x

    logging.trace('SIDD path = %s' % hps.sidd_path)
    logging.trace('Num GPUs Available: %s' % torch.cuda.device_count())
    hps.device = 'cuda' if torch.cuda.device_count() else 'cpu'

    # output log dir
    logdir = os.path.abspath(os.path.join('experiments', 'sidd', hps.logdir)) + '/'
    
    if hps.no_resume:
        if os.path.exists(logdir):
            shutil.rmtree(logdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    hps.logdirname = hps.logdir
    hps.logdir = logdir

    medium_sidd_path = hps.sidd_path
    if hps.dataset_type == 'full':
        path = hps.sidd_path.split("/")
        path[-2] = "SIDD_Medium_Raw"
        medium_sidd_path = "/".join(path)

    hps.num_workers = 5
    hps.raw = 'raw' in hps.sidd_path.lower() and hps.n_channels == 4
    train_dataset = IterableSIDDMediumDataset(
        sidd_medium_path=hps.sidd_path,
        train_or_test='train',
        cam=hps.camera,
        iso=hps.iso,
        patch_size=(hps.patch_height, hps.patch_height),
        is_raw=hps.raw,
        num_patches_per_image=hps.n_patches_per_image
    )
    train_dataloader = DataLoader(train_dataset, batch_size=hps.n_batch_train, shuffle=False, num_workers=hps.num_workers, pin_memory=True)
    hps.n_tr_inst = train_dataset.cnt_inst
    logging.trace('# training scene instances (cam = {}, iso = {}) = {}'.format(
        str(hps.camera), str(hps.iso), hps.n_tr_inst))
    test_dataset = IterableSIDDMediumDataset(
        sidd_medium_path=hps.sidd_path,
        train_or_test='test',
        cam=hps.camera,
        iso=hps.iso,
        patch_size=(hps.patch_height, hps.patch_height),
        is_raw=hps.raw,
        num_patches_per_image=hps.n_patches_per_image
    )
    test_dataloader = DataLoader(test_dataset, batch_size=hps.n_batch_test, shuffle=False, num_workers=hps.num_workers, pin_memory=True)
    hps.n_ts_inst = test_dataset.cnt_inst
    logging.trace('# testing scene instances (cam = {}, iso = {}) = {}'.format(
        str(hps.camera), str(hps.iso), hps.n_ts_inst))

    input_key_name = 'noise' if hps.raw else 'noisy'
    x_shape = next(iter(train_dataloader))[input_key_name].shape
    hps.x_shape = x_shape
    hps.n_dims = np.prod(x_shape[1:])

    # calculate data stats and baselines
    if hps.raw:
        logging.trace('calculating data stats and baselines...')
        pat_stats_calculator = PatchStatsCalculator(train_dataloader, x_shape[-1], n_channels=hps.n_channels,
                                                        save_dir=logdir, file_postfix='')
        pat_stats = pat_stats_calculator.calc_stats()
        nll_gauss, nll_sdn = pat_stats_calculator.calc_baselines(test_dataloader)
    else:
        nll_gauss = 0
        nll_sdn = 0

    # Noise model
    logging.trace('Building Noise Model...')
    hps.tensorboard_save_dir = "./tensorboard_data_6/{}".format(hps.logdir.split('/')[-2])
    hps.model_save_dir = os.path.join(hps.logdir, 'saved_models')
    
    hps.param_inits = init_params()
    nm = NoiseModel(x_shape[1:], hps.arch, hps.flow_permutation, hps.param_inits, hps.lu_decomp, hps.device, hps.raw)
    nm.to(hps.device)

    autocast_enable = True
    optimizer = torch.optim.Adam(nm.parameters(), lr=hps.lr, betas=(0.9, 0.999), eps=1e-08)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=autocast_enable)
    hps.num_params = np.sum([np.prod(params.shape) for params in nm.parameters()])
    logging.trace('number of parameters = {}'.format(hps.num_params))
    hps.temp = 1.0
    hps_logger(hps.logdir + 'hps.txt', hps, nm.get_layer_names(), hps.num_params)

    start_epoch = 1
    if not os.path.exists(hps.model_save_dir): # no resume
        os.makedirs(hps.model_save_dir)
        if not os.path.exists(hps.tensorboard_save_dir):
            os.makedirs(hps.tensorboard_save_dir)
    else: # resume mode
        models = sorted(os.listdir(hps.model_save_dir))
        last_epoch = str(max([int(i.split("_")[1]) for i in models[1:]]))
        saved_model_file_name = 'epoch_{}_noise_model_net.pth'.format(last_epoch)
        saved_model_file_path = os.path.join(hps.model_save_dir, saved_model_file_name)

        nm, optimizer, grad_scaler, start_epoch = load_checkpoint(nm, optimizer, grad_scaler, saved_model_file_path)
        start_epoch += 1
        logging.trace('found an existing previous checkpoint, resuming from epoch {}'.format(start_epoch))

    logging.trace('Logging to ' + logdir)
    # NLL: negative log likelihood
    # NLL_G: for Gaussian baseline
    # NLL_SDN: for camera NLF baseline
    # sdz: standard deviation of the base measure (sanity check)
    log_columns = ['epoch', 'NLL', 'sdz']
    kld_columns = ['KLD_NM', 'KLD_R']
    if hps.raw:
        log_columns.extend(['NLL_G', 'NLL_SDN'])
        kld_columns.extend(['KLD_G', 'KLD_NLF'])
    train_logger = ResultLogger(logdir + 'train.txt', log_columns + ['train_time'], start_epoch > 1)
    test_logger = ResultLogger(logdir + 'test.txt', log_columns + ['msg'], start_epoch > 1)
    sample_logger = ResultLogger(logdir + 'sample.txt', log_columns + ['sample_time'] + kld_columns, start_epoch > 1)

    writer = SummaryWriter(hps.tensorboard_save_dir)
    test_loss_best = np.inf
    for epoch in range(start_epoch, hps.epochs):
        validation = epoch < 10 or (epoch < 100 and epoch % 10 == 0) or epoch % hps.epochs_full_valid == 0.
        is_best = 0

        # Training loop
        train_loss = []
        train_sdz = []
        train_curr_time = time.time()
        nm.train()
        for n_batch, batch_images in enumerate(train_dataloader):
            optimizer.zero_grad()
            step = (epoch - 1) * (train_dataset.__len__()/hps.n_batch_train) + n_batch
            kwargs = {
                'x': batch_images[input_key_name].to(hps.device),
                'clean': batch_images['clean'].to(hps.device),
                'iso': batch_images['iso'].to(hps.device),
                'cam': batch_images['cam'].to(hps.device),
                # 'writer': writer,
                # 'step': step
            }
            if 'nlf0' in batch_images.keys():
                kwargs.update({
                    'nlf0': batch_images['nlf0'].to(hps.device),
                    'nlf1': batch_images['nlf1'].to(hps.device)
                })
            
            with torch.cuda.amp.autocast(enabled=autocast_enable):
                nll, sd_z = nm.loss(**kwargs)
            train_loss.append(nll.item())
            train_sdz.append(sd_z.item())
            writer.add_scalar(
                'NLL_per_dim',
                train_loss[-1] / np.prod(batch_images[input_key_name].shape[1:]),
                step
            )

            grad_scaler.scale(nll).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        train_loss_mean = np.mean(train_loss)
        train_time = time.time() - train_curr_time
        log_content = {
            'epoch': epoch,
            'train_time': int(train_time),
            'NLL': train_loss_mean,
            'sdz': np.mean(train_sdz)
        }
        if hps.raw:
            log_content.update({
                'NLL_G': nll_gauss,
                'NLL_SDN': nll_sdn
            })
        train_logger.log(log_content)

        # Testing loop
        if validation:
            test_loss = []
            test_sdz = []
            test_curr_time = time.time()
            nm.eval()
            with torch.no_grad():
                for n_batch, batch_images in enumerate(test_dataloader):
                    step = (epoch - 1) * (test_dataset.__len__()/hps.n_batch_test) + n_batch
                    kwargs = {
                        'x': batch_images[input_key_name].to(hps.device),
                        'clean': batch_images['clean'].to(hps.device),
                        'iso': batch_images['iso'].to(hps.device),
                        'cam': batch_images['cam'].to(hps.device)
                    }
                    if 'nlf0' in batch_images.keys():
                        kwargs.update({
                            'nlf0': batch_images['nlf0'].to(hps.device),
                            'nlf1': batch_images['nlf1'].to(hps.device)
                        })
                    with torch.cuda.amp.autocast(enabled=autocast_enable):
                        nll, sd_z = nm.loss(**kwargs)
                    test_loss.append(nll.item())
                    test_sdz.append(sd_z.item())
                    writer.add_scalar(
                        'Validation NLL per dim',
                        test_loss[-1] / np.prod(batch_images[input_key_name].shape[1:]),
                        step
                    )
            test_loss_mean = np.mean(test_loss)

            save_checkpoint(
                nm,
                optimizer,
                grad_scaler,
                epoch,
                os.path.join(hps.model_save_dir, 'epoch_{}_noise_model_net.pth'.format(epoch))
            )

            if test_loss_mean < test_loss_best:
                test_loss_best = test_loss_mean
                save_checkpoint(nm, optimizer, grad_scaler, epoch, os.path.join(hps.model_save_dir, 'best_model.pth'))
                is_best = 1

            log_content = {
                'epoch': epoch,
                'NLL': test_loss_mean,
                'sdz': np.mean(test_sdz),
                'msg': is_best
            }
            if hps.raw:
                log_content.update({
                    'NLL_G': nll_gauss,
                    'NLL_SDN': nll_sdn
                })
            test_logger.log(log_content)
            test_time = time.time() - test_curr_time

        # Sampling
        sample_loss = []
        sample_sdz = []
        sample_time = 0
        n_models = 4 if hps.raw else 2
        kldiv = np.zeros(n_models)
        sample_loss_mean = 0
        if hps.do_sample and validation:
            is_fix = False  # to fix the camera and ISO
            iso_vals = [100, 400, 800, 1600, 3200]
            iso_fix = [100]
            cam_fix = [['IP', 'GP', 'S6', 'N6', 'G4'].index('S6')]
            nlf_s6 = [[0.000479, 0.000002], [0.001774, 0.000002], [0.003696, 0.000002], [0.008211, 0.000002],
                    [0.019930, 0.000002]]
            sample_curr_time = time.time()
            nm.eval()
            with torch.no_grad():
                count = 0
                for n_batch, batch_images in enumerate(test_dataloader):
                    step = (epoch - 1) * (test_dataset.__len__()/hps.n_batch_test) + n_batch
                    count +=1
                    kwargs = {
                        'clean': batch_images['clean'].to(hps.device),
                        'eps_std': torch.tensor(hps.temp, device=hps.device),
                        # 'writer': writer,
                        # 'step': step
                    }
                    if is_fix:
                        kwargs.update({
                            'iso': iso_fix[0].to(hps.device),
                            'cam': cam_fix[0].to(hps.device),
                            'nlf0': [nlf_s6[iso_vals.index(iso_fix[0])][0]],
                            'nlf1': [nlf_s6[iso_vals.index(iso_fix[0])][0]]
                        })
                    else:
                        kwargs.update({
                            'iso': batch_images['iso'].to(hps.device),
                            'cam': batch_images['cam'].to(hps.device)
                        })

                        if 'nlf0' in batch_images.keys():
                            kwargs.update({
                                'nlf0': batch_images['nlf0'].to(hps.device),
                                'nlf1': batch_images['nlf1'].to(hps.device)
                            })

                    x_sample_val = nm.sample(**kwargs)
                    kwargs.update({'x': x_sample_val})
                    kwargs.pop('eps_std')
                    # kwargs.pop('writer')
                    # kwargs.pop('step')
                    with torch.cuda.amp.autocast(enabled=autocast_enable):
                        nll, sd_z = nm.loss(**kwargs)
                    sample_loss.append(nll.item())
                    sample_sdz.append(sd_z.item())

                    # marginal KL divergence
                    vis_mbs_dir = os.path.join(hps.logdir, 'samples_epoch_%04d' % epoch, 'samples_%.1f' % hps.temp)
                    kldiv_batch, cnt_batch = calc_kldiv_mb(
                        batch_images,
                        x_sample_val.data.to('cpu'),
                        vis_mbs_dir,
                        pat_stats['sc_in_sd'] if hps.raw else None,
                        n_models,
                        is_raw=hps.raw
                    )
                    kldiv += kldiv_batch / cnt_batch

            sample_loss_mean = np.mean(sample_loss)
            kldiv /= count
            kldiv = list(kldiv)
            sample_time = time.time() - sample_curr_time
            
            log_content ={
                'epoch': epoch,
                'NLL': sample_loss_mean,
                'sdz': np.mean(sample_sdz),
                'sample_time': sample_time
            }
            if hps.raw:
                log_content.update({
                    'NLL_G': nll_gauss,
                    'NLL_SDN': nll_sdn,
                    'KLD_NLF': kldiv.pop(1),
                    'KLD_G': kldiv.pop(0)
                })
            log_content.update({
                'KLD_NM': kldiv[0],
                'KLD_R': kldiv[1]
            })
            sample_logger.log(log_content)

        if validation:
            print("{}, epoch: {}, tr_loss: {:.1f}, ts_loss: {:.1f}, sm_loss: {:.1f}, tr_time: {:.2f}, ts_time: {:.2f}, " \
                "sm_time: {:.2f}, T_time: {:.2f}, best:{}, kld:{:.3f}".format(
                hps.logdirname,
                epoch,
                train_loss_mean,
                test_loss_mean,
                sample_loss_mean,
                train_time,
                test_time,
                sample_time,
                train_time + test_time + sample_time,
                is_best,
                kldiv[0]
            ))
        
    writer.close()

    total_time = time.time() - total_time
    logging.trace('Total time = %f' % total_time)
    logging.trace("Finished!")

if __name__ == "__main__":
    hps = arg_parser()
    main(hps)