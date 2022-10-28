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
import sys
sys.path.append('../')
from data_loader.loader import SIDDMediumDataset
from data_loader.sidd_utils import pack_raw, unpack_raw
from torch.utils.data import Dataset, DataLoader
from data_loader.utils import hps_loader, ResultLogger
from model.dncnn import DnCNN
from train_noise_model import init_params
from noise_model import NoiseModel
from utils.train_utils import weights_init_orthogonal, mean_psnr, batch_PSNR, batch_SSIM
import torch.nn as nn
import torch
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter
import torchvision.utils as utils
import os
import random
import urllib.request
from scipy.io import loadmat, savemat
import numpy as np
import h5py
import cv2
import time

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--num_workers", type=int, default=5, help="number of workers used by dataloader")
parser.add_argument("--batch_size", type=int, default=138, help="Training batch size")
parser.add_argument("--sidd_path", type=str, help='Path to SIDD Medium Raw/SRGB dataset')
parser.add_argument("--depth", type=int, default=9, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--log_dir", type=str, default='logs', help='Path to where you want to store logs')
parser.add_argument("--iso", type=int, default=None, help="ISO level for dataset images")
parser.add_argument("--cam", type=str, default=None, help="CAM type for dataset images")
parser.add_argument('--model', type=str, default='DnCNN_Real', help='choose a type of model')
parser.add_argument('--noise_model_path', type=str, help='path to a noise model')
parser.add_argument('--exp_name', default='')
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument('--data_split', type=int, default=0, help='0 == sidd train-validation split, 1 == noise flow split')
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--save_denoised', action='store_true', default=False)
parser.add_argument("--nm_load_epoch", type=int, default=None, help="noise model epoch to be loaded")
args = parser.parse_args()

torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

def download_url_to_file(url, file):
	file_data = urllib.request.urlopen(url)
	data_to_write = file_data.read()
	with open(file, 'wb') as f:
		f.write(data_to_write)

def denoise_raw_test_data(noisy):
	n_pt = noisy.shape[0]
	denoised = np.zeros(noisy.shape)
	for p in range(n_pt):
		noisy_patch = np.squeeze(noisy[p, :, :])
		noisy_patch = pack_raw(noisy_patch.cpu())
		noisy_patch = noisy_patch[np.newaxis, :, :, :]
		noisy_patch = np.transpose(noisy_patch, (0, 3, 1, 2))
		noisy_patch = torch.tensor(noisy_patch).cuda()
		denoised_patch = model(noisy_patch)
		denoised_patch = np.transpose(denoised_patch.cpu(), (0, 2, 3, 1))
		denoised_patch = unpack_raw(np.squeeze(denoised_patch))
		denoised[p, :, :] = denoised_patch

	denoised = torch.tensor(denoised).cuda()

	return denoised

def get_test_data(val_or_test='validation', is_raw=True):
	if val_or_test == 'validation':
		noisy_mat_file = 'ValidationNoisyBlocksRaw' if is_raw else 'ValidationNoisyBlocksSrgb'
		ref_mat_file = 'ValidationGtBlocksRaw' if is_raw else 'ValidationGtBlocksSrgb'
	else:
		noisy_mat_file = 'BenchmarkNoisyBlocksRaw' if is_raw else 'BenchmarkNoisyBlocksSrgb'
		ref_mat_file = None

	noisy_mat_path = os.path.join('../data', '{}.mat'.format(noisy_mat_file))
	ref_mat_path = os.path.join('../data', '{}.mat'.format(ref_mat_file))

	# download?
	if not os.path.exists(noisy_mat_path):
		noisy_mat_url = 'ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/{}.mat'.format(noisy_mat_file)
		print('downloading ' + noisy_mat_url)
		print('to ' + ref_mat_path)
		download_url_to_file(noisy_mat_url, noisy_mat_path)
	if ref_mat_file and not os.path.exists(ref_mat_path):
		ref_mat_url = 'ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/{}.mat'.format(ref_mat_file)
		print('downloading ' + ref_mat_url)
		print('to ' + ref_mat_path)
		download_url_to_file(ref_mat_url, ref_mat_path)

	noisy_mat1 = loadmat(noisy_mat_path)['{}'.format(noisy_mat_file)]
	ref_mat1 = loadmat(ref_mat_path)['{}'.format(ref_mat_file)] if ref_mat_file else None
	data_type = noisy_mat1.dtype

	# if val_or_test == 'validation':
	# 	exc_iso = [1, 3, 5, 7, 10, 11, 13, 14, 15, 18, 19, 20, 23, 24, 25, 28, 31, 33, 35, 38]
	# 	noisy_mat1 = np.delete(noisy_mat1, exc_iso, axis=0)
	# 	ref_mat1 = np.delete(ref_mat1, exc_iso, axis=0)

	if not is_raw:
		noisy_mat1 = np.transpose(noisy_mat1, (0, 1, 4, 2, 3))
		ref_mat1 = np.transpose(ref_mat1, (0, 1, 4, 2, 3)) if ref_mat_file else None

	noisy_mat1 = torch.tensor(noisy_mat1).to(torch.float)
	ref_mat1 = torch.tensor(ref_mat1).to(torch.float) if ref_mat_file else None

	return noisy_mat1, ref_mat1, data_type

def save_checkpoint(model, optimizer, epoch_num, checkpoint_dir):
	checkpoint = {'epoch_num' : epoch_num, 'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
	torch.save(checkpoint, checkpoint_dir)

def load_checkpoint(model, checkpoint_dir):
	checkpoint = torch.load(checkpoint_dir)
	model.load_state_dict(checkpoint['state_dict'])
	return model, checkpoint['epoch_num']

def save_denoised_patches(path, denoised, gt, noisy, psnrs):
	if not os.path.exists(path):	
		os.makedirs(path, exist_ok=True)

	for i in range(len(psnrs)):
		for j in range(psnrs[i].shape[0]):
			cv2.imwrite(path + '/{}_denoised.png'.format(psnrs[i][j]), np.array(denoised[i][j].cpu()).transpose((1, 2, 0)))
			cv2.imwrite(path + '/{}_gt.png'.format(psnrs[i][j]), np.array(gt[i][j].cpu()).transpose((1, 2, 0)))
			cv2.imwrite(path + '/{}_noisy_real.png'.format(psnrs[i][j]), np.array(noisy[i][j].cpu()).transpose((1, 2, 0)))
	
def lr_schedule(epoch):
	initial_lr = args.lr
	if epoch <= 30:
		lr = initial_lr
	elif epoch <= 60:
		lr = initial_lr / 10
	elif epoch <= 80:
		lr = initial_lr / 20
	else:
		lr = initial_lr / 20
	return lr

min_est_sigma = 0.24186
max_est_sigma = 11.507

args.is_raw = 'raw' in args.sidd_path.lower()
args.n_channels = 4 if args.is_raw else 3
data_range = 1. if args.is_raw else 255.
x_shape = [4, 32, 32] if args.is_raw else [3, 32, 32]

nm = None # Noise model
if args.model.__contains__('DnCNN_NM'):
	nm_path = os.path.abspath(os.path.join('experiments', 'sidd', args.noise_model_path))
	hps = hps_loader(os.path.join(nm_path, 'hps.txt'))
	hps.param_inits = init_params()
	nm = NoiseModel(x_shape, hps.arch, hps.flow_permutation, hps.param_inits, hps.lu_decomp, hps.device, hps.raw)
	nm.to(hps.device)

	logdir = nm_path + '/saved_models'
	models = sorted(os.listdir(hps.model_save_dir))
	if args.nm_load_epoch:
		last_epoch = args.nm_load_epoch
	else:
		last_epoch = str(max([int(i.split("_")[1]) for i in models[1:]]))
	saved_model_file_name = 'epoch_{}_noise_model_net.pth'.format(last_epoch)
	saved_model_file_path = os.path.join(hps.model_save_dir, saved_model_file_name)

	nm, nm_epoch = load_checkpoint(nm, saved_model_file_path)
	print('Noise model epoch is {}'.format(nm_epoch))

train_dataset = SIDDMediumDataset(
	sidd_medium_path=args.sidd_path,
	patch_sampling='dncnn',
	train_or_test='train_dncnn' if args.data_split == 0 else 'train',
	cam=args.cam,
	iso=args.iso,
	is_raw=args.is_raw,
	first_im_idx=10,
	last_im_idx=11,
	model=nm,
	temp=hps.temp if nm else None,
	device=hps.device if nm else None
)

loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
print('Number of training samples: {}'.format(len(train_dataset.file_names_tuple)))

if args.data_split == 1:
	val_dataset = SIDDMediumDataset(
		sidd_medium_path=args.sidd_path,
		patch_sampling='dncnn',
		train_or_test='val',
		cam=args.cam,
		iso=args.iso,
		is_raw=args.is_raw,
		first_im_idx=10,
		last_im_idx=11,
	)

	loader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	print('Number of validation samples: {}'.format(len(val_dataset.file_names_tuple)))

model = DnCNN(channels=args.n_channels, num_of_layers=args.depth, features=32)
model.apply(weights_init_orthogonal)
criterion = nn.MSELoss(reduction='sum')
model.cuda()
criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

args.exp_name = '{}_data_split_{}'.format(args.exp_name, args.data_split)
args.log_dir = os.path.join(args.log_dir, args.model, args.exp_name)
print(args.log_dir)
args.model_save_dir = os.path.join(args.log_dir, 'saved_models')
if not os.path.exists(args.log_dir):
	os.makedirs(args.log_dir)
if not os.path.exists(args.model_save_dir):
	os.makedirs(args.model_save_dir)

start_epoch = 1

if args.resume:
	with open(os.path.join(args.log_dir, 'val.txt')) as f:
		lines = f.readlines()
		saved_model_file_name = 'epoch_{}_dncnn_model.pth'.format(int(lines[-1].split('\t')[-3])//10 *10 )
		saved_model_file_path = os.path.join(args.model_save_dir, saved_model_file_name)

	model, start_epoch = load_checkpoint(model, saved_model_file_path)
	start_epoch += 1
val_logger = ResultLogger(args.log_dir + '/val.txt', ['epoch', 'PSNR', 'SSIM', 'best_epoch', 'best_PSNR', 'best_SSIM'], start_epoch > 1)
train_logger = ResultLogger(args.log_dir + '/train.txt', ['epoch', 'PSNR', ], start_epoch > 1)

writer = SummaryWriter(args.log_dir)
# writer.add_graph(model, next(iter(loader_train))['noise'].cuda())

step = 0
best_psnr_val = 0
best_ssim_val = 0
best_epoch_val = 0
input_key_name = 'noise' if args.is_raw else 'noisy'
for epoch in range(start_epoch, args.epochs):
	current_lr = lr_schedule(epoch)
	# set learning rate
	for param_group in optimizer.param_groups:
		param_group["lr"] = current_lr
	print('learning rate %f' % current_lr)

	# train
	model.train()
	train_curr_time = time.time()
	loss_train = 0
	train_cntr = 0
	psnr_train = 0
	for i, data in enumerate(loader_train, 0):
		# training step
		model.zero_grad()
		optimizer.zero_grad()
		input_image, clean = data[input_key_name], data['clean']

		if args.is_raw:
			noise, clean = input_image.cuda(), clean.cuda()
			noisy = clean + noise
		else:
			clean = clean.cuda()
			noisy = input_image.cuda()

		denoised = model(noisy)

		loss = criterion(denoised, clean) / (noisy.size()[0]*2)
		loss.backward()
		optimizer.step()
		loss_train += loss

		# results
		denoised = torch.clamp(denoised, 0., data_range)
		psnr_train += batch_PSNR(denoised, clean, data_range)
		train_cntr += 1
		# writer.add_scalar('train loss', loss.item(), step)
		# writer.add_scalar('train PSNR', psnr_train, step)
		step += 1

	loss_train /= train_cntr
	psnr_train /= train_cntr
	writer.add_scalar('train epoch loss', loss_train, epoch)
	train_logger.log({
		'epoch': epoch,
		'loss': loss_train,
		'PSNR': psnr_train
	})
	time_train = time.time() - train_curr_time

	# validate
	model.eval()
	val_curr_time = time.time()
	loss_val = 0
	val_cntr = 0
	psnr_val = 0	
	ssim_val = 0
	run_test = False
	with torch.no_grad():
		all_noisy = []
		all_clean = []
		all_denoised = []
		all_psnrs = []

		if args.data_split == 0:
			noisy_mat, ref_mat, _ = get_test_data('validation', args.is_raw)
			for i in range(noisy_mat.shape[0]):
				noisy = noisy_mat[i].cuda()
				clean = ref_mat[i].cuda()

				if args.is_raw:
					denoised = denoise_raw_test_data(noisy)
				else:
					denoised = model(noisy)
				loss = criterion(denoised, clean) / (noisy.size()[0]*2)
				denoised = torch.clamp(denoised, 0., data_range)

				loss_val += loss.item()
				mean_psnr_val, psnrs = mean_psnr(denoised, clean, data_range)
				psnr_val += mean_psnr_val
				ssim_val += batch_SSIM(denoised, clean, data_range)
				val_cntr += 1

				all_noisy.append(noisy)
				all_clean.append(clean)
				all_denoised.append(denoised)
				all_psnrs.append(psnrs)

		else:
			for i, data in enumerate(loader_val, 0):
				noisy, clean = data[input_key_name].cuda(), data['clean'].cuda()

				if args.is_raw:
					denoised = denoise_raw_test_data(noisy)
				else:
					denoised = model(noisy)
				loss = criterion(denoised, clean) / (noisy.size()[0]*2)
				denoised = torch.clamp(denoised, 0., data_range)

				loss_val += loss.item()
				mean_psnr_val, psnrs = mean_psnr(denoised, clean, data_range)
				psnr_val += mean_psnr_val
				ssim_val += batch_SSIM(denoised, clean, data_range)
				val_cntr += 1

				all_noisey.append(noisy)
				all_clean.append(clean)
				all_denoised.append(denoised)
				all_psnrs.append(psnrs)

		loss_val /= val_cntr
		psnr_val /= val_cntr
		ssim_val /= val_cntr

		if psnr_val > best_psnr_val:
			run_test = True
			best_psnr_val = psnr_val
			best_epoch_val = epoch
			if args.save_denoised and epoch > 5:
				save_denoised_patches(
					os.path.join(args.log_dir, 'patches_val_{}'.format(epoch)),
					all_denoised,
					all_clean,
					all_noisy,
					all_psnrs
				)

		if ssim_val > best_ssim_val:
			best_ssim_val = ssim_val

		val_logger.log({
			'epoch': epoch,
			'loss': loss_val,
			'PSNR': psnr_val,
			'SSIM': ssim_val,
			'best_epoch': best_epoch_val,
			'best_SSIM': best_ssim_val,
			'best_PSNR': best_psnr_val
		})
	time_val = time.time() - val_curr_time

	if run_test:
		run_test = False
		# Testing
		with torch.no_grad():
			model.eval()
			noisy_mat, _, data_type = get_test_data('test', args.is_raw)
			denoised_test = np.zeros(noisy_mat.shape)
			for i in range(noisy_mat.shape[0]):
				noisy = noisy_mat[i].cuda()
				if args.is_raw:
					denoised = denoise_raw_test_data(noisy)
				else:
					denoised = model(noisy)
				denoised = torch.clamp(denoised, 0., data_range)
				denoised_test[i] = denoised.cpu()

			if not args.is_raw:
				denoised_test = np.transpose(denoised_test, (0, 1, 3, 4, 2))
			denoised_test_file = 'SubmitRaw' if args.is_raw else 'SubmitSrgb'
			denoised_test_path = os.path.join(args.log_dir, '{}_{}.mat'.format(denoised_test_file, epoch))
			savemat(denoised_test_path, {denoised_test_file: denoised_test.astype(data_type)})

	print('[epoch {}] [{}] [{}] PSNR_train: {:.2f}, loss_train: {:.2f},' \
		  ' PSNR_val: {:.2f}, ssim_val: {:.4f}, loss_val: {:.2f}, train_time: {:.2f}, validation_time: {:.2f}'.format(
			  epoch,
			  args.model,
			  args.exp_name,
			  psnr_train,
			  loss_train,
			  psnr_val,
			  ssim_val,
			  loss_val,
			  time_train,
			  time_val
		  ))

	# save model
	if epoch % 10 == 0:
		save_checkpoint(
				model,
				optimizer,
				epoch,
				os.path.join(args.model_save_dir, 'epoch_{}_dncnn_model.pth'.format(epoch))
			)

writer.close()
