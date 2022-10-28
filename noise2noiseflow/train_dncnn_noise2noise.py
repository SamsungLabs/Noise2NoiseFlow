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
from data_loader.loader import SIDDFullRawDataset, SIDDMediumDataset, SIDDFullRawDatasetV2, SIDDMediumDatasetV2, SIDDFullRawDatasetWrapper, SIDDMediumDatasetWrapper
from torch.utils.data import Dataset, DataLoader
from model.dncnn import DnCNN
from utils.train_utils import batch_PSNR, weights_init_orthogonal
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse
from tensorboardX import SummaryWriter
import torchvision.utils as utils
import os
import shutil


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

def save_checkpoint(model, optimizer, epoch_num, checkpoint_dir):
	checkpoint = {'epoch_num' : epoch_num, 'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
	torch.save(checkpoint, checkpoint_dir)

def load_checkpoint(model, optimizer, checkpoint_dir):
	checkpoint = torch.load(checkpoint_dir)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	return model, optimizer, checkpoint['epoch_num']

parser = argparse.ArgumentParser(description="DnCNN_noise2noise")
# parser.add_argument("--name", type=str, default="", help="Name of the model")
parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
parser.add_argument("--sidd_path", type=str, help='Path to SIDD Full Raw dataset')
parser.add_argument("--depth", type=int, default=9, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--logdir", type=str, default='test', help='Path to where you want to store logs')
parser.add_argument("--iso", type=int, default=None, help="ISO level for dataset images")
parser.add_argument("--cam", type=str, default=None, help="CAM type for dataset images")
parser.add_argument("--swap_images", default=False, action='store_true', help="use (n1, n2) and (n2, n1) as training pairs.")
parser.add_argument("--no_resume", default=False, action='store_true', help="resume the training from a previous checkpoint")

args = parser.parse_args()

logdir = os.path.abspath(os.path.join('experiments', 'noise2noise', args.logdir)) + '/'

if args.no_resume:
	if os.path.exists(logdir):
		shutil.rmtree(logdir)

if not os.path.exists(logdir):
	os.makedirs(logdir, exist_ok=True)

args.logdirname = args.logdir
args.logdir = logdir
args.tensorboard_save_dir = os.path.join(args.logdir, 'tensorboard_logs')
args.model_save_dir = os.path.join(args.logdir, 'saved_models')
args.best_model_dir = os.path.join(args.logdir, 'best_model')


train_dataset = SIDDFullRawDatasetV2(sidd_full_path=args.sidd_path, patch_sampling='dncnn', train_or_test = 'train', cam=args.cam, iso=args.iso)
val_dataset = SIDDFullRawDatasetV2(sidd_full_path=args.sidd_path, patch_sampling='dncnn', train_or_test='test', cam=args.cam, iso=args.iso)
test_dataset = SIDDMediumDatasetV2(sidd_medium_path='../data/SIDD_Medium_Raw/Data', patch_sampling='dncnn', train_or_test='test', cam=args.cam, iso=args.iso)

loader_train = SIDDFullRawDatasetWrapper(train_dataset, batch_size=args.batch_size)
loader_val = SIDDFullRawDatasetWrapper(val_dataset, batch_size=args.batch_size)
loader_test = SIDDMediumDatasetWrapper(test_dataset, batch_size=args.batch_size)

print('Number of training samples: {}'.format(len(train_dataset)))
print('Number of validation/test samples: {}'.format(len(val_dataset)))

num_channels = 4
model = DnCNN(channels=num_channels, num_of_layers=args.depth)

model.apply(weights_init_orthogonal)
criterion = nn.MSELoss(reduction='sum')

# device_ids = [0]
# model = nn.DataParallel(model, device_ids=device_ids).cuda()
model.cuda()
criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

start_epoch = 1
if not os.path.exists(args.model_save_dir): # no resume
	os.makedirs(args.model_save_dir)
	os.makedirs(args.tensorboard_save_dir)
	os.makedirs(args.best_model_dir)
else: # resume mode
	models = sorted(os.listdir(args.model_save_dir))
	last_epoch = str(max([int(i.split("_")[1]) for i in models[1:]]))
	saved_model_file_name = 'epoch_{}_dncnn_model_net.pth'.format(last_epoch)
	saved_model_file_path = os.path.join(args.model_save_dir, saved_model_file_name)
	print(saved_model_file_path)
	model, optimizer, start_epoch = load_checkpoint(model, optimizer, saved_model_file_path)
	start_epoch += 1
	print('found an existing previous checkpoint, resuming from epoch {}'.format(start_epoch))

writer = SummaryWriter(args.tensorboard_save_dir)
# writer.add_graph(model, next(iter(train_dataloader))['noisy1'].cuda())

best_psnr = 0
step = 0
for epoch in range(start_epoch, args.epochs):
	epoch_loss = 0
	current_lr = lr_schedule(epoch)
	# set learning rate
	for param_group in optimizer.param_groups:
		param_group["lr"] = current_lr

	# train
	for i, data in enumerate(loader_train, 0):
		# training step
		model.train()
		model.zero_grad()
		optimizer.zero_grad()

		noisy1, noisy2 = data['noisy1'], data['noisy2']
		noisy1, noisy2 = noisy1.cuda(), noisy2.cuda()

		denoised1 = model(noisy1)
		loss1 = criterion(denoised1, noisy2) / (noisy1.size()[0]*2)
		loss = loss1

		if args.swap_images:
			denoised2 = model(noisy2)
			loss2 = criterion(denoised2, noisy1) / (noisy2.size()[0]*2)
			loss += loss2
			loss /= 2


		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()

		writer.add_scalar('train loss', loss.item(), step)
		step += 1

	writer.add_scalar('train epoch loss', epoch_loss, epoch)
	# the end of each epoch

	model.eval()
	with torch.no_grad():
		# validate
		loss_val = 0
		loss_cntr = 0
		for val_data in loader_val:
			noisy1, noisy2 = val_data['noisy1'], val_data['noisy2']
			noisy1, noisy2 = noisy1.cuda(), noisy2.cuda()

			denoised = model(noisy1)
			loss = criterion(denoised, noisy2) / (noisy1.size()[0]*2)
			denoised = torch.clamp(denoised, 0., 1.)

			loss_val += loss.item()
			loss_cntr += 1
		loss_val /= loss_cntr
		writer.add_scalar('validation loss', loss_val, epoch)

		psnr_test = 0
		test_cntr = 0
		loss_test = 0
		for test_data in loader_test:
			noise, gt = test_data['noise'], test_data['clean']
			noisy = noise + gt
			noisy, gt = noisy.cuda(), gt.cuda()

			denoised = model(noisy)
			loss = criterion(denoised, gt) / (noisy.size()[0]*2)
			denoised = torch.clamp(denoised, 0., 1.)

			psnr_test += batch_PSNR(denoised, gt, 1.)
			loss_test += loss.item()
			test_cntr += 1
		loss_test /= test_cntr
		psnr_test /= test_cntr

		is_best = 0
		if best_psnr < psnr_test:
			is_best = 1
			best_psnr = psnr_test
			torch.save(model.state_dict(), os.path.join(args.best_model_dir, 'n2n_dncnn_best_model.pth'.format(epoch)))

		print("[epoch %d] model_name: %s, loss val: %.4f, loss_test: %.4f, PSNR_test: %.4f, is_best: %d" % (epoch, args.logdirname, loss_val, loss_test, psnr_test, is_best))
		writer.add_scalar('test PSNR', psnr_test, epoch)
		writer.add_scalar('test loss', loss_test, epoch)

		# log the images
		# denoised = torch.clamp(model(noisy), 0., 1.)
		# Img = utils.make_grid(noisy.data, nrow=8, normalize=True, scale_each=True)
		# Imgn = utils.make_grid(gt.data, nrow=8, normalize=True, scale_each=True)
		# Irecon = utils.make_grid(denoised.data, nrow=8, normalize=True, scale_each=True)
		# writer.add_image('noisy image', Img, epoch)
		# writer.add_image('clean image', Imgn, epoch)
		# writer.add_image('reconstructed image', Irecon, epoch)

	# save model
	if epoch % 10 == 0:
		save_checkpoint(model, optimizer, epoch, os.path.join(args.model_save_dir, 'epoch_{}_dncnn_model_net.pth'.format(epoch)))

writer.close()
