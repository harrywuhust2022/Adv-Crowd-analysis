import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from src.crowd_count import CrowdCounter
from src.data_loader import ImageDataLoader
from src import utils

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy

import torchvision.transforms.functional as F
from matplotlib import cm as CM

import torch.backends.cudnn as cudnn
import torch

import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import math
from torchvision import datasets, transforms
from utils_adv_patch import *
import argparse


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        v = Variable(torch.from_numpy(x).type(dtype), requires_grad=False, volatile=True)
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

# ******************************************* kappa *********************************************
def generate_kappa_schedule():
    kappa_schedule = []

    for j in range(50):
        kappa_schedule.append(1)

    kappa_value = 1.0
    step = 0.5 / 350

    for i in range(350):
        kappa_value -= step
        kappa_schedule.append(kappa_value)

    for k in range(500):
        kappa_schedule.append(0.5)

    return kappa_schedule


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using cuda: ", format(device))


#FIXME correct done
def attack(net, tgt_img_var, patch_var, mask_var, patch_init_var, gt_data_var, target_var, criterion):
    # net.eval()
    adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)

    loss_scalar = 0

    # print("now next patch: \n")

    for i in range(args.attack_epoch):

        adv_tgt_img_var = Variable(adv_tgt_img_var.data, requires_grad=True)
        just_the_patch_var = Variable(patch_var.data, requires_grad=True)

        adv_out_var = net(adv_tgt_img_var, gt_data_var)  # 要把gt_data 和 im_data 喂进去， ，并且输入都必须是numpy数组
        loss_data = criterion(adv_out_var, target_var)
        loss_reg = F.l1_loss(torch.mul(mask_var, just_the_patch_var), torch.mul(mask_var, patch_init_var))
        loss = (1 - args.alpha) * loss_data + args.alpha * loss_reg
        loss.backward()

        adv_tgt_img_grad = adv_tgt_img_var.grad.clone()
        adv_tgt_img_var.grad.data.zero_()
        patch_var -= torch.clamp(0.5 * args.lr * adv_tgt_img_grad, -2, 2)

        adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
        adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
        loss_scalar = loss.item()

        # print("attack_loss_epoch: ", loss_scalar)

    return adv_tgt_img_var, patch_var


#FIXME correct done
def train(net, patch, patch_shape, mask, patch_init, data_loader_train, criterion, optimizer, kappa, epoch, output_dir,
          method, dataset_name, Loss_list):

    patch_shape_orig = patch_shape

    net.train()
    epoch_loss = 0.0

    for blob in data_loader_train:
        im_data = blob['data']  # (1,1,645,876)  # np数组
        gt_data = blob['gt_density']  # (1,1,327,546) np数组

        data_shape = im_data.shape  # (1,1,786,1024)
        im_data_gt = torch.from_numpy(im_data)
        tgt_img_var = Variable(im_data_gt.to(device))

        gt_data_var = torch.from_numpy(gt_data)
        gt_data_var = Variable(gt_data_var.to(device))

        if args.patch_type == 'circle':
            patch, mask, patch_init, rx, ry, patch_shape = circle_transform(patch, mask, patch_init, data_shape,
                                                                            patch_shape)
        elif args.patch_type == 'square':
            patch, mask, patch_init, rx, ry = square_transform(patch, mask, patch_init, data_shape, patch_shape)

        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        patch_init = torch.FloatTensor(patch_init)
        patch, mask = patch.to(device), mask.to(device)
        patch_init = patch_init.to(device)
        patch_var, mask_var = Variable(patch), Variable(mask)
        patch_init_var = Variable(patch_init).to(device)

        # FIXME the parameter is -1 ?
        target_var = Variable(-1 * gt_data_var.data.clone(), requires_grad=True).to(device)

        adv_tgt_img_var, patch_var = attack(net, tgt_img_var, patch_var, mask_var, patch_init_var, gt_data_var,
                                            target_var, criterion)

        adv_out_var = net(adv_tgt_img_var, gt_data_var)
        normal_out_var = net(tgt_img_var, gt_data_var)
        # FIXME final balance 0.5 and 0.5

        loss_data = (1 - kappa[epoch]) * criterion(adv_out_var, gt_data_var) \
                    + kappa[epoch] * criterion(normal_out_var, gt_data_var)

        epoch_loss += loss_data.item()
        optimizer.zero_grad()
        loss_data.backward()
        optimizer.step()

        masked_patch_var = torch.mul(mask_var, patch_var)
        patch = masked_patch_var.data.cpu().numpy()
        mask = mask_var.data.cpu().numpy()
        patch_init = patch_init_var.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        new_mask = np.zeros(patch_shape)
        new_patch_init = np.zeros(patch_shape)

        for x in range(new_patch.shape[0]):
            for y in range(new_patch.shape[1]):
                new_patch[x][y] = patch[x][y][ry:ry + patch_shape[-2], rx:rx + patch_shape[-1]]
                new_mask[x][y] = mask[x][y][ry:ry + patch_shape[-2], rx:rx + patch_shape[-1]]
                new_patch_init[x][y] = patch_init[x][y][ry:ry + patch_shape[-2], rx:rx + patch_shape[-1]]

        patch = new_patch
        mask = new_mask
        patch_init = new_patch_init

        patch = zoom(patch, zoom=(1, 1, patch_shape_orig[2] / patch_shape[2], patch_shape_orig[3] / patch_shape[3]),
                     order=1)
        mask = zoom(mask, zoom=(1, 1, patch_shape_orig[2] / patch_shape[2], patch_shape_orig[3] / patch_shape[3]),
                    order=0)
        patch_init = zoom(patch_init,
                          zoom=(1, 1, patch_shape_orig[2] / patch_shape[2], patch_shape_orig[3] / patch_shape[3]),
                          order=1)

    Loss_list.append(epoch_loss / data_loader_train.get_num_samples())

    # save model parameter
    save_name = os.path.join(output_dir, '{}_{}_{}_{}.h5'.format(method, dataset_name, epoch,
                                                                 epoch_loss / data_loader_train.get_num_samples()))
    save_net(save_name, net)

    # for observation
    print("epoch: ", epoch)
    print(Loss_list)

    train_loss_txt = open('./adv_train_0.08/train_loss.txt', 'a')
    train_loss_txt.write(str(Loss_list[epoch]))
    train_loss_txt.write('\n')
    train_loss_txt.close()

    return adv_tgt_img_var, patch, adv_out_var, mask, patch_shape


def test(patch, mask, patch_shape, data_loader_val, net):
    mae_gt = 0.0
    mse_gt = 0.0
    mse_adv = 0.0
    mae_adv = 0.0

    net.eval()

    for blob in data_loader_val:
        im_data = blob['data']  # (1,1,645,876)  # np数组
        gt_data = blob['gt_density']  # (1,1,327,546) np数组

        data_shape = im_data.shape  # (1,1,786,1024)

        im_data_gt = torch.from_numpy(im_data)
        tgt_img_var = Variable(im_data_gt.to(device))

        gt_data_var = torch.from_numpy(gt_data)
        gt_data_var = Variable(gt_data_var.to(device))

        density_map = net(tgt_img_var, gt_data_var)

        if args.patch_type == 'circle':
            patch_full, mask_full, _, _, _, _ = circle_transform_test(patch, mask, patch.copy(),
                                                                                      data_shape,
                                                                                      patch_shape, True)

        elif args.patch_type == 'square':
            patch_full, mask_full, patch_init, rx, ry = square_transform(patch, mask, patch_init, data_shape,
                                                                         patch_shape)

        patch_full, mask_full = torch.FloatTensor(patch_full), torch.FloatTensor(mask_full)
        patch_full, mask_full = patch_full.to(device), mask_full.to(device)
        patch_var, mask_var = Variable(patch_full), Variable(mask_full)

        adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
        adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)

        adv_out_var = net(adv_tgt_img_var, gt_data_var)  # 要把gt_data 和 im_data 喂进去， ，并且输入都必须是numpy数组

        density_map = density_map.data.detach().cpu().numpy()
        adv_out = adv_out_var.data.detach().cpu().numpy()

        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        adv_count = np.sum(adv_out)

        mae_gt += abs(gt_count - et_count)
        mse_gt += ((gt_count - et_count)*(gt_count - et_count))

        mae_adv += abs(gt_count - adv_count)
        mse_adv += ((gt_count - adv_count)*(gt_count - adv_count))

    mae_gt = mae_gt / data_loader_val.get_num_samples()
    mse_gt = np.sqrt(mse_gt / data_loader_val.get_num_samples())

    mae_adv = mae_adv / data_loader_val.get_num_samples()
    mse_adv = np.sqrt(mse_adv / data_loader_val.get_num_samples())

    print('\nMAE_gt: %0.2f, MSE_gt: %0.2f' % (mae_gt, mse_gt))
    print('\nMAE_adv: %0.2f, MSE_adv: %0.2f' % (mae_adv, mse_adv))

    f = open('./adv_train_0.08/adv_results.txt', 'a')
    f.write('adv_mae: %s \n' % str(mae_adv))
    f.write('adv_mse: %s \n' % str(mse_adv))
    f.write('\n')
    f.close()

    f = open('./adv_train_0.08/normal_results.txt', 'a')
    f.write('normal_mae: %s \n' % str(mae_gt))
    f.write('normal_mse: %s \n' % str(mse_gt))
    f.write('\n')
    f.close()


def main():
    train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
    train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
    val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
    val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

    # data_path = './data/original/shanghaitech/part_A_final/test_data/images/'
    # gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'
    # model_path = './pretrain_models/mcnn_shtechA_660.h5'

    if not os.path.exists('./adv_train_0.08'):
        os.makedirs('./adv_train_0.08')
    output_dir = './adv_train_0.08'
    method = 'MCNN'
    dataset_name = 'A'

    data_loader_train = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
    data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)

    '''
    # eval
    net = CrowdCounter()
    trained_model = os.path.join(model_path)
    load_net(trained_model, net)
    net.to(device)
    net.eval()
    '''

    net = CrowdCounter()
    weights_normal_init(net, dev=0.01)
    net.to(device)

    params = list(net.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr_train_models)

    kappa = generate_kappa_schedule()

    criterion = torch.nn.MSELoss()

    cudnn.benchmark = True

    Loss_list = []

    # 初始化patch，MCNN特需(1,1,1024*patch_size,1024*patch_size)
    if args.patch_type == 'circle':  # image_size = 1024(default)
        patch, mask, patch_shape = init_patch_circle(args.image_size, args.patch_size)
        patch_init = patch.copy()

    elif args.patch_type == 'square':
        patch, patch_shape = init_patch_square(args.image_size, args.patch_size)
        patch_init = patch.copy()
        mask = np.ones(patch_shape)

    for epoch in range(0, args.train_epoch):

        adv_tgt_img_var, patch, adv_out_var, mask, patch_shape = train(net, patch, patch_shape, mask, patch_init, data_loader_train, criterion, optimizer, kappa, epoch, output_dir, method, dataset_name, Loss_list)

        test(patch, mask, patch_shape, data_loader_val, net)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Patch Attack Parameters')

    parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate for the patch optimization')
    parser.add_argument('--lr_train_models', type=float, default=0.0001, help='learning rate for model')
    parser.add_argument('--patch_type', type=str, default='circle')
    parser.add_argument("--patch_size", default=0.08, type=float, help="0.02 | 0.04 | 0.08 | 0.16")
    parser.add_argument("--image_size", default=200, type=str, help="this size is for the 9 patch training set")
    parser.add_argument("--train_epoch", default=800, type=int, help="the training epochs")
    # parser.add_argument("--keep", default=100, type=str, help="randomized ablation parameter")
    # parser.add_argument('--max_count', type=int, default='400', help='the max iteration numbers of patch optimization')
    parser.add_argument("--alpha", default=0, type=float, help="balance in the attack() loss function")
    parser.add_argument("--attack_epoch", default=5, type=int, help='epochs needed for every patch')

    args = parser.parse_args()

    main()
