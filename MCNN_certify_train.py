import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from src.crowd_count import CrowdCounter
from src.data_loader import ImageDataLoader
from src import utils
import argparse
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
from utils_mean import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


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


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_cuda:
        v = (torch.from_numpy(x).type(dtype)).to(device)
    if is_training:
        v = Variable(v, requires_grad=True, volatile=False)
    else:
        v = Variable(v, requires_grad=False, volatile=True)
    return v


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def train(net, data_loader, patch_shape, optimizer, val_loader, criterion, patch, mask, patch_init, output_dir, method, dataset_name):
    mae = 0.0
    mse = 0.0

    Loss_list_warm = []
    mae_list_warm = []
    mse_list_warm = []

    Loss_list_ablated = []
    mae_list_ablated = []
    mse_list_ablated = []

    for epoch in range(0, args.end_epoch):
        net.train()
        epoch_loss = 0.0
        # warm up
        if epoch < 20:

            for blob in data_loader:
                im_data = blob['data']  # (1,1,645,876)  # np数组
                gt_data = blob['gt_density']  # (1,1,327,546) np数组

                im_data_gt = torch.from_numpy(im_data)
                tgt_img_var = Variable(im_data_gt.to(device))

                gt_data_var = torch.from_numpy(gt_data)
                gt_data_var = Variable(gt_data_var.to(device))

                adv_out = net(tgt_img_var, gt_data_var)
                loss_data = criterion(adv_out, gt_data_var)
                epoch_loss += loss_data.item()
                optimizer.zero_grad()
                loss_data.backward()
                optimizer.step()

            Loss_list_warm.append(epoch_loss / data_loader.get_num_samples())

            # save model parameter
            save_name = os.path.join(output_dir, '{}_{}_{}_{}.h5'.format(method, dataset_name, epoch, epoch_loss / data_loader.get_num_samples()))
            save_net(save_name, net)

            # **************************************validate*************************************
            with torch.no_grad():
                net.eval()
                for blob in val_loader:
                    im_data = blob['data']  # (1,1,704,1024)
                    gt_data = blob['gt_density']

                    img_var = np_to_variable(im_data, is_cuda=True, is_training=False)
                    target_var = np_to_variable(gt_data, is_cuda=True, is_training=False)

                    img_ablation_var = random_mask_batch_one_sample(img_var, args.keep, reuse_noise=False)

                    density_map_var = net(img_ablation_var, target_var)
                    output = density_map_var.data.detach().cpu().numpy()

                    gt_count = np.sum(gt_data)
                    et_count = np.sum(output)

                    mae += abs(gt_count - et_count)
                    mse += ((gt_count - et_count) * (gt_count - et_count))

                mae = mae / val_loader.get_num_samples()
                mse = np.sqrt(mse / val_loader.get_num_samples())

            mae_list_warm.append(mae)
            mse_list_warm.append(mse)

            # for observation
            train_loss_txt = open('./Shanghai_A_Retrain_100/train_loss.txt', 'a')
            train_loss_txt.write(str(Loss_list_warm[epoch]))
            train_loss_txt.write('\n')
            train_loss_txt.close()

            train_loss_txt = open('./Shanghai_A_Retrain_100/ablated_mae_epoch.txt', 'a')
            train_loss_txt.write(str(mae_list_warm[epoch]))
            train_loss_txt.write('\n')
            train_loss_txt.close()

            train_loss_txt = open('./Shanghai_A_Retrain_100/ablated_mse_epoch.txt', 'a')
            train_loss_txt.write(str(mse_list_warm[epoch]))
            train_loss_txt.write('\n')
            train_loss_txt.close()

        elif epoch > 20 or epoch == 20:
            for blob in data_loader:
                im_data = blob['data']  # (1,1,645,876)  # np数组
                gt_data = blob['gt_density']  # (1,1,327,546) np数组

                # data_shape = im_data.shape  # (1,1,786,1024)

                im_data_gt = torch.from_numpy(im_data)
                tgt_img_var = Variable(im_data_gt.to(device))

                gt_data_var = torch.from_numpy(gt_data)
                gt_data_var = Variable(gt_data_var.to(device))

                '''
                if args.patch_type == 'circle':
                    patch, mask, patch_init, rx, ry, patch_shape = circle_transform(patch, mask, patch_init, data_shape,
                                                                                    patch_shape, True)
                elif args.patch_type == 'square':
                    patch, mask, patch_init, rx, ry = square_transform(patch, mask, patch_init, data_shape, patch_shape)

                # patch 和 mask现在和输入的img 维度相同 ， patch: 随机放置了一个圆(圆内像素值为随机数)，其余像素为0
                patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
                patch_init = torch.FloatTensor(patch_init)
                patch, mask = patch.to(device), mask.to(device)
                # patch_init = patch_init.to(device)
                patch_var, mask_var = Variable(patch), Variable(mask)
                # patch_init_var = Variable(patch_init).to(device)

                # add patch to the image
                adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
                '''
                # randomized ablation
                adv_final_var = random_mask_batch_one_sample(tgt_img_var, args.keep, reuse_noise=False)

                adv_out = net(adv_final_var, gt_data_var)
                loss_data = criterion(adv_out, gt_data_var)
                epoch_loss += loss_data.item()
                optimizer.zero_grad()
                loss_data.backward()
                optimizer.step()

            Loss_list_ablated.append(epoch_loss / data_loader.get_num_samples())

            # save model parameter
            save_name = os.path.join(output_dir, '{}_{}_{}_{}.h5'.format(method, dataset_name, epoch, epoch_loss / data_loader.get_num_samples()))
            save_net(save_name, net)

            # **************************************validate*************************************
            with torch.no_grad():
                net.eval()
                for blob in val_loader:
                    im_data = blob['data']  # (1,1,704,1024)
                    gt_data = blob['gt_density']

                    img_var = np_to_variable(im_data, is_cuda=True, is_training=False)
                    target_var = np_to_variable(gt_data, is_cuda=True, is_training=False)

                    img_ablation_var = random_mask_batch_one_sample(img_var, args.keep, reuse_noise=False)

                    density_map_var = net(img_ablation_var, target_var)
                    output = density_map_var.data.detach().cpu().numpy()

                    gt_count = np.sum(gt_data)
                    et_count = np.sum(output)

                    mae += abs(gt_count - et_count)
                    mse += ((gt_count - et_count) * (gt_count - et_count))

                mae = mae / val_loader.get_num_samples()
                mse = np.sqrt(mse / val_loader.get_num_samples())

            mae_list_ablated.append(mae)
            mse_list_ablated.append(mse)

            # for observation
            train_loss_txt = open('./Shanghai_A_Retrain_100/train_loss.txt', 'a')
            train_loss_txt.write(str(Loss_list_ablated[epoch-20]))
            train_loss_txt.write('\n')
            train_loss_txt.close()

            train_loss_txt = open('./Shanghai_A_Retrain_100/ablated_mae_epoch.txt', 'a')
            train_loss_txt.write(str(mae_list_ablated[epoch-20]))
            train_loss_txt.write('\n')
            train_loss_txt.close()

            train_loss_txt = open('./Shanghai_A_Retrain_100/ablated_mse_epoch.txt', 'a')
            train_loss_txt.write(str(mse_list_ablated[epoch-20]))
            train_loss_txt.write('\n')
            train_loss_txt.close()

        # adjust lr
        elif epoch == 70:  # decrease learning rate after 200 epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * 0.1

        elif epoch == 240:  # decrease learning rate after 200 epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * 0.01

        elif epoch == 400:  # decrease learning rate after 200 epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * 0.001


def main():

    method = 'MCNN'
    dataset_name = 'A'

    if not os.path.exists('./Shanghai_A_Retrain_100'):
        os.makedirs('./Shanghai_A_Retrain_100')
    output_dir = './Shanghai_A_Retrain_100'

    net = CrowdCounter()
    weights_normal_init(net, dev=0.01)
    net.to(device)

    params = list(net.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)

    criterion = torch.nn.MSELoss()

    train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
    train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
    val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
    val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

    data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
    data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)

    if args.patch_type == 'circle':  # image_size = 1024(default)
        patch, mask, patch_shape = init_patch_circle(args.image_size, args.patch_size)
        patch_init = patch.copy()

    elif args.patch_type == 'square':
        patch, patch_shape = init_patch_square(args.image_size, args.patch_size)
        patch_init = patch.copy()
        mask = np.ones(patch_shape)

    print("strat training!\n")
    train(net, data_loader, patch_shape, optimizer, data_loader_val, criterion,
          patch, mask, patch_init, output_dir, method, dataset_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Certify Training parameters')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patch_type', type=str, default='circle')
    parser.add_argument("--patch_size", default=0.02, type=float, help="0.02 | 0.04 | 0.08 | 0.16")
    parser.add_argument("--image_size", default=1024, type=str)
    parser.add_argument("--end_epoch", default=800, type=int, help="the training epochs")
    parser.add_argument("--keep", default=100, type=str, help="randomized ablation parameter")
    args = parser.parse_args()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("using cuda: ", format(device))

    main()
