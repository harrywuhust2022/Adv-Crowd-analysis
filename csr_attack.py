import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from CSR_src.model import *
from CSR_src.dataset import *
# from CSR_src.make_dataset import *
from CSR_src.image import *

import torch.backends.cudnn as cudnn
import torch

import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import math
from torchvision import datasets, transforms
from adv_utils_patch_csr import *

transform1 = transforms.Compose([transforms.ToTensor()])
transform2 = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])
unloader = transforms.ToPILImage()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda:0')
print('use cuda ==> {}'.format(device))

root = './data'
# now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')

path_sets = [part_A_test]
path_sets2 = [part_A_train]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths2 = []
for path in path_sets2:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths2.append(img_path)

model_path = './1model_best.pth.tar'
model = CSRNet()
model = model.to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(img_paths,
                        shuffle=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225]),
                        ]), train=False),
    batch_size=1)

train_loader = torch.utils.data.DataLoader(
    dataset.listDataset(img_paths2,
                        shuffle=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225]),
                        ]), train=False),
    batch_size=1)


epsilon = 0.1
max_count = 1000
patch_type = 'circle'
patch_size = 0.04
image_size = 1024
train_size = 300
test_size = 300
plot_all = 1
epochs = 1
zeta = 0.9
lr = 1e-3
alpha = 0.1
gamma = 0.01
min_out = 0
max_out = 255

criterion = torch.nn.CrossEntropyLoss()

criterion_1 = torch.nn.MSELoss()

# epoch_size = len(test_loader)
epoch_size = 2


def train(epoch, patch, patch_shape, mask, patch_init):
    model.eval()

    n_iter = 0

    mae_gt = 0
    mae_adv = 0
    mse_adv = 0
    mse_gt = 0

    patch_shape_orig = patch_shape

    for i, (img, target) in tqdm(enumerate(train_loader)):
        print(i)
        print(img.shape)
        print(target.shape)

        img = img.cuda()
        target = target.cuda()

        tgt_img_var = Variable(img)
        target = Variable(target)

        model_pre_var = model(tgt_img_var)

        print(model_pre_var.shape)

        data_shape = img.data.cpu().numpy().shape
        beta = np.random.randn(data_shape)

        if patch_type == 'circle':
            patch, mask, patch_init, rx, ry, patch_shape, beta = circle_transform(patch, mask, patch_init, data_shape,
                                                                            patch_shape, beta, True)

        elif patch_type == 'square':
            patch, mask, patch_init, rx, ry, beta = square_transform(patch, mask, patch_init, data_shape, beta, patch_shape)

        # patch 和 mask现在和输入的img 维度相同 ， patch: 随机放置了一个圆(圆内像素值为随机数)，其余像素为0
        patch, mask, beta = torch.FloatTensor(patch), torch.FloatTensor(mask), torch.FloatTensor(beta)

        patch_init = torch.FloatTensor(patch_init)

        patch, mask, beta = patch.cuda(), mask.cuda(), beta.cuda()

        patch_init = patch_init.cuda()

        patch_var, mask_var, beta_var = Variable(patch), Variable(mask), Variable(beta)

        patch_init_var = Variable(patch_init).cuda()

        target_var = Variable(10 * model_pre_var.data.clone(), requires_grad=True).cuda()

        adv_tgt_img_var, patch_var, adv_out_var = attack(tgt_img_var, patch_var, beta_var, mask_var, patch_init_var,
                                                         target_var=target_var)

        masked_patch_var = torch.mul(mask_var, patch_var)

        patch = masked_patch_var.data.cpu().numpy()

        mask = mask_var.data.cpu().numpy()

        patch_init = patch_init_var.data.cpu().numpy()

        adv_out = adv_out_var.data.detach().cpu().numpy()

        if not os.path.exists('./results_0.02_CAN'):
            os.mkdir('./results_0.02_CAN')
        if not os.path.exists('./results_0.02_CAN/images_adv'):
            os.mkdir('./results_0.02_CAN/images_adv')
        if not os.path.exists('./results_0.02_CAN/images_gt'):
            os.mkdir('./results_0.02_CAN/images_gt')
        if not os.path.exists('./results_0.02_CAN/patch'):
            os.mkdir('./results_0.02_CAN/patch')

        if not os.path.exists('./results_0.02_CAN/density map_gt'):
            os.mkdir('./results_0.02_CAN/density map_gt')

        if not os.path.exists('./results_0.02_CAN/density map_adv'):
            os.mkdir('./results_0.02_CAN/density map_adv')

        # 存储 unattacked 图
        if not os.path.exists('./results_0.02_CAN/density map_pre'):
            os.mkdir('./results_0.02_CAN/density map_pre')

        imgpath, full_imgname = os.path.split(img_paths2[i])

        imgname, imgext = os.path.splitext(full_imgname)

        gt_file = h5py.File(img_paths2[i].replace(
            '.jpg', '.h5').replace('images', 'ground_truth'), 'r')

        groundtruth = np.asarray(gt_file['density'])

        # save ground_truth
        plt.imsave('./results_0.02_CSR/density map_gt/{}'.format(full_imgname), groundtruth, format='png',
                   cmap=plt.cm.jet)

        adv_out = adv_out[0][0]
        plt.imsave('./results_0.02_CSR/density map_adv/{}'.format(full_imgname), adv_out, format='png', cmap=plt.cm.jet)

        model_pre = model_pre_var.data.detach().cpu().numpy()
        model_pre = model_pre[0][0]
        plt.imsave('./results_0.02_CSR/density map_pre/{}'.format(full_imgname), model_pre, format='png',
                   cmap=plt.cm.jet)

        tgt_img = tgt_img_var.data.detach().cpu().numpy()
        tgt_img = np.squeeze(tgt_img)
        tgt_img = np.swapaxes(tgt_img, 0, 1)
        tgt_img = np.swapaxes(tgt_img, 1, 2)
        plt.imsave('./results_0.02_CSR/images_gt/{}'.format(full_imgname), tgt_img, format='png', cmap=plt.cm.jet)

        adv_tgt_img = adv_tgt_img_var.data.detach().cpu().numpy()
        adv_tgt_img = np.squeeze(adv_tgt_img)
        adv_tgt_img = np.swapaxes(adv_tgt_img, 0, 1)
        adv_tgt_img = np.swapaxes(adv_tgt_img, 1, 2)
        plt.imsave('./results_0.02_CSR/images_adv/{}'.format(full_imgname), adv_tgt_img, format='png', cmap=plt.cm.jet)

        patch_c = np.squeeze(patch_copy)
        patch_c = np.swapaxes(patch_c, 0, 1)
        patch_c = np.swapaxes(patch_c, 1, 2)
        plt.imsave('./results_0.02_CSR/patch/{}'.format(full_imgname), patch_c, format='png', cmap=plt.cm.jet)


        output_gt = model_pre_var.data.detach().cpu().numpy()
        output_gt = output_gt[0][0]

        gt_file = h5py.File(img_paths2[i].replace(
            '.jpg', '.h5').replace('images', 'ground_truth'), 'r')

        groundtruth = np.asarray(gt_file['density'])

        mae_adv += abs(adv_out.sum() - np.sum(groundtruth))
        mae_gt += abs(output_gt.sum() - np.sum(groundtruth))

        mse_adv += ((adv_out.sum() - np.sum(groundtruth))*(adv_out.sum() - np.sum(groundtruth)))
        mse_gt += (output_gt.sum() - np.sum(groundtruth))*(output_gt.sum() - np.sum(groundtruth))

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

        if i >= epoch_size-1:
            break

        n_iter += 1

    return patch, mask, patch_init, patch_shape


def attack(tgt_img_var, patch_var, beta_var, mask_var, patch_init_var, target_var):
    model.eval()

    adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)

    count = 0
    loss_scalar = 1

    while loss_scalar > 0.1:

        count += 1

        adv_tgt_img_var = Variable(adv_tgt_img_var.data, requires_grad=True)

        just_the_patch_var = Variable(patch_var.data, requires_grad=True)

        adv_out_var = model(adv_tgt_img_var)

        loss_data = criterion(adv_out_var, target_var)

        loss_reg = F.l1_loss(torch.mul(mask_var, just_the_patch_var), torch.mul(mask_var, patch_init_var))

        loss_1 = (1 - alpha) * loss_data + alpha * loss_reg

        beta_shape = beta_var.data.cpu().numpy().shape
        beta_data = beta_var.data.cpu().numpy()\
        sum1, sum2 = 0
        for k in beta_shape[0]:
            sum1 += np.sum(criterion_1(beta_data[k][], beta_data[k+1][]))
        for k in beta_shape[1]:
            sum1 += np.sum(criterion_1(beta_data[k][], beta_data[k+1][]))
        for k in beta_shape[2]:
            sum1 += np.sum(criterion_1(beta_data[k][], beta_data[k+1][]))
        for j in beta_shape[0]:
            sum2 += np.sum(criterion_1(beta_data[][j], beta_data[][j+1]))
        for j in beta_shape[1]:
            sum2 += np.sum(criterion_1(beta_data[][j], beta_data[][j+1]))
        for j in beta_shape[2]:
            sum2 += np.sum(criterion_1(beta_data[][j], beta_data[][j+1]))

        loss_sm = sum1 + sum2

        loss = gamma * loss_sm + loss_1

        loss.backward()

        adv_tgt_img_grad = adv_tgt_img_var.grad.clone()

        Vdp = zeta * patch_var + (1 - zeta) * (adv_tgt_img_grad)/sum(torch.abs(adv_tgt_img_grad))

        patch_var = patch_var - lr * Vdp

        beta_var_grad = beta_var.grad.clone()

        beta_var = beta_var - lr * beta_var_grad

        beta_var.grad.data.zero_()

        adv_tgt_img_var.grad.data.zero_()

        patch_var -= torch.clamp(0.5 * lr * adv_tgt_img_grad, -2, 2)

        adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)

        adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)

        loss_scalar = loss.item()

        if count > max_count - 1:
            break

    return adv_tgt_img_var, patch_var, adv_out_var


# def test(patch, mask, patch_shape, epoch):
def test(patch, mask, patch_shape, epoch):
    model.eval()

    mae_gt = 0
    mse_gt = 0
    mse_adv = 0
    mae_adv = 0

    for i, (img, target) in tqdm(enumerate(test_loader)):

        tgt_img_var = Variable(img.cuda(), requires_grad=True)

        target_var = Variable(target.cuda(), requires_grad=True)

        data_shape = img.data.cpu().numpy().shape

        if patch_type == 'circle':
            patch_full, mask_full, _, rx, ry, _ = circle_transform(patch, mask, patch.copy(), data_shape, patch_shape)
        elif patch_type == 'square':
            patch_full, mask_full, _, _, _ = square_transform(patch, mask, patch.copy(), data_shape, patch_shape,
                                                              norotate=args.norotate)

        patch_full, mask_full = torch.FloatTensor(patch_full), torch.FloatTensor(mask_full)

        patch_full, mask_full = patch_full.cuda(), mask_full.cuda()

        patch_var, mask_var = Variable(patch_full), Variable(mask_full)

        adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)

        adv_out_var = model(adv_tgt_img_var)

        gt_out_var = model(tgt_img_var)

        gt_file = h5py.File(img_paths[i].replace(
            '.jpg', '.h5').replace('images', 'ground_truth'), 'r')

        groundtruth = np.asarray(gt_file['density'])

        # 此时output是adv_out
        output_adv = adv_out_var.data.detach().cpu().numpy()
        output_gt = gt_out_var.data.detach().cpu().numpy()

        mae_adv += abs(output_adv.sum() - np.sum(groundtruth))
        mae_gt += abs(output_gt.sum() - np.sum(groundtruth))

        mse_adv += (output_adv.sum() - np.sum(groundtruth)) * (output_adv.sum() - np.sum(groundtruth))
        mse_gt += (output_gt.sum() - np.sum(groundtruth)) * (output_gt.sum() - np.sum(groundtruth))

    mae_adv = mae_adv / len(test_loader)
    mae_gt = mae_gt / len(test_loader)

    mse_gt = np.sqrt(mse_gt / len(test_loader))
    mse_adv = np.sqrt(mse_adv / len(test_loader))

    print('\nMAE_gt: %0.2f, MSE_gt: %0.2f' % (mae_gt, mse_gt))
    print('\nMAE_adv: %0.2f, MSE_adv: %0.2f' % (mae_adv, mse_adv))

    with open('./mae.txt', 'a') as file_handle:
        file_handle.write(str(mae_gt))
        file_handle.write('\n')

    with open('./mae_adv.txt', 'a') as file_handle_1:
        file_handle_1.write(str(mae_adv))
        file_handle_1.write('\n')

    with open('./mse_gt.txt', 'a') as file_handle_1:
        file_handle_1.write(str(mse_gt))
        file_handle_1.write('\n')

    with open('./mse_adv.txt', 'a') as file_handle_1:
        file_handle_1.write(str(mse_adv))
        file_handle_1.write('\n')


def main():
    cudnn.benchmark = True
    # 初始化patch
    if patch_type == 'circle':
        patch, mask, patch_shape = init_patch_circle(image_size,
                                                     patch_size)
        patch_init = patch.copy()

    elif patch_type == 'square':
        patch, patch_shape = init_patch_square(image_size, patch_size)
        patch_init = patch.copy()
        mask = np.ones(patch_shape)

    for epoch in range(1, epochs + 1):
        patch, mask, patch_init, patch_shape = train(epoch, patch, patch_shape, mask, patch_init)
        test(patch, mask, patch_shape, epoch)
        print('this is the ', epoch, 'epochs')


if __name__ == '__main__':
    main()