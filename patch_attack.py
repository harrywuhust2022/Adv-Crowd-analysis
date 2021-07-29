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
from utils import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("using cuda: ", format(device))


def main():
    train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
    train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
    val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
    val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

    data_path = './data/original/shanghaitech/part_A_final/test_data/images/'
    gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'

    model_path = './pretrain_models/MCNN_A.h5'

    """
    if not os.path.exists('./attack_results'):
        os.makedirs('./attack_results')

    if not os.path.exists('./attack_results/results'):
        os.makedirs('./attack_results/results')
    """
    data_loader_train = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
    data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)

    test_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

    net = CrowdCounter()
    trained_model = os.path.join(model_path)
    load_net(trained_model, net)
    net.cuda()
    net.eval()

    criterion = torch.nn.MSELoss()

    if args.patch_type == 'circle':
        patch, mask, patch_shape = init_patch_circle(args.image_size, args.patch_size)
        patch_init = patch.copy()

    elif args.patch_type == 'square':
        patch, patch_shape = init_patch_square(args.image_size, args.patch_size)
        patch_init = patch.copy()
        mask = np.ones(patch_shape)

    for epoch in range(1, args.train_epoch + 1):
        adv_tgt_img_var, patch, adv_out_var, mask, patch_shape = train(net, patch, patch_shape, mask, patch_init, data_loader_train, criterion)

        test(patch, mask, patch_shape, data_loader_val, net)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Patch Attack Parameters')

    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--patch_type', type=str, default='circle')
    parser.add_argument("--patch_size", default=0.16, type=float, help="0.02 | 0.04 | 0.08 | 0.16")
    parser.add_argument("--image_size", default=200, type=str, help="this size is for the 9 patch training set")
    parser.add_argument("--train_epoch", default=100, type=int, help="the training epochs")
    # parser.add_argument("--keep", default=100, type=str, help="randomized ablation parameter")
    # parser.add_argument('--max_count', type=int, default='400', help='the max iteration numbers of patch optimization')
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--attack_epoch", default=500, type=int)

    args = parser.parse_args()

    main()
