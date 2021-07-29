from __future__ import division
import shutil
import numpy as np
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from scipy.ndimage.interpolation import rotate, zoom
from PIL import Image


def load_as_float(path):
    return np.array(Image.open(path)).astype(np.float32)


def imresize(arr, sz):
    height, width = sz
    return np.array(Image.fromarray(arr.astype('uint8')).resize((width, height), resample=Image.BILINEAR))


def tensor2array(tensor, max_value=255, colormap='rainbow'):
    if max_value is None:
        max_value = tensor.max()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('3'):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        if (tensor.size(0) == 3):
            array = 0.5 + tensor.numpy().transpose(1, 2, 0)*0.5
        elif (tensor.size(0) == 2):
            array = tensor.numpy().transpose(1, 2, 0)
    return array


def transpose_image(array):
    return array.transpose(2, 0, 1)


def save_checkpoint(save_path, dispnet_state, exp_pose_state, flownet_state, optimizer_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose', 'flownet', 'optimizer']
    states = [dispnet_state, exp_pose_state, flownet_state, optimizer_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))


def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements,
    # we can find the desired rectangular bounds.
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]


def crop_patch(patch):
    pass


class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr
    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])-2

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def init_patch_square(image_size, patch_size):
    # get mask
    # image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size)#**(0.5))
    patch = np.random.rand(1,1,noise_dim,noise_dim)  # MCNN只能黑白图
    return patch, patch.shape


def init_patch_circle(image_size, patch_size):
    patch, patch_shape = init_patch_square(image_size, patch_size)
    mask = createCircularMask(patch_shape[-2], patch_shape[-1]).astype('float32')
    mask = np.array([[mask]])
    return patch, mask, patch.shape


def circle_transform(patch, mask, patch_init, data_shape, patch_shape, margin=0, center=True, norotate=False, fixed_loc=(-1,-1)):
    patch = patch + np.random.random()*0.1 - 0.05

    patch = np.clip(patch, 0.,1.)
    patch = patch * mask
    x = np.zeros(data_shape)
    xm = np.zeros(data_shape)  # (1,1,1024,786)
    xp = np.zeros(data_shape)

    image_w, image_h = data_shape[-1], data_shape[-2]  # 得到宽和高

    zoom_factor = 1 + 0.05*(np.random.random() - 0.5)
    patch = zoom(patch, zoom=(1,1,zoom_factor, zoom_factor), order=1)
    mask = zoom(mask, zoom=(1,1,zoom_factor, zoom_factor), order=0)
    patch_init = zoom(patch_init, zoom=(1,1,zoom_factor, zoom_factor), order=1)

    patch_shape = patch.shape
    m_size = patch.shape[-1]

    for i in range(x.shape[0]):
        # random rotation
        if not norotate:
            rot = 10*(np.random.random() - 0.5)
            for j in range(patch[i].shape[0]):
                patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False, order=1)
                patch_init[i][j] = rotate(patch_init[i][j], angle=rot, reshape=False, order=1)

        if fixed_loc[0] < 0 or fixed_loc[1] < 0:
            if center:
                random_x = (image_w - m_size) // 2

            else:
                random_x = m_size + margin + np.random.choice(image_w - 2*m_size - 2*margin -2)
            assert(random_x + m_size < x.shape[-1])

            if center:
                random_y = (image_h - m_size) // 2

            else:
                random_y = m_size + np.random.choice(image_h - 2*m_size -2)
            assert(random_y + m_size < x.shape[-2])

        else:
            random_x = fixed_loc[0]
            random_y = fixed_loc[1]

        # apply patch to dummy image
        x[i][0][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch[i][0]

        # apply mask to dummy image
        xm[i][0][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = mask[i][0]

        # apply patch_init to dummy image
        xp[i][0][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch_init[i][0]

    return x, xm, xp, random_x, random_y, patch_shape


def circle_transform_test(patch, mask, patch_init, data_shape, margin=0, center=True, norotate=False, fixed_loc=(-1,-1)):
    #print("patch_shape_0: ", patch.shape)
    patch = patch + np.random.random()*0.1 - 0.05
    #print("patch_shape_1: ", patch.shape)
    patch = np.clip(patch, 0.,1.)  # 将patch里的随机数值clip到(0,1)之间
    #print("patch_shape_2: ", patch.shape)
    #print("mask_shape: ", mask.shape)
    #print("patch_shape: ", patch.shape)

    patch = patch * mask

    x = np.zeros(data_shape)
    xm = np.zeros(data_shape)  # (1,1,1024,786)
    xp = np.zeros(data_shape)

    # get shape
    image_w, image_h = data_shape[-1], data_shape[-2]  # 得到宽和高

    # zoom()缩放数组
    zoom_factor = 1 + 0.05*(np.random.random() - 0.5)
    patch = zoom(patch, zoom=(1,1,zoom_factor, zoom_factor), order=1)
    mask = zoom(mask, zoom=(1,1,zoom_factor, zoom_factor), order=0)
    patch_init = zoom(patch_init, zoom=(1,1,zoom_factor, zoom_factor), order=1)

    patch_shape = patch.shape
    m_size = patch.shape[-1]
    # print("m_size: ", m_size)
    for i in range(x.shape[0]):
        # random rotation
        if not norotate:
            rot = 10*(np.random.random() - 0.5)
            for j in range(patch[i].shape[0]):
                patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False, order=1)
                patch_init[i][j] = rotate(patch_init[i][j], angle=rot, reshape=False, order=1)

        if fixed_loc[0] < 0 or fixed_loc[1] < 0:
            if center:
                random_x = (image_w - m_size) // 2

            else:
                random_x = m_size + margin + np.random.choice(image_w - 2*m_size - 2*margin -2)
            assert(random_x + m_size < x.shape[-1])

            if center:
                random_y = (image_h - m_size) // 2

            else:
                random_y = m_size + np.random.choice(image_h - 2*m_size -2)
            assert(random_y + m_size < x.shape[-2])

        else:
            random_x = fixed_loc[0]
            random_y = fixed_loc[1]

        # apply patch to dummy image
        x[i][0][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch[i][0]

        # apply mask to dummy image
        xm[i][0][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = mask[i][0]

        # apply patch_init to dummy image
        xp[i][0][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch_init[i][0]

    return x, xm, xp, random_x, random_y, patch_shape


def init_patch_from_image(image_path, mask_path, image_size, patch_size):
    noise_size = np.floor(image_size*np.sqrt(patch_size))
    patch_image = load_as_float(image_path)
    # return patch, mask, patch.shape
    patch_image = imresize(patch_image, (int(noise_size), int(noise_size)))/128. -1
    patch = np.array([patch_image.transpose(2,0,1)])

    mask_image = load_as_float(mask_path)
    mask_image = imresize(mask_image, (int(noise_size), int(noise_size)))/256.
    mask = np.array([mask_image.transpose(2,0,1)])
    return patch, mask, patch.shape


def square_transform(patch, mask, patch_init, data_shape, patch_shape, norotate=False):
    # get dummy image
    image_w, image_h = data_shape[-1], data_shape[-2]
    x = np.zeros(data_shape)
    xm = np.zeros(data_shape)
    xp = np.zeros(data_shape)
    # get shape
    m_size = patch_shape[-1]

    for i in range(x.shape[0]):

        # random rotation
        if not norotate:
            rot = np.random.choice(4)
            for j in range(patch[i].shape[0]):
                patch[i][j] = np.rot90(patch[i][j], rot)
                mask[i][j] = np.rot90(mask[i][j], rot)

                patch_init[i][j] = np.rot90(patch_init[i][j], rot)

        # random location
        random_x = np.random.choice(image_w-m_size-1)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_w)
        random_y = np.random.choice(image_h-m_size-1)
        if random_y + m_size > x.shape[-2]:
            while random_y + m_size > x.shape[-2]:
                random_y = np.random.choice(image_h)

        # apply patch to dummy image
        x[i][0][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch[i][0]
        x[i][1][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch[i][1]
        x[i][2][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch[i][2]
        # apply mask to dummy image
        xm[i][0][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = mask[i][0]
        xm[i][1][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = mask[i][1]
        xm[i][2][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = mask[i][2]

        # apply patch_init to dummy image
        xp[i][0][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch_init[i][0]
        xp[i][1][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch_init[i][1]
        xp[i][2][random_y:random_y+patch_shape[-2], random_x:random_x+patch_shape[-1]] = patch_init[i][2]

    # mask = np.copy(x)
    # mask[mask != 0] = 1.0

    return x, xm, xp, random_x, random_y


epsilon = 1e-8


def compute_epe(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    h_gt, w_gt = gt.size()
    bs = 1
    nc = 1
    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    if nc == 3:
        valid = gt[:,2,:,:]
        epe = epe * valid
        avg_epe = epe.sum()/(valid.sum() + epsilon)
    else:
        avg_epe = epe.sum()/(bs*h_gt*w_gt)


    if type(avg_epe) == Variable: avg_epe = avg_epe.data

    return avg_epe.item()


def compute_cossim(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    #u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    #u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    #v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    similarity = nn.functional.cosine_similarity(gt[:,:2], pred)
    if nc == 3:
        valid = gt[:,2,:,:]
        similarity = similarity * valid
        avg_sim = similarity.sum()/(valid.sum() + epsilon)
    else:
        avg_sim = similarity.sum()/(bs*h_gt*w_gt)


    if type(avg_sim) == Variable: avg_sim = avg_sim.data

    return avg_sim.item()


def multiscale_cossim(gt, pred):
    assert(len(gt)==len(pred))
    loss = 0
    for (_gt, _pred) in zip(gt, pred):
        loss +=  - nn.functional.cosine_similarity(_gt, _pred).mean()

    return loss