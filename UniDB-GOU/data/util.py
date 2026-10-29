import math
import os
import pickle
import random

import cv2
import numpy as np
import torch

import re
from collections import defaultdict


# Files & IO

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'npz']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def _get_paths_from_images_neighbor(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        # dictionary to collect maximum slice per exam
        max_slices = defaultdict(int)

        for f in fnames:
            # extract exam name and slice number
            match = re.match(r"(.+)_slice_(\d+)\.npz", f)
            if match:
                exam, slice_num = match.groups()
                slice_num = int(slice_num)
                # keep maximum slice number for each exam
                max_slices[exam] = max(max_slices[exam], slice_num)

        # convert to normal dict if needed
        max_slices = dict(max_slices)
        
        for fname in sorted(fnames):
            match = re.match(r"(.+)_slice_(\d+)\.npz", fname)
            exam, slice_num = match.groups()
            if exam in ["Exam6867Series16", "Exam2113Series17", "Exam7575-Series15", "Exam7684-Series14", "Exam27484-Series15"]:
                print('ignore')
            # elif exam in ["Exam2576Series17"]:
            #     nogo_list = ['1', '2', str(max_slices[exam]-1), str(max_slices[exam])]
            #     if is_image_file(fname) and slice_num not in nogo_list:# and eval(slice_num) in [44, 46, 48, 50, 52, 54, 56, 58]:
            #         img_path = os.path.join(dirpath, fname)
            #         images.append(img_path)
            else:
                nogo_list = ['1', '2', str(max_slices[exam]-1), str(max_slices[exam])]
                if is_image_file(fname) and slice_num not in nogo_list:# and eval(slice_num) in [25, 28, 30, 32, 33, 34, 36, 38, 40, 42]:
                    img_path = os.path.join(dirpath, fname)
                    images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb meta info'''
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def get_image_paths(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
            return paths, sizes
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
            return paths
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))

def get_image_paths_neighbor(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
            return paths, sizes
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images_neighbor(dataroot))
            return paths
        
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))

def _read_img_lmdb(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif env == "npz":
        img = np.load(path)
        if 'rnufft' in img:
            img = np.abs(img['rnufft'])
        elif 'grasp' in img:
            img = np.abs(img['grasp'])
        else:
            img = np.abs(img['softgating'])
        img = (img - np.min(img))/ (np.max(img)- np.min(img)) * 255
    else:
        img = _read_img_lmdb(env, path, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        # img = img[:, :, :3]
        img = np.transpose(img, (1, 2, 0))
    return img



def read_img_bias(env, path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif env == "npz":
        img = np.load(path)
        if 'rnufft' in img:
            bias = img['bias_field']
            img = np.abs(img['rnufft'])
            mode = 'bias'
        else:
            img = np.abs(img['grasp'])
            mode = 'grasp'
        img = (img - np.min(img))/ (np.max(img)- np.min(img)) * 255
    else:
        img = _read_img_lmdb(env, path, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        if mode == 'bias':
            bias = np.expand_dims(bias, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        # img = img[:, :, :3]
        img = np.transpose(img, (1, 2, 0))
        if mode == 'bias':
            bias = np.transpose(bias, (1, 2, 0))
    if mode == 'bias':
        return img, bias
    else:
        return img


def read_img_wo_minmax(env, path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif env == "npz":
        img = np.load(path)
        if 'rnufft' in img:
            img = np.abs(img['rnufft'])
        else:
            img = np.abs(img['grasp'])
    else:
        img = _read_img_lmdb(env, path, size)
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        # img = img[:, :, :3]
        img = np.transpose(img, (1, 2, 0))
    return img

# image processing
# process on numpy image

def augment(img, hflip=True, rot=True, mode=None, swap=None):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img
    if mode in ['LQ','GT', 'SRker']:
        return _augment(img)
    elif mode in ['LQGT_1ch', 'LQGT_1ch_window', 'LQGT_5ch_ref_copied', 'LQGT_5ch_ref_copied_bias', 'LQGT_5ch_ref_copied_window', 'LQGT_5ch_ref_copied_extend_aug']:
        if swap and random.random() < 0.5:
            img.reverse()
        return [_augment(I) for I in img]


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


# Functions
# matlab 'imresize' function, now only support 'bicubic'

def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3

    weight = (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))
    return weight


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round
    is_numpy = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.transpose(2, 0, 1)).cuda()
        is_numpy = True
    device = img.device
    # device = torch.device("cuda")

    is_batch = True
    if len(img.shape) == 3: # C, H, W
        img = img[None]
        is_batch = False

    B, in_C, in_H, in_W = img.size()
    img = img.view(-1, in_H, in_W)
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    # print(weights_H.device, indices_H.device, device)
    weights_H, indices_H = weights_H.to(device), indices_H.to(device)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W = weights_W.to(device), indices_W.to(device)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(B*in_C, in_H + sym_len_Hs + sym_len_He, in_W).to(device)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(B*in_C, out_H, in_W).to(device)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[:, i, :] = (img_aug[:, idx:idx + kernel_width, :].transpose(1, 2).matmul(
            weights_H[i][None,:,None].repeat(B*in_C,1, 1))).squeeze()

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(B*in_C, out_H, in_W + sym_len_Ws + sym_len_We).to(device)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(B*in_C, out_H, out_W).to(device)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, :, i] = (out_1_aug[:, :, idx:idx + kernel_width].matmul(
            weights_W[i][None,:,None].repeat(B*in_C, 1, 1))).squeeze()

    out_2 = out_2.contiguous().view(B, in_C, out_H, out_W)
    if not is_batch:
        out_2 = out_2[0]
    return out_2.cpu().numpy().transpose(1, 2, 0) if is_numpy else out_2


if __name__ == '__main__':
    # test imresize function
    # read images
    img = cv2.imread('test.png')
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # imresize
    scale = 1 / 4
    import time
    total_time = 0
    for i in range(10):
        start_time = time.time()
        rlt = imresize(img, scale, antialiasing=True)
        use_time = time.time() - start_time
        total_time += use_time
    print('average time: {}'.format(total_time / 10))

    import torchvision.utils
    torchvision.utils.save_image((rlt * 255).round() / 255, 'rlt.png', nrow=1, padding=0,
                                 normalize=False)
