import copy
from math import sqrt

import numpy as np
import torch


def calc_padding_conv2dtranspose(input_size, kernel, stride, output_size):
    return (-output_size + (input_size - 1) * stride + kernel) / 2


def calc_padding_conv2d(input_size, kernel, stride, output_size):
    return (((output_size - 1) * stride) + kernel - input_size) / 2


if __name__ == '__main__':
    print(calc_padding_conv2d(64, 3, 2, 32))


def recompose(img):
    if len(img.shape) == 3:
        return img
    n_patch = img.shape[0]
    dim_patch = img.shape[1]
    original_height = int(sqrt(n_patch) * dim_patch)
    num_patch_per_row = original_height // dim_patch
    out = np.zeros((original_height, original_height, 8), dtype='float32')

    start_row = stop_row = 0
    start_col = stop_col = 0
    for i in range(n_patch):
        if i % num_patch_per_row == 0:
            start_col = 0
            start_row = stop_row
            stop_row = stop_row + dim_patch
        else:
            start_col = stop_col
        stop_col = start_col + dim_patch
        out[start_row:stop_row, start_col:stop_col, :] = img[i, :, :, :]
    return out


def norm_min_max_channels(data, channels):
    for im in range(data.shape[0]):
        for i in range(channels):
            x = data[im, :, :, i]
            ma = np.max(x)
            mi = np.min(x)
            x = (x - mi) / (ma - mi)
            data[im, :, :, i] = x
    return data


def norm_min_max(data):
    data = copy.deepcopy(data)
    for im in range(data.shape[0]):
        x = data[:, :, :]
        ma = np.max(x)
        mi = np.min(x)
        x = (x - mi) / (ma - mi)
        data[:, :, :] = x
    return data


def norm_linalg(data):
    return data / np.linalg.norm(data, keepdims=True)


def norm_mean(data):
    data = copy.deepcopy(data)
    if len(data.shape) == 4:
        # Keras
        means = np.mean(data, (0, 1, 2))
        stds = np.std(data, (0, 1, 2))
        data = (data - means) / stds
    else:
        # Pytorch
        for l in range(data.shape[0]):
            x = data[l, :, :]
            x = (x - torch.mean(x, 0)) / torch.std(x, 0)
            data[l, :, :] = x
    return data


def inv_norm_mean(original, data):
    m = original.mean()
    std = original.std()
    return (data * std) + m


def norm_max_val(data):
    return data / 2048


def highpass(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs
