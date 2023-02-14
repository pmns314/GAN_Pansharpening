from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch


def create_patches(img, dim_patch, stride):
    """ Divides the given image in patches

    Parameters
    ----------

    img : ndarray
        the image to divide in patches
    dim_patch : int
        dimension of the single patch
    stride : int
        number of pixel stride between adjacent patches.
        If no overlapping is desired, set it to the same value of dim_patch

    """
    dim_im_row, dim_im_col = img.shape[:2]
    channels = img.shape[2]

    if stride == 0:
        stride = dim_patch
    starts_row = range(0, dim_im_row - dim_patch + 1, stride)
    stops_row = range(dim_patch, dim_im_row + 1, stride)
    starts_col = range(0, dim_im_col - dim_patch + 1, stride)
    stops_col = range(dim_patch, dim_im_col + 1, stride)

    zz_row = list(zip(starts_row, stops_row))
    zz_col = list(zip(starts_col, stops_col))

    patches = np.empty((len(zz_row) * len(zz_col), dim_patch, dim_patch, channels))
    cnt = 0
    for start_row, end_row in zz_row:
        for start_col, end_col in zz_col:
            patches[cnt, :, :, :] = img[start_row:end_row, start_col:end_col, :]
            cnt += 1

    return patches


def augment_data(original):
    """ Augment the input data with a horizontal and vertical flip respectively.
    The number of output patches is three times the input

    Parameters
    ----------
    original:ndarray
        the data to augment. Must be at least three-dimensional
    """
    flip_ud = np.flip(original, 1)
    flip_lr = np.flip(original, 2)
    # flip_ud_lr = np.flip(original, (1, 2))

    return np.concatenate((original, flip_ud, flip_lr))


def recompose(img):
    """ Composes image from patches.
    Patches are placed from the top-left to the bottom-right

    Parameters
    ----------
    img : ndarray
        array of patches to compose the output from

    """
    if len(img.shape) == 3:
        return img
    n_patch = img.shape[0]
    dim_patch = img.shape[1]
    channels = img.shape[3]
    original_height = int(sqrt(n_patch) * dim_patch)
    num_patch_per_row = original_height // dim_patch
    out = np.zeros((original_height, original_height, channels), dtype='float32')

    start_row = stop_row = 0
    stop_col = 0
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


def adjust_image(img, ms_lr=None):
    """  Transforms the input torch tensor into a numpy image

    Parameters
    ----------
    img : torch.Tensor
        tensor of patches. Must be four-dimensional
    ms_lr: torch.Tensor, optional
        if set, it equalizes img with respect to this data

    """
    img = torch.permute(img, (0, 2, 3, 1)).detach().cpu().numpy()
    img = recompose(img)
    img = np.squeeze(img)
    if ms_lr is None:
        return img * 2048.0

    np.clip(img, 0, 1, out=img)
    img = img * 2048.0
    ms_lr = torch.permute(ms_lr, (0, 2, 3, 1)).detach().cpu().numpy()
    ms_lr = recompose(ms_lr)
    ms_lr = np.squeeze(ms_lr) * 2048.0
    mgen = np.mean(img, (0, 1)) + 1e-12
    mgt = np.mean(ms_lr, (0, 1))
    img = (img - mgen) + mgt
    return np.round(img)


def norm_max_val(data):
    """ Normalizes the data according to the maximum value possible ( 2**11 )
    Parameters
    ----------
    data : ndarray
        data to normalize
    """

    return data / 2048


tol1 = [104, 103, 33]
tol2 = [215, 664, 941]


def linear_strech(data, calculate_limits=True):
    """ Performs a linear stretching of the image.
    Parameters
    ----------
        data : ndarray
            data to stretch
        calculate_limits: bool, optional
            if True, calculates new limit values for the stretching of each band.
            Otherwise, uses current limits

    """
    global tol1
    global tol2
    N, M = data.shape[:2]
    NM = N * M
    data = (data * 2048).astype(int).astype(float)
    for i in [0, 1, 2]:
        band = (data[:, :, i]).flatten()

        if calculate_limits:
            hb, levelb = np.histogram(band, int(np.max(band) - np.min(band)))
            chb = np.cumsum(hb)
            tol1[i] = np.ceil(levelb[np.where(chb > NM * 0.01)[0][0]])
            tol2[i] = np.ceil(levelb[np.where(chb < NM * 0.99)[0][-1]])

        t_1 = tol1[i]
        t_2 = tol2[i]
        band[band < t_1] = t_1
        band[band > t_2] = t_2
        band = (band - t_1) / (t_2 - t_1)
        data[:, :, i] = np.reshape(band, (N, M))

    return data


def view_image(data, calculate_limits=True):
    """ Display image as RGB after performing linear stretching on the extracted bands,
    If input has 8 channels, bands [0,2,4] are selected for display. Otherwise, bands [0,1,2] are taken

    Parameters
    ----------

    data : ndarray
        data to display
    calculate_limits : bool, optional
        if True, calculates new limits for linear stretching
    """
    channels = data.shape[2]
    if channels == 8:
        xx = linear_strech(data[:, :, (0, 2, 4)], calculate_limits)
    else:
        xx = linear_strech(data[:, :, (0, 1, 2)], calculate_limits)
    plt.figure()
    plt.imshow((xx[:, :, ::-1]))
