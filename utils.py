import copy
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from pytorch_models import *


def calc_padding_conv2dtranspose(input_size, kernel, stride, output_size):
    return (-output_size + (input_size - 1) * stride + kernel) / 2


def calc_padding_conv2d(input_size, kernel, stride, output_size):
    return (((output_size - 1) * stride) + kernel - input_size) / 2


def create_patches(img, dim_patch, stride):
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
    flip_ud = np.flip(original, 1)
    flip_lr = np.flip(original, 2)
    # flip_ud_lr = np.flip(original, (1, 2))

    return np.concatenate((original, flip_ud, flip_lr))


def recompose(img):
    if len(img.shape) == 3:
        return img
    n_patch = img.shape[0]
    dim_patch = img.shape[1]
    channels = img.shape[3]
    original_height = int(sqrt(n_patch) * dim_patch)
    num_patch_per_row = original_height // dim_patch
    out = np.zeros((original_height, original_height, channels), dtype='float32')

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


def adjust_image(img, ms_lr=None):
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
    img = (img / mgen) * mgt
    return np.round(img)


def norm_min_max(data):
    data = copy.deepcopy(data)
    for im in range(data.shape[0]):
        x = data[:, :, :]
        ma = np.max(x)
        mi = np.min(x)
        x = (x - mi) / (ma - mi)
        data[:, :, :] = x
    return data


def norm_mean(data):
    data = copy.deepcopy(data)

    # Pytorch
    for l in range(data.shape[0]):
        x = data[l, :, :]
        x = (x - torch.mean(x, 0)) / torch.std(x, 0)
        data[l, :, :] = x

    return data


def norm_max_val(data):
    return data / 2048


tol1 = [104, 103, 33]
tol2 = [215, 664, 941]


def linear_strech(data, calculate_limits=True):
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
    channels = data.shape[2]
    if channels == 8:
        xx = linear_strech(data[:, :, (0, 2, 4)], calculate_limits)
    else:
        xx = linear_strech(data[:, :, (0, 1, 2)], calculate_limits)
    plt.figure()
    plt.imshow((xx[:, :, ::-1]))


def create_model(network_type: str, channels: int, device: str = "cpu", evaluation=False, **kwargs):
    """ Creates the model network

    Parameters
    ----------
    network_type : str
        type of the network. It must match one of the available networks (case-insensitive) otherwise raises KeyError
    channels : int
        number of channels of the data the network will handle
    device : str
        the device onto which train the network (either cpu or a cuda visible device)
    evaluation : bool
        True if the networks must be defined for evaluation False otherwise.
        If True, definition doesn't need loss and optimizer
    """
    network_type = network_type.strip().upper()
    try:
        model = GANS[network_type].value(channels, device)

        if not evaluation:
            # Adversarial Loss Definition
            adv_loss = None if kwargs['adv_loss_fn'] is None else kwargs['adv_loss_fn'].strip().upper()
            if adv_loss is not None:
                adv_loss = AdvLosses[adv_loss].value()

            # Reconstruction Loss Definition
            rec_loss = None if kwargs['loss_fn'] is None else kwargs['loss_fn'].strip().upper()
            if rec_loss is not None:
                print(rec_loss)
                rec_loss = Losses[rec_loss].value()

            model.define_losses(rec_loss=rec_loss, adv_loss=adv_loss)
        return model
    except KeyError:
        pass

    try:
        model = CNNS[network_type].value(channels, device)

        if not evaluation:
            # Reconstruction Loss Definition
            loss_fn = None if kwargs['loss_fn'] is None else kwargs['loss_fn'].strip().upper()
            if loss_fn is not None:
                loss_fn = Losses[loss_fn].value()

            # Optimizer definition
            optimizer = None if kwargs['optimizer'] is None else kwargs['optimizer'].strip().upper()
            if optimizer is not None:
                optimizer = Optimizers[optimizer].value(model.parameters(), lr=kwargs['lr'])

            model.compile(loss_fn, optimizer)
        else:
            model.compile()

        return model
    except KeyError:
        pass

    raise KeyError(f"Model not defined!\n\n"
                   f"Use one of the followings:\n"
                   f"GANS: \t{[e.name for e in GANS]}\n"
                   f"CNN: \t{[e.name for e in CNNS]}\n"
                   f"Losses: \t{[e.name for e in Losses]}\n"
                   f"Adversarial Losses: \t{[e.name for e in AdvLosses]}\n"
                   f"Optimizers: \t{[e.name for e in Optimizers]}\n"
                   )