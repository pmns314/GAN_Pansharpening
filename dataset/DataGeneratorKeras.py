import copy

import cv2
import h5py

import numpy as np

from sklearn.preprocessing import minmax_scale, MinMaxScaler
import matplotlib.pyplot as plt
from keras.utils.data_utils import Sequence

from utils import *


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, filename, batch_size=32, preprocessing=False, shuffle=True):
        """Initialization"""
        self.file = h5py.File(filename, 'r')

        shape = self.file["gt"].shape
        self.channels = shape[1]
        self.num_samples = shape[0]
        self.dims = shape[2:]
        self.preprocessing = preprocessing
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(self.num_samples)
        self.on_epoch_end()

        self.scaler = MinMaxScaler()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        if index == len(self) - 1:
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def generate_batch(self):
        for idx in range(len(self)):
            yield self[idx]
        self.on_epoch_end()

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""
        # Initialization
        # pan = np.empty((len(indexes), *self.dims, 1))
        # ms = np.empty((len(indexes), *self.dims, self.channels))
        # gt = np.empty((len(indexes), *self.dims, self.channels))

        # Generate data

        indexes = np.sort(indexes)

        if not self.preprocessing:
            pan = np.transpose(self.file["pan"][indexes], (0, 3, 2, 1))
            ms = np.transpose(self.file["lms"][indexes], (0, 3, 2, 1))
            gt = np.transpose(self.file["gt"][indexes], (0, 3, 2, 1))
            ms_lr = np.transpose(self.file["ms"][indexes], (0, 3, 2, 1))
        else:
            pan = np.transpose(highpass(self.file["pan"][indexes]), (0, 3, 2, 1))
            ms = np.transpose(self.file["lms"][indexes], (0, 3, 2, 1))
            gt = np.transpose(self.file["gt"][indexes], (0, 3, 2, 1))
            ms_lr = np.transpose(highpass(self.file["ms"][indexes]), (0, 3, 2, 1))

        pan = norm_mean(pan)
        ms = norm_mean(ms)
        ms_lr = norm_mean(ms_lr)
        gt = norm_mean(gt)

        return (pan, ms, ms_lr), gt


if __name__ == '__main__':
    gen = DataGenerator("../datasets/W3/train.h5")

    a, b, c = next(gen.generate_batch())[0]
    data = b[0:1, :, :, :]
    print(data.shape)

    data1 = norm_min_max(data)

    data2 = norm_mean(data)

    data3 = norm_min_max_channels(data, 8)

    data4 = norm_linalg(data)

    data5 = norm_max_val(data)

    print("Min Max")
    print(np.min(data1[0, :, :, 0]), (np.max(data1[0, :, :, 0])))
    print("Mean")
    print(np.min(data2[0, :, :, 0]), np.max(data2[0, :, :, 0]))
    print("Min Max channels")
    print(np.min(data3[0, :, :, 0]), np.max(data3[0, :, :, 0]))
    print("linalg")
    print(np.min(data4[0, :, :, 0]), np.max(data4[0, :, :, 0]))
    print("Max Val")
    print(np.min(data5[0, :, :, 0]), np.max(data5[0, :, :, 0]))

    aa1 = data1[0, :, :, 2::-1]
    aa2 = data2[0, :, :, 2::-1]
    aa3 = data3[0, :, :, 2::-1]
    aa4 = data4[0, :, :, 2::-1]
    aa5 = data5[0, :, :, 2::-1]

    plt.imshow(aa1)
    plt.figure()
    plt.imshow(aa2)
    plt.figure()
    plt.imshow(aa3)
    plt.figure()
    plt.imshow(aa4)
    plt.figure()
    plt.imshow(aa5)
    plt.show()
