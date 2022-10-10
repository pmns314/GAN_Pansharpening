import os
import shutil

from scipy.io import loadmat
import h5py
import imageio
import numpy as np
from scipy.io import loadmat

if __name__ == '__main__':
    images_folder = "..\\data\\original_workspace"
    output_folder = "..\\datasets"

    gt = []
    pan = []
    ms = []
    ms_lr = []

    for satellite in os.listdir(images_folder):
        satellite_folder = os.path.join(images_folder, satellite)
        gt = []
        pan = []
        ms = []
        ms_lr = []
        dataset_folder = os.path.join(output_folder, satellite)
        for workspace in os.listdir(satellite_folder):
            mat = loadmat(os.path.join(satellite_folder, workspace))
            gt.append(mat['I_GT'])
            pan.append(mat['I_PAN'])
            ms.append(mat['I_MS'])
            ms_lr.append(mat['I_MS_LR'])

        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)
        os.mkdir(dataset_folder)

        # 70% train - 15% val - 15% test
        n = len(gt)
        if n == 3:
            train_index = 1
            val_index = 1
        else:
            train_index = round(n * .7)
            val_index = round(n * .15)

        # Saving data
        pan = np.transpose(np.expand_dims(np.array(pan, dtype=np.float32), -1), (0, 3, 1, 2))
        gt = np.transpose(np.array(gt, dtype=np.float32), (0, 3, 1, 2))
        ms = np.transpose(np.array(ms, dtype=np.float32), (0, 3, 1, 2))
        ms_lr = np.transpose(np.array(ms_lr, dtype=np.float32), (0, 3, 1, 2))

        with h5py.File(os.path.join(dataset_folder, 'train.h5'), 'a') as f:
            f.create_dataset('pan', data=pan[:train_index])
            f.create_dataset('gt', data=gt[:train_index])
            f.create_dataset('lms', data=ms[:train_index])
            f.create_dataset('ms', data=ms_lr[:train_index])

        with h5py.File(os.path.join(dataset_folder, 'val.h5'), 'a') as f:
            f.create_dataset('pan', data=pan[train_index:train_index+val_index])
            f.create_dataset('gt', data=gt[train_index:train_index+val_index])
            f.create_dataset('lms', data=ms[train_index:train_index+val_index])
            f.create_dataset('ms', data=ms_lr[train_index:train_index+val_index])

        with h5py.File(os.path.join(dataset_folder, 'test.h5'), 'a') as f:
            f.create_dataset('pan', data=pan[train_index+val_index:])
            f.create_dataset('gt', data=gt[train_index+val_index:])
            f.create_dataset('lms', data=ms[train_index+val_index:])
            f.create_dataset('ms', data=ms_lr[train_index+val_index:])

    print("Dataset Created")
