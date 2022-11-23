import os
import shutil

import h5py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from utils import create_patches, augment_data

if __name__ == '__main__':
    input_folder = f"..\\data\\FR"
    output_folder = f"..\\datasets\\FR2"
    dim_patch = 64
    overlap = 16
    ratio = 4
    gt = []
    pan = []
    ms = []
    ms_lr = []

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    for satellite in os.listdir(input_folder):
        satellite_input_folder = os.path.join(input_folder, satellite)
        satellite_output_folder = os.path.join(output_folder, satellite)
        if os.path.exists(satellite_output_folder):
            shutil.rmtree(satellite_output_folder)
        os.mkdir(satellite_output_folder)
        with open(f"{satellite_output_folder}/dataset_info.txt", "w") as info:
            info.write(f"Dataset Satellite: {satellite}\n")
            cnt = 1
            train = True
            for filename in os.listdir(satellite_input_folder):
                info.write(f"\nImage {cnt} name: {filename}\n")
                # Load Images in Memory
                mat = loadmat(os.path.join(satellite_input_folder, filename))
                gt = np.array(mat['I_GT'])
                pan = np.array(mat['I_PAN'])
                pan = np.reshape(pan, (*pan.shape, 1))
                ms = np.array(mat['I_MS'])
                ms_lr = np.array(mat['I_MS_LR'])

                for dim_patch in [2 ** e for e in range(6, int(np.log2(gt.shape[0])) + 1)]:
                    test_data = {}
                    if train:
                        with h5py.File(f"{satellite_output_folder}/train_{cnt}_{dim_patch}.h5", 'a') as f_train:
                            with h5py.File(f"{satellite_output_folder}/val_{cnt}_{dim_patch}.h5", 'a') as f_val:
                                for (img, name, dim_patch_img, stride_img) in [(gt, 'gt', dim_patch, overlap),
                                                                               (pan, 'pan', dim_patch, overlap),
                                                                               (ms_lr, 'ms_lr', dim_patch // ratio,
                                                                                overlap // ratio),
                                                                               (ms, 'ms', dim_patch, overlap)]:
                                    # Extracting 1/4 of the image for testing
                                    H, W = img.shape[:2]
                                    img_1 = img[:H // 2, :W // 2, :]
                                    img_2 = img[H // 2:, :W // 2, :]
                                    img_3 = img[:H // 2, W // 2:, :]
                                    img_4 = img[H // 2:, W // 2:, :]

                                    img_train = np.concatenate((img_1, img_2, img_3))
                                    test_data[name] = img_4

                                    # Creating Training Patches
                                    patch_img_train_original = create_patches(img_train, dim_patch_img, stride_img)
                                    patch_img_train_aug = augment_data(patch_img_train_original)

                                    # Division in Training 90 % training - 10 % validation
                                    np.random.shuffle(patch_img_train_aug)
                                    total_patch = patch_img_train_aug.shape[0]
                                    index_val = int(total_patch * .9)
                                    patch_img_train = patch_img_train_aug[:index_val, :, :, :]
                                    patch_img_val = patch_img_train_aug[index_val:, :, :, :]

                                    # Saving data
                                    f_train.create_dataset(name, data=patch_img_train)
                                    f_val.create_dataset(name, data=patch_img_val)

                                info.write(f"\nTrain File: train_{cnt}_{dim_patch}\n")
                                info.write(f"\tNum Patches: {patch_img_train.shape[0]}\n")
                                info.write(f"\tPatch Size: {patch_img_train.shape[1]}x{patch_img_train.shape[2]}\n")
                                info.write(f"Validation File: val_{cnt}_{dim_patch}\n")
                                info.write(f"\tNum Patches: {patch_img_val.shape[0]}\n")
                                info.write(f"\tPatch Size: {patch_img_val.shape[1]}x{patch_img_val.shape[2]}\n\n")
                        # Save Dataset
                        train = False
                    else:
                        test_data['gt'] = gt
                        test_data['pan'] = pan
                        test_data['ms'] = ms
                        test_data['ms_lr'] = ms_lr

                    with h5py.File(os.path.join(satellite_output_folder, f'test_{cnt}_{dim_patch}.h5'), 'a') as f_test:

                        # Creating Training Patches
                        patch_gt_test = create_patches(test_data['gt'], dim_patch, dim_patch)
                        patch_pan_test = create_patches(test_data['pan'], dim_patch, dim_patch)
                        patch_ms_test = create_patches(test_data['ms'], dim_patch, dim_patch)
                        patch_ms_lr_test = create_patches(test_data['ms_lr'], dim_patch, dim_patch)

                        # Saving data
                        f_test.create_dataset('gt', data=patch_gt_test)
                        f_test.create_dataset('pan', data=patch_pan_test)
                        f_test.create_dataset('ms', data=patch_ms_test)
                        f_test.create_dataset('ms_lr', data=patch_ms_lr_test)

                        info.write(f"Test File: test_{cnt}_{dim_patch}\n")
                        info.write(f"\tNum Patches: {patch_gt_test.shape[0]}\n")
                        info.write(f"\tPatch Size: {patch_gt_test.shape[1]}x{patch_gt_test.shape[2]}\n")
                cnt += 1
        #     # Patch Division
        #     patch_gt = create_patches(gt, dim_patch, overlap)
        #     patch_pan = create_patches(pan, dim_patch, overlap)
        #     patch_ms = create_patches(ms, dim_patch, overlap)
        #     patch_ms_lr = create_patches(ms_lr, dim_patch // ratio, overlap // ratio)
        #
        #     # Data augmentation
        #     patch_gt_aug = augment_data(patch_gt)
        #     patch_pan_aug = augment_data(patch_pan)
        #     patch_ms_aug = augment_data(patch_ms)
        #     patch_ms_lr_aug = augment_data(patch_ms_lr)
        #
        # # Saving data
        # pan = np.transpose(np.expand_dims(np.array(pan, dtype=np.float32), -1), (0, 3, 1, 2))
        # gt = np.transpose(np.array(gt, dtype=np.float32), (0, 3, 1, 2))
        # ms = np.transpose(np.array(ms, dtype=np.float32), (0, 3, 1, 2))
        # ms_lr = np.transpose(np.array(ms_lr, dtype=np.float32), (0, 3, 1, 2))
        #
        # with h5py.File(os.path.join(dataset_folder, 'train.h5'), 'a') as f:
        #     f.create_dataset('pan', data=pan[:train_index])
        #     f.create_dataset('gt', data=gt[:train_index])
        #     f.create_dataset('lms', data=ms[:train_index])
        #     f.create_dataset('ms', data=ms_lr[:train_index])
        #
        # with h5py.File(os.path.join(dataset_folder, 'val.h5'), 'a') as f:
        #     f.create_dataset('pan', data=pan[train_index:train_index + val_index])
        #     f.create_dataset('gt', data=gt[train_index:train_index + val_index])
        #     f.create_dataset('lms', data=ms[train_index:train_index + val_index])
        #     f.create_dataset('ms', data=ms_lr[train_index:train_index + val_index])
        #
        # with h5py.File(os.path.join(dataset_folder, 'test.h5'), 'a') as f:
        #     f.create_dataset('pan', data=pan[train_index + val_index:])
        #     f.create_dataset('gt', data=gt[train_index + val_index:])
        #     f.create_dataset('lms', data=ms[train_index + val_index:])
        #     f.create_dataset('ms', data=ms_lr[train_index + val_index:])

    print("Dataset Created")
