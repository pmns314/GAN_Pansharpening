import os
import shutil

import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from utils import create_patches, augment_data

if __name__ == '__main__':
    input_folder = f"..\\data\\FR"
    output_folder = f"../datasets/KFold"
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
        print(f"Generation {satellite}")
        satellite_input_folder = os.path.join(input_folder, satellite)
        satellite_output_folder = os.path.join(output_folder, satellite)
        if os.path.exists(satellite_output_folder):
            shutil.rmtree(satellite_output_folder)
        os.mkdir(satellite_output_folder)
        with open(f"{satellite_output_folder}/dataset_info.txt", "w") as info:
            info.write(f"Dataset Satellite: {satellite}\n")
            cnt = 1
            sat_dir = os.listdir(satellite_input_folder)
            for filename in tqdm(sat_dir):
                info.write(f"\nImage {cnt} name: {filename}\n")
                # Load Images in Memory
                mat = loadmat(os.path.join(satellite_input_folder, filename))
                gt = np.array(mat['I_GT'])
                pan = np.array(mat['I_PAN'])
                pan = np.reshape(pan, (*pan.shape, 1))
                ms = np.array(mat['I_MS'])
                ms_lr = np.array(mat['I_MS_LR'])
                test_data = {}

                H, W = gt.shape[:2]
                h, w = ms_lr.shape[:2]

                gt_1 = gt[:H // 2, :W // 2, :]
                gt_2 = gt[H // 2:, :W // 2, :]
                gt_3 = gt[:H // 2, W // 2:, :]
                gt_4 = gt[H // 2:, W // 2:, :]
                gts = [gt_1, gt_2, gt_3, gt_4]

                pan_1 = pan[:H // 2, :W // 2, :]
                pan_2 = pan[H // 2:, :W // 2, :]
                pan_3 = pan[:H // 2, W // 2:, :]
                pan_4 = pan[H // 2:, W // 2:, :]
                pans = [pan_1, pan_2, pan_3, pan_4]

                ms_1 = ms[:H // 2, :W // 2, :]
                ms_2 = ms[H // 2:, :W // 2, :]
                ms_3 = ms[:H // 2, W // 2:, :]
                ms_4 = ms[H // 2:, W // 2:, :]
                mss = [ms_1, ms_2, ms_3, ms_4]

                ms_lr_1 = ms_lr[:h // 2, :w // 2, :]
                ms_lr_2 = ms_lr[h // 2:, :w // 2, :]
                ms_lr_3 = ms_lr[:h // 2, w // 2:, :]
                ms_lr_4 = ms_lr[h // 2:, w // 2:, :]
                ms_lrs =[ms_lr_1, ms_lr_2, ms_lr_3, ms_lr_4]

                for quadrante in range(4):
                    with h5py.File(f"{satellite_output_folder}/test_{cnt}_{quadrante}_64.h5", 'a') as f_test:
                        patch_gt_test = create_patches(gts[quadrante], 64, 64)
                        patch_pan_test = create_patches(pans[quadrante], 64, 64)
                        patch_ms_test = create_patches(mss[quadrante], 64, 64)
                        patch_ms_lr_test = create_patches(ms_lrs[quadrante], 16, 16)

                        f_test.create_dataset('gt', data=np.transpose(patch_gt_test, (0, 3, 1, 2)))
                        f_test.create_dataset('pan', data=np.transpose(patch_pan_test, (0, 3, 1, 2)))
                        f_test.create_dataset('ms', data=np.transpose(patch_ms_test, (0, 3, 1, 2)))
                        f_test.create_dataset('ms_lr', data=np.transpose(patch_ms_lr_test, (0, 3, 1, 2)))

                    with h5py.File(f"{satellite_output_folder}/train_{cnt}_{quadrante}_64.h5", 'a') as f_train:
                        patch_gt_train = create_patches(gts[quadrante], 64, 16)
                        patch_pan_train = create_patches(pans[quadrante], 64, 16)
                        patch_ms_train = create_patches(mss[quadrante], 64, 16)
                        patch_ms_lr_train = create_patches(ms_lrs[quadrante], 16, 4)

                        patch_gt_aug = augment_data(patch_gt_train)
                        patch_pan_aug = augment_data(patch_pan_train)
                        patch_ms_aug = augment_data(patch_ms_train)
                        patch_ms_lr_aug = augment_data(patch_ms_lr_train)

                        f_train.create_dataset('gt', data=np.transpose(patch_gt_aug, (0, 3, 1, 2)))
                        f_train.create_dataset('pan', data=np.transpose(patch_pan_aug, (0, 3, 1, 2)))
                        f_train.create_dataset('ms', data=np.transpose(patch_ms_aug, (0, 3, 1, 2)))
                        f_train.create_dataset('ms_lr', data=np.transpose(patch_ms_lr_aug, (0, 3, 1, 2)))

                cnt+=1
            info.write(f"\nNum Patch test: {patch_gt_test.shape[0]}\n")
            info.write(f"\nNum Patch train: {patch_gt_aug.shape[0]}\n")

    print("Dataset Created")
