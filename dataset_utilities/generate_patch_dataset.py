import os
import shutil

import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from utils import create_patches, augment_data

if __name__ == '__main__':

    for resolution in ["FR", "RR"]:
        print(f"Generation {resolution} dataset")
        input_folder = f"..\\data\\{resolution}"
        base_output_folder = f"../datasets/{resolution}"
        dim_patch = 64
        overlap = 16
        ratio = 4
        gt = []
        pan = []
        ms = []
        ms_lr = []

        # Settings for the Dataset Division made up of:
        # filename, flag_for_training, flag_for_validation, flag_for_test
        settings = [("Test", False, False, True),
                    ("Train", True, False, False),
                    ("Train&Val", True, True, False),
                    ("Train&Val&Test", True, True, True)]

        for folder_name, extract_training, extract_validation, extract_testing in settings:
            print(f"Generating {folder_name}")
            output_folder = f"{base_output_folder}/{folder_name}"
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.mkdir(output_folder)

            for satellite in os.listdir(input_folder):
                print(f"Generation patches for {satellite}")
                satellite_input_folder = os.path.join(input_folder, satellite)
                satellite_output_folder = os.path.join(output_folder, satellite)
                if os.path.exists(satellite_output_folder):
                    shutil.rmtree(satellite_output_folder)
                os.mkdir(satellite_output_folder)
                with open(f"{satellite_output_folder}/dataset_info.txt", "w") as info:
                    info.write(f"Dataset Satellite: {satellite}\n")
                    cnt = 1
                    for filename in tqdm(os.listdir(satellite_input_folder)):
                        info.write(f"\nImage {cnt} name: {filename}\n")
                        # Load Images in Memory
                        mat = loadmat(os.path.join(satellite_input_folder, filename))
                        gt = np.array(mat['I_GT'])
                        pan = np.array(mat['I_PAN'])
                        pan = np.reshape(pan, (*pan.shape, 1))
                        ms = np.array(mat['I_MS'])
                        ms_lr = np.array(mat['I_MS_LR'])
                        test_data = {}
                        for dim_patch in [2 ** e for e in range(6, int(np.log2(gt.shape[0])) + 1)]:
                            if extract_training:
                                with h5py.File(f"{satellite_output_folder}/train_{cnt}_{dim_patch}.h5", 'a') as f_train:
                                    with h5py.File(f"{satellite_output_folder}/val_{cnt}_{dim_patch}.h5", 'a') as f_val:
                                        p = None
                                        for (img, name, dim_patch_img, stride_img) in [(gt, 'gt', dim_patch, overlap),
                                                                                       (pan, 'pan', dim_patch, overlap),
                                                                                       (ms_lr, 'ms_lr', dim_patch // ratio,
                                                                                        overlap // ratio),
                                                                                       (ms, 'ms', dim_patch, overlap)]:

                                            H, W = img.shape[:2]
                                            img_1 = img[:H // 2, :W // 2, :]
                                            img_2 = img[H // 2:, :W // 2, :]
                                            img_3 = img[:H // 2, W // 2:, :]
                                            img_4 = img[H // 2:, W // 2:, :]

                                            if extract_testing:
                                                img_train = np.concatenate((img_1, img_2, img_3))
                                                test_data[name] = img_4
                                            else:
                                                img_train = np.concatenate((img_1, img_2, img_3, img_4))

                                            # Creating Training Patches
                                            patch_img_train_original = create_patches(img_train, dim_patch_img, stride_img)
                                            patch_img_train_aug = augment_data(patch_img_train_original)

                                            if extract_validation:
                                                # Divisione 90% - 10%
                                                if p is None:
                                                    p = np.random.permutation(patch_img_train_aug.shape[0])
                                                patch_img_train_aug = patch_img_train_aug[p]
                                                total_patch = patch_img_train_aug.shape[0]
                                                index_val = int(total_patch * .9)
                                                patch_img_train = patch_img_train_aug[:index_val, :, :, :]
                                                patch_img_val = patch_img_train_aug[index_val:, :, :, :]

                                                val_data = np.transpose(patch_img_val, (0, 3, 1, 2))
                                                f_val.create_dataset(name, data=val_data)

                                            else:
                                                patch_img_train = patch_img_train_aug

                                            train_data = np.transpose(patch_img_train, (0, 3, 1, 2))
                                            f_train.create_dataset(name, data=train_data)
                                if patch_img_train.shape[0] == 0:
                                    os.remove(f"{satellite_output_folder}/train_{cnt}_{dim_patch}.h5")
                                    os.remove(f"{satellite_output_folder}/val_{cnt}_{dim_patch}.h5")
                                    continue

                                info.write(f"\nTrain File: train_{cnt}_{dim_patch}\n")
                                info.write(f"\tNum Patches: {patch_img_train.shape[0]}\n")
                                info.write(f"\tPatch Size: {patch_img_train.shape[1]}x{patch_img_train.shape[2]}\n")
                                if extract_validation:
                                    info.write(f"Validation File: val_{cnt}_{dim_patch}\n")
                                    info.write(f"\tNum Patches: {patch_img_val.shape[0]}\n")
                                    info.write(f"\tPatch Size: {patch_img_val.shape[1]}x{patch_img_val.shape[2]}\n\n")
                                else:
                                    os.remove(f"{satellite_output_folder}/val_{cnt}_{dim_patch}.h5")
                            if extract_testing:
                                # Generazione Test
                                if not bool(test_data):
                                    test_data['gt'] = gt
                                    test_data['pan'] = pan
                                    test_data['ms'] = ms
                                    test_data['ms_lr'] = ms_lr

                                # Creating Training Patches without overlap
                                patch_gt_test = create_patches(test_data['gt'], dim_patch, dim_patch)
                                if patch_gt_test.shape[0] == 0:
                                    continue
                                patch_pan_test = create_patches(test_data['pan'], dim_patch, dim_patch)
                                patch_ms_test = create_patches(test_data['ms'], dim_patch, dim_patch)
                                patch_ms_lr_test = create_patches(test_data['ms_lr'], dim_patch // ratio,
                                                                  dim_patch // ratio)

                                with h5py.File(os.path.join(satellite_output_folder, f'test_{cnt}_{dim_patch}.h5'),
                                               'a') as f_test:
                                    # Saving data
                                    f_test.create_dataset('gt', data=np.transpose(patch_gt_test, (0, 3, 1, 2)))
                                    f_test.create_dataset('pan', data=np.transpose(patch_pan_test, (0, 3, 1, 2)))
                                    f_test.create_dataset('ms', data=np.transpose(patch_ms_test, (0, 3, 1, 2)))
                                    f_test.create_dataset('ms_lr', data=np.transpose(patch_ms_lr_test, (0, 3, 1, 2)))

                                    info.write(f"Test File: test_{cnt}_{dim_patch}\n")
                                    info.write(f"\tNum Patches: {patch_gt_test.shape[0]}\n")
                                    info.write(f"\tPatch Size: {patch_gt_test.shape[1]}x{patch_gt_test.shape[2]}\n")
                        cnt += 1

    print("Dataset Created")
