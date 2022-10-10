import argparse
import os
import shutil

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import io as scio
from torch.utils.data import DataLoader

import constants
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.CNNs.APNN import APNN
from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_model',
                        default='test',
                        help='Provide name of the model. Defaults to test',
                        type=str
                        )
    parser.add_argument('-d', '--dataset_path',
                        default=f'{constants.DATASET_DIR}',
                        help='Provide path to the dataset. Defaults to ROOT/datasets',
                        type=str
                        )
    args = parser.parse_args()

    satellite = "W3"
    dataset_path = args.dataset_path
    model_name = args.name_model

    model_name = "W3_1_32_01_mse"

    result_folder = f"./results/{satellite}_full_hdf5"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    tests = ["test_1_256.h5", "test_3_512.h5"]
    for test_index in range(len(tests)):
        test = tests[test_index]
        test_set_path = f"{dataset_path}/{satellite}/{test}"

        test_dataloader = DataLoader(DatasetPytorch(test_set_path),
                                     batch_size=64,
                                     shuffle=False)

        model_result_folder = f"{result_folder}/{model_name}"
        if not os.path.exists(model_result_folder):
            os.makedirs(model_result_folder)

        model = APNN(test_dataloader.dataset.channels)

        pan, ms, _, gt = next(enumerate(test_dataloader))[1]
        if len(pan.shape) == 3:
            pan = torch.unsqueeze(pan, 0)

        gt = torch.permute(gt, (0, 2, 3, 1))
        gt = gt.detach().numpy()

        filename = f"{model_result_folder}/test_{test_index}.h5"
        if not os.path.exists(filename):
            initialized = False
        else:
            initialized = True

        [patch, channels, height, width] = gt.shape
        with h5py.File(filename, "a") as f:
            model_path = f"pytorch_models\\trained_models\\W3_all\\{model_name}"
            if not initialized:
                f.create_dataset("gt", data=gt, dtype='d')
                f.create_dataset("gen", data=gt, maxshape=(None, channels, height, width), dtype='d')
                start = 1
            else:
                start = f["gen"].shape[0]+1
            m = torch.load(f"{model_path}\\model.pth", map_location=torch.device('cpu'))
            num_chks = m['tot_epochs']

            print(f"Start:{start}   Stop: {num_chks}")
            for i in tqdm(range(start, num_chks+1)):
                trained_model = torch.load(f"{model_path}\\checkpoints\\checkpoint_{i}.pth",
                                           map_location=torch.device('cpu'))
                model.load_state_dict(trained_model['model_state_dict'])

                gen = model(ms, pan)
                # From NxCxHxW to NxHxWxC
                gen = torch.permute(gen, (0, 2, 3, 1))
                gen = gen.detach().numpy()

                f["gen"].resize((f["gen"].shape[0] + gen.shape[0]), axis=0)
                f["gen"][-gen.shape[0]:] = gen

