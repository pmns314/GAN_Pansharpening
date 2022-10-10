import matlab.engine
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import io as scio
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.CNNs.APNN import APNN
import xlrd as xl  # Import xlrd package

ratio = 4.0
L = 11
Qblocks_size = 32.0
flag_cut_bounds = 1
dim_cut = 21
thvalues = 0
if __name__ == '__main__':

    eng = matlab.engine.start_matlab()
    s = eng.genpath("C:\\Users\\pmans\\Documents\Magistrale\\Remote Sensing\\Materiale Tesi\\"
                    "DLPan-Toolbox-main\\DLPan-Toolbox-main")
    eng.addpath(s, nargout=0)

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

        result_folder = f"./results/{satellite}_full4"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        result_model_folder = f"{result_folder}/{model_name}"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        model_folder = f"./pytorch_models/trained_models/{satellite}_all/{model_name}"
        model_path = f"{model_folder}/model.pth"
        tests = ["test_1_256", "test_3_512"]
        for test_index in [0, 1]:
            test_folder = f"{result_model_folder}/test_{test_index}"
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            test_set_path = dataset_path + satellite + f"/{tests[test_index]}.h5"
            if os.path.exists(test_set_path):
                test_dataloader = DataLoader(DatasetPytorch(test_set_path),
                                             batch_size=64,
                                             shuffle=False)

                model = APNN(test_dataloader.dataset.channels)

                # Read Excel
                filename = f"{result_model_folder}/test_{test_index}.csv"

                if os.path.exists(filename):
                    cc = pd.read_csv(filename, sep=";")
                    start = int(cc['Epochs'].iloc[-1] + 1)
                else:
                    start = 1

                stop = len(os.listdir(f"{model_folder}/checkpoints"))

                # Generation Images
                pan, ms, _, gt = next(enumerate(test_dataloader))[1]
                if len(pan.shape) == 3:
                    pan = torch.unsqueeze(pan, 0)
                gt = torch.permute(gt, (0, 2, 3, 1))
                gt_ten = gt
                gt = gt.detach().numpy()
                gt = gt.astype('double')
                gt = np.squeeze(gt) * 2048
                for i in tqdm(range(start, stop)):
                    df = pd.DataFrame(columns=["Epochs", "Q2n", "Q_avg", "SAM", "ERGAS", "SCC"])
                    # Load Pre trained Model
                    trained_model = torch.load(f"{model_folder}/checkpoints/checkpoint_{i}.pth",
                                               map_location=torch.device('cpu'))
                    model.load_state_dict(trained_model['model_state_dict'])

                    gen = model(ms, pan)
                    # From NxCxHxW to NxHxWxC
                    gen = torch.permute(gen, (0, 2, 3, 1))
                    gen_ten = gen
                    gen = gen.detach().numpy()
                    gen = gen.astype('double')
                    gen = np.squeeze(gen) * 2048
                    Q_avg, SAM, ERGAS, SCC, Q = eng.indexes_evaluation(
                        matlab.double(gen.tolist()),
                        matlab.double(gt.tolist()),
                        ratio, L, Qblocks_size, flag_cut_bounds, dim_cut, thvalues, nargout=5)
                    row = [i, Q, Q_avg, SAM, ERGAS, SCC]
                    # df.loc[df.index[-1]] = row
                    df.loc[0] = row

                    df.to_csv(filename, index=False, header=True if i == 1 else False, mode='a', sep=";")
