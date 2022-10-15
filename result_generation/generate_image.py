""" Loads the model, generates the high resolution image and saves it a .mat file"""
import argparse
import os
import shutil

import matlab.engine
import numpy as np
import torch
from scipy import io as scio
from torch.utils.data import DataLoader

import constants
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.CNNs.APNN import APNN
from pytorch_models.GANs.PSGAN import PSGAN

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
    result_folder = f"../results/{satellite}"
    model_name = args.name_model
    model_name = f"Psgan"
    model_path1 = f"../pytorch_models/trained_models/{satellite}/PSGAN/{model_name}/model.pth"

    index_test = 2
    test_set_path = dataset_path + satellite + f"/test_{index_test}_512.h5"
    if os.path.exists(test_set_path):
        test_dataloader = DataLoader(DatasetPytorch(test_set_path),
                                     batch_size=64,
                                     shuffle=False)

        model = PSGAN(test_dataloader.dataset.channels)

        # Load Pre trained Model
        trained_model = torch.load(model_path1, map_location=torch.device('cpu'))
        model.generator.load_state_dict(trained_model['gen_state_dict'])

        # Generation Images
        pan, ms, _, gt = next(enumerate(test_dataloader))[1]
        if len(pan.shape) == 3:
            pan = torch.unsqueeze(pan, 0)

        gen = model.generator(ms, pan)
        # From NxCxHxW to NxHxWxC
        gt = torch.permute(gt, (0, 2, 3, 1))
        gen = torch.permute(gen, (0, 2, 3, 1))

        gen = gen.detach().numpy()
        gt = gt.detach().numpy()

        # ######################################## Testing Toolbox Python ###################
        # L = 11;
        # Qblocks_size = 32;
        # flag_cut_bounds = 1;
        # dim_cut = 21;
        # th_values = 0;
        # printEPS = 0;
        # ratio = 4.0;
        # gen = np.squeeze(gen) * 2048
        # gt = np.squeeze(gt) * 2048
        # from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
        # Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut, th_values)
        # print(f"Python:\n\tQ2n={Q2n:.5f}\t Q_avg={Q_avg:.5f}\t ERGAS={ERGAS:.5f}\t SAM={SAM:.5f}")
        #
        # eng = matlab.engine.start_matlab()
        # Q_avg, SAM, ERGAS, SCC, Q = eng.indexes_evaluation(
        #     matlab.double(gen.tolist()),
        #     matlab.double(gt.tolist()),
        #     ratio, L, float(Qblocks_size), flag_cut_bounds, dim_cut, th_values, nargout=5)
        # print(f"MATLAB:\n\tQ2n={Q:.5f}\t Q_avg={Q_avg:.5f}\t ERGAS={ERGAS:.5f}\t SAM={SAM:.5f}")
        ################################################################################

        print(f"Saving {model_name}_test_{index_test}.mat")
        scio.savemat(f"{result_folder}/{model_name}/{model_name}_test_{index_test}.mat", dict(gen_decomposed=gen, gt_decomposed=gt))
