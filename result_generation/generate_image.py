""" Loads the model, generates the high resolution image and saves it a .mat file"""
import argparse
import os
import shutil

import torch
from scipy import io as scio
from torch.utils.data import DataLoader

import constants
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.CNNs.APNN import APNN

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
    result_folder = f"./results/{satellite}_full3"
    model_name = args.name_model
    model_name = f"W3_1_32_0001_mse"
    model_path1 = f"./pytorch_models/trained_models/{satellite}_all/{model_name}/model.pth"

    index_test = 3
    test_set_path = dataset_path + satellite + f"/test_{index_test}_512.h5"
    if os.path.exists(test_set_path):
        test_dataloader = DataLoader(DatasetPytorch(test_set_path),
                                     batch_size=64,
                                     shuffle=False)

        model = APNN(test_dataloader.dataset.channels)

        # Load Pre trained Model
        trained_model = torch.load(model_path1, map_location=torch.device('cpu'))
        model.load_state_dict(trained_model['model_state_dict'])


        # Generation Images
        pan, ms, _, gt = next(enumerate(test_dataloader))[1]
        if len(pan.shape) == 3:
            pan = torch.unsqueeze(pan, 0)

        gen = model(ms, pan)
        # From NxCxHxW to NxHxWxC
        gt = torch.permute(gt, (0, 2, 3, 1))
        gen = torch.permute(gen, (0, 2, 3, 1))

        gen = gen.detach().numpy()
        gt = gt.detach().numpy()
        print(f"Saving {model_name}_test_{index_test}.mat")
        scio.savemat(f"{result_folder}/{model_name}/{model_name}_test_{index_test}.mat", dict(gen_decomposed=gen, gt_decomposed=gt))

