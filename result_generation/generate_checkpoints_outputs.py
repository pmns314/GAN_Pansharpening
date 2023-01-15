""" Loads the model, generates the high resolution image and saves it a .mat file"""
import argparse
import os

import numpy as np
from scipy.io import savemat

from skimage import io as io

from torch.utils.data import DataLoader

from constants import *
from dataset.DatasetPytorch import DatasetPytorch
from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
from train_file import create_model
from utils import *


def gen_image(show_image=False, model_file="model.pth"):
    model_path_file = f"../{folder_path}/checkpoints/{model_file}"

    # Load Pre trained Model
    model.load_model(model_path_file, weights_only=True)
    model.to(device)
    # Generation Images
    pan, ms, ms_lr, gt = next(enumerate(test_dataloader))[1]

    if len(pan.shape) == 3:
        pan = torch.unsqueeze(pan, 0)

    gen = model.generate_output(pan.to(device), evaluation=True, ms=ms.to(device))
    # From NxCxHxW to NxHxWxC
    gen = adjust_image(gen, ms_lr)
    gt = adjust_image(gt)

    Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut,
                                                th_values)
    print(f"Q2n: {Q2n :.4f}\t Q_avg: {Q_avg:.4f}\t ERGAS: {ERGAS:.4f}\t SAM: {SAM:.4f}")

    if show_image is True:
        view_image(np.concatenate([gt, gen], 1))
        plt.show()

    if not os.path.exists(f"{folder_path}/checkpoints_output"):
        os.makedirs(f"{folder_path}/checkpoints_output")

    number = model_file.split("_")[1][:-4]
    filename = f"checkpoint_{number}.mat"
    if os.path.exists(filename):
        os.remove(filename)
    # import imageio
    # imageio.v3.imwrite(filename, gen)
    # gen = np.transpose(gen, (2, 0, 1))
    if data_out_format == "mat":
        savemat(f"../{folder_path}/checkpoints_output/{filename}", dict(gen=gen))
    else:
        io.imsave(filename, gen, check_contrast=False)


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_model',
                        default='test',
                        help='Provide name of the model. Defaults to test',
                        type=str
                        )
    parser.add_argument('-t', '--type_model',
                        default='psgan',
                        help='Provide type of the model. Defaults to PSGAN',
                        type=str
                        )
    parser.add_argument('-d', '--dataset_path',
                        default=f'{DATASET_DIR}',
                        help='Provide name of the model. Defaults to ROOT/datasets',
                        type=str
                        )
    parser.add_argument('-s', '--satellite',
                        default='W2',
                        help='Provide satellite to use as training. Defaults to W3',
                        type=str
                        )
    parser.add_argument('-mp', '--model_path',
                        default="pytorch_models/trained_models",
                        help='Path of the output folder',
                        type=str
                        )
    parser.add_argument('-o', '--output_path',
                        default=f"{ROOT_DIR}/results/GANs",
                        help='Path of the output folder',
                        type=str
                        )
    parser.add_argument('-f', '--force',
                        action='store_true',
                        help='Boolean indicating if forcing GPU Max Memory allowed'
                        )

    args = parser.parse_args()

    model_name = args.name_model
    model_type = args.type_model
    satellite = args.satellite
    dataset_path = args.dataset_path
    result_folder = args.output_path
    model_path = args.model_path
    # Device Definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    data_out_format = "mat"

    index_test = 2
    satellite = "W2"
    model_type = "PanGan"
    model_name = "pangan_v2.6"

    test_set_path = f"{dataset_path}/FR3/Test/{satellite}/test_{index_test}_512.h5"
    test_dataloader = DataLoader(DatasetPytorch(test_set_path),
                                 batch_size=64,
                                 shuffle=False)
    folder_path = f"{model_path}/{satellite}/{model_type}/{model_name}"
    model = create_model(model_type, test_dataloader.dataset.channels, device=device)
    for file in os.listdir(f"../{folder_path}/checkpoints"):
        gen_image(False, file)

