""" Loads the model, generates the high resolution image and saves it a .mat file"""
import argparse

import numpy as np
from scipy import io as scio
from torch.utils.data import DataLoader

from constants import *
from dataset.DatasetPytorch import DatasetPytorch
from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
from train_gan import create_model
from utils import *

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_model',
                        default='pancolorgan_v2.3',
                        help='Provide name of the model. Defaults to test',
                        type=str
                        )
    parser.add_argument('-t', '--type_model',
                        default='pancolorgan',
                        help='Provide type of the model. Defaults to PSGAN',
                        type=str
                        )
    parser.add_argument('-d', '--dataset_path',
                        default=f'{DATASET_DIR}',
                        help='Provide name of the model. Defaults to ROOT/datasets',
                        type=str
                        )
    parser.add_argument('-s', '--satellite',
                        default='W3',
                        help='Provide satellite to use as training. Defaults to W3',
                        type=str
                        )
    parser.add_argument('-mp', '--model_path',
                        default="pytorch_models/trained_models",
                        help='Path of the output folder',
                        type=str
                        )
    parser.add_argument('-o', '--output_path',
                        default=f"{ROOT_DIR}/results/",
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

    model_path1 = f"{model_path}/{model_type}/{model_name}/model.pth"
    index_test = 3
    test_set_path = f"{dataset_path}/FR/{satellite}/test_{index_test}_512.h5"

    # Device Definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if os.path.exists(test_set_path):
        test_dataloader = DataLoader(DatasetPytorch(test_set_path),
                                     batch_size=64,
                                     shuffle=False)

        model = create_model(model_type, test_dataloader.dataset.channels, device=device, train_spat_disc=None, use_highpass=None)

        # Load Pre trained Model
        trained_model = torch.load(model_path1, map_location=torch.device(device))
        model.generator.load_state_dict(trained_model['gen_state_dict'])
        model.to(device)
        print(f"Best Epoch : {trained_model['best_epoch']}")
        # Generation Images
        pan, ms, ms_lr, gt = next(enumerate(test_dataloader))[1]

        if len(pan.shape) == 3:
            pan = torch.unsqueeze(pan, 0)

        gen = model.generate_output(pan.to(device), ms=ms.to(device), ms_lr=ms_lr.to(device))
        # From NxCxHxW to NxHxWxC
        gen = torch.permute(gen, (0, 2, 3, 1)).detach().cpu().numpy()
        gen = recompose(gen)
        np.clip(gen, 0, 1, out=gen)
        gen = np.squeeze(gen) * 2048.0

        gt = np.squeeze(recompose(torch.squeeze(torch.permute(gt, (0, 2, 3, 1))).detach().numpy())) * 2048.0

        Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut,
                                                    th_values)
        print(f"Q2n: {Q2n :.3f}\t Q_avg: {Q_avg:.3f}\t ERGAS: {ERGAS:.3f}\t SAM: {SAM:.3f}")

        view_image(gt)
        view_image(gen)
        view_image(np.concatenate([gt, gen], 1))
        plt.show()
        print(f"Saving {model_name}_test_{index_test}.mat")
        if not os.path.exists(f"{result_folder}/{model_type}"):
            os.makedirs(f"{result_folder}/{model_type}")
        scio.savemat(f"{result_folder}/{model_type}/{model_name}_test_{index_test}.mat",
                     dict(gen_decomposed=gen, gt_decomposed=gt))