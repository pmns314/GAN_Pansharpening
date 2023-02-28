""" Loads the model, generates the high resolution image and saves it a .mat file"""
import argparse

import matplotlib.pyplot as plt
from scipy.io import savemat
from skimage import io as io
from torch.utils.data import DataLoader

from constants import *
from dataset.DatasetPytorch import DatasetPytorch
from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
from train_file import create_model
from utils import *


def gen_image(model_type, model_name, satellite, satellite_test, index_test, show_image=False, model_file="model.pth"):
    """ Generate the fused image

    Parameters
    ----------
    model_type : str
        type of the model to use.
        It must match one of the available networks (case-insensitive) otherwise raises KeyError
    model_name : str
        name of the model to use
    satellite : str
        name of the satellite that acquired the testing image
    index_test : int
        index of the testing image data tu use for fusion
    show_image : bool
        if True, the result image is shown
    model_file : str
        name of file storing the weights of the network
    """
    model_path1 = f"{model_path}/{satellite}/{model_type}/{model_name}/{model_file}"
    test_set_path = f"{dataset_path}/FR3/Test/{satellite_test}/test_{index_test}_512.h5"

    if os.path.exists(test_set_path):
        test_dataloader = DataLoader(DatasetPytorch(test_set_path),
                                     batch_size=64,
                                     shuffle=False)

        model = create_model(model_type, test_dataloader.dataset.channels, device=device, evaluation=True)

        # Load Pre trained Model
        model.load_model(model_path1, weights_only=True)
        model.to(device)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        # Generation Images
        pan, ms, ms_lr, gt = next(enumerate(test_dataloader))[1]

        if len(pan.shape) == 3:
            pan = torch.unsqueeze(pan, 0)

        gen = model.generate_output(pan.to(device), evaluation=True,
                                    ms=ms.to(device) if model.use_ms_lr is False else ms_lr.to(device))
        # From NxCxHxW to NxHxWxC
        gen = adjust_image(gen, ms_lr)
        gt = adjust_image(gt)

        Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut,
                                                    th_values)
        res_str = f"Q2n: {Q2n :.4f}  Q avg: {Q_avg:.4f}  ERGAS: {ERGAS:.4f}  SAM: {SAM:.4f}"
        print(res_str)
        # view_image(gt)
        # view_image(gen)
        if show_image is True:
            ms = adjust_image(ms)
            pan = adjust_image(pan)
            fig = view_image(np.concatenate([gt, gen], 1), pan=pan, ms=ms)
            plt.text(0.05, 0.05, res_str, fontsize=14, transform=plt.gcf().transFigure)
            thismanager = plt.get_current_fig_manager()
            thismanager.window.wm_geometry("+500+100")

            plt.show()

        print(f"Saving {model_name}_test_{index_test}.mat")
        if not os.path.exists(f"{result_folder}/{satellite}"):
            os.makedirs(f"{result_folder}/{satellite}")

        if not os.path.exists(f"{result_folder}/{satellite}/{model_type}"):
            os.makedirs(f"{result_folder}/{satellite}/{model_type}")
        if not os.path.exists(f"{result_folder}/{satellite}/gt_{index_test}.tif"):
            io.imsave(f"{result_folder}/{satellite}/gt_{index_test}.tif", gt, check_contrast=False)
            pass

        filename = f"{result_folder}/{satellite}/{model_name}_test_{index_test}.mat"
        if os.path.exists(filename):
            os.remove(filename)

        savemat(filename, dict(gen=gen))


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_model',
                        default='apnn_v2.a',
                        help='Provide name of the model. Defaults to test',
                        type=str
                        )
    parser.add_argument('-t', '--type_model',
                        default='apnn',
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
    parser.add_argument('-st', '--satellite_test',
                        default='W2',
                        help='Provide satellite to use as training. Defaults to W3',
                        type=str
                        )
    parser.add_argument('-mp', '--model_path',
                        default="pytorch_models/trained_models",
                        help='Path of the output folder',
                        type=str
                        )
    parser.add_argument('-i', '--index_test',
                        default=1,
                        help='Index of the testing image',
                        type=int
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
    satellite_test = args.satellite_test
    dataset_path = args.dataset_path
    result_folder = args.output_path
    model_path = args.model_path
    index_test = args.index_test
    # Device Definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    gen_image(model_type, model_name, satellite, satellite_test, index_test, True, "model.pth")
