""" Generate images for all the given lr, tests and training combinations"""

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

    result_folder = f"./results/{satellite}_full"
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.makedirs(result_folder)

    model_name = args.name_model

    patch_sizes = []
    file_num = -1
    for file in os.listdir(dataset_path + satellite):
        if file.startswith("train"):
            if file.endswith(".h5"):
                patch_sizes.append(file[8:-3])
                file_num = file[6]

    patch_sizes = [32]
    for patch_size in patch_sizes:
        for lr in ['01', '001', '0001', '-05']:
            model_name_base = f"{satellite}_{file_num}_{patch_size}_{lr}"
            model_path1 = f"./pytorch_models/trained_models/{satellite}_full/{model_name_base}/model.pth"
            #####################################
            if os.path.exists(model_path1):
            ####################################
                for test_index in [1, 2, 3]:
                    for patch_size_test in [64, 128, 256, 512]:
                        test_set_path = dataset_path + satellite + f"/test_{test_index}_{patch_size_test}.h5"
                        if os.path.exists(test_set_path):
                            test_dataloader = DataLoader(DatasetPytorch(test_set_path),
                                                         batch_size=64,
                                                         shuffle=False)

                            model = APNN(test_dataloader.dataset.channels)

                            # Load Pre trained Model
                            trained_model = torch.load(model_path1, map_location=torch.device('cpu'))
                            model.load_state_dict(trained_model['model_state_dict'])
                            #optimizer.load_state_dict(trained_model['optimizer_state_dict'])
                            # trained_epochs = trained_model['epoch']
                            loss_fn = trained_model['loss_fn']


                            # Testing
                            # model.test_loop(test_dataloader, loss_fn)

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
                            print(f"Saving {model_name_base}_{test_index}_{str(patch_size_test)}.mat")
                            scio.savemat(f"{result_folder}/{model_name_base}_{test_index}_{str(patch_size_test)}.mat",
                                         dict(gen_decomposed=gen, gt_decomposed=gt))

    # out_gt = np.zeros((256, 256, 8))
    # out_gen = np.zeros((256, 256, 8))
    # n_patch = gen.shape[0]
    # dim_patch = gen.shape[1]
    # num_patch_per_row = 256 // dim_patch
    # row = -1
    # col = 0
    #
    # start_row = stop_row = 0
    # start_col = stop_col = 0
    # for i in range(n_patch):
    #     if i % num_patch_per_row == 0:
    #         start_col = 0
    #         start_row = stop_row
    #         stop_row = stop_row + dim_patch
    #     else:
    #         start_col = stop_col
    #     stop_col = start_col + dim_patch
    #     out_gt[start_row:stop_row, start_col:stop_col, :] = gt[i, :, :, :]
    #     out_gen[start_row:stop_row, start_col:stop_col, :] = gen[i, :, :, :]
    #
    #
    # plt.imshow(out_gt[:, :, :3])
    # plt.figure()
    # plt.imshow(out_gen[:, :, :3])
    # plt.show()

    #
