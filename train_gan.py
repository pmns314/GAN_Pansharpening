import argparse
import os
import shutil

import numpy as np
import torch
from git import Repo
from torch.utils.data import DataLoader

from constants import *
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.GANs.PSGAN import PSGAN

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_model',
                        default='test',
                        help='Provide name of the model. Defaults to test',
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
    parser.add_argument('-e', '--epochs',
                        default=10000,
                        help='Provide number of epochs. Defaults to 1000',
                        type=int
                        )
    parser.add_argument('-lr', '--learning_rate',
                        default=0.01,
                        help='Provide learning rate. Defaults to 0.001',
                        type=float
                        )
    parser.add_argument('-r', '--resume',
                        default=None,
                        help='Provide path to the partially trained model. Defaults to None',
                        type=str
                        )
    parser.add_argument('-ck', '--checkpoints',
                        default=None,
                        help='Path to the checkpoints',
                        type=str
                        )
    args = parser.parse_args()

    repo = Repo(ROOT_DIR + "/.git")

    file_name = args.name_model
    satellite = args.satellite
    dataset_path = args.dataset_path
    epochs = args.epochs
    lr = args.learning_rate
    pretrained_model_path = args.resume
    chk_path = args.checkpoints

    train_dataset = f"train_1_32.h5"
    val_dataset = f"val_1_32.h5"
    test_dataset1 = f"test_1_256.h5"
    test_dataset2 = f"test_3_512.h5"

    # Device Definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Data Loading
    train_dataloader = DataLoader(DatasetPytorch(f"{dataset_path}/{satellite}/{train_dataset}"), batch_size=64,
                                  shuffle=True)
    val_dataloader = DataLoader(DatasetPytorch(f"{dataset_path}/{satellite}/{val_dataset}"), batch_size=64,
                                shuffle=True)

    test_dataloader1 = DataLoader(DatasetPytorch(f"{dataset_path}/{satellite}/{test_dataset1}"), batch_size=64,
                                  shuffle=False)
    test_dataloader2 = DataLoader(DatasetPytorch(f"{dataset_path}/{satellite}/{test_dataset2}"), batch_size=64,
                                  shuffle=False)
    # Model Creation
    model = PSGAN(train_dataloader.dataset.channels)
    model.set_optimizers(lr)
    model.to(device)

    output_path = os.path.join(ROOT_DIR, 'pytorch_models', 'trained_models', satellite, model.name, file_name)
    if pretrained_model_path is not None:
        trained_model = torch.load(f"{pretrained_model_path}/model.pth", map_location=torch.device(device))
        model.generator.load_state_dict(trained_model['gen_state_dict'])
        model.discriminator.load_state_dict(trained_model['disc_state_dict'])
        model.gen_opt.load_state_dict(trained_model['gen_optimizer_state_dict'])
        model.disc_opt.load_state_dict(trained_model['disc_optimizer_state_dict'])
        trained_epochs = trained_model['tot_epochs']
        best_vloss_g = trained_model['gen_best_loss']
        best_vloss_d = trained_model['disc_best_loss']
    else:
        trained_epochs = 0
        best_vloss_g = +np.inf
        best_vloss_d = +np.inf
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

    if chk_path is None:
        chk_path = f"{output_path}\\checkpoints"
        if not os.path.exists(chk_path):
            os.makedirs(chk_path)

    # Setting up index evaluation
    test_1 = {}
    pan, ms, _, gt = next(enumerate(test_dataloader1))[1]
    if len(pan.shape) == 3:
        pan = torch.unsqueeze(pan, 0)
    gt = torch.permute(gt, (0, 2, 3, 1))
    test_1['pan'] = pan
    test_1['ms'] = ms
    test_1['gt'] = torch.squeeze(gt).detach().numpy()
    test_1['filename'] = f"{output_path}/test_0.csv"

    test_2 = {}
    pan, ms, _, gt = next(enumerate(test_dataloader2))[1]
    if len(pan.shape) == 3:
        pan = torch.unsqueeze(pan, 0)
    gt = torch.permute(gt, (0, 2, 3, 1))
    test_2['pan'] = pan
    test_2['ms'] = ms
    test_2['gt'] = torch.squeeze(gt).detach().numpy()
    test_2['filename'] = f"{output_path}/test_1.csv"

    # Model Training
    model.my_training(epochs, best_vloss_d, best_vloss_g,
                      output_path, chk_path,
                      train_dataloader, val_dataloader,
                      [test_1, test_2],
                      pretrained_epochs=trained_epochs, device=device)

    # Commit and Push new model
    origin = repo.remote(name='origin')
    origin.pull()
    repo.git.add(output_path)
    repo.index.commit(f"model {file_name} - {model.name} trained")
    origin.push()
