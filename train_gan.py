import argparse
import os
import shutil

import numpy as np
import torch
from git import Repo
from torch.utils.data import DataLoader

from constants import *
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.GANs import *

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
                        help='Boolean indicating if resuming the training or starting a new one deleting the one '
                             'already existing, if any',
                        type=bool
                        )
    parser.add_argument('-o', '--output_path',
                        default="pytorch_models/trained_models",
                        help='Path of the output folder',
                        type=str
                        )
    parser.add_argument('-c', '--commit',
                        default=True,
                        help='Boolean indicating if commit is to git is needed',
                        type=bool
                        )
    args = parser.parse_args()

    repo = Repo(ROOT_DIR + "/.git")

    file_name = args.name_model
    satellite = args.satellite
    dataset_path = args.dataset_path
    epochs = args.epochs
    lr = args.learning_rate
    resume_flag = args.resume
    output_base_path = args.output_path
    flag_commit = args.commit

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
    model = PanGan(train_dataloader.dataset.channels, device)
    model.to(device)
    # Model Loading if resuming training
    output_path = os.path.join(output_base_path, model.name, file_name)
    if resume_flag and os.path.exists(f"{output_path}/model.pth"):
        model.load_model(f"{output_path}", lr)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

    model.set_optimizers_lr(lr)

    # Checkpoint path definition
    chk_path = f"{output_path}\\checkpoints"
    if not os.path.exists(chk_path):
        os.makedirs(chk_path)

    # Setting up index evaluation
    test_1 = {}
    pan, ms, ms_lr, gt = next(enumerate(test_dataloader1))[1]
    if len(pan.shape) == 3:
        pan = torch.unsqueeze(pan, 0)
    gt = torch.permute(gt, (0, 2, 3, 1))
    test_1['pan'] = pan
    test_1['ms'] = ms
    test_1['ms_lr'] = ms_lr
    test_1['gt'] = torch.squeeze(gt).detach().numpy()
    test_1['filename'] = f"{output_path}/test_0.csv"

    test_2 = {}
    pan, ms, ms_lr, gt = next(enumerate(test_dataloader2))[1]
    if len(pan.shape) == 3:
        pan = torch.unsqueeze(pan, 0)
    gt = torch.permute(gt, (0, 2, 3, 1))
    test_2['pan'] = pan
    test_2['ms'] = ms
    test_2['ms_lr'] = ms_lr
    test_2['gt'] = torch.squeeze(gt).detach().numpy()
    test_2['filename'] = f"{output_path}/test_1.csv"

    # Model Training
    model.train_model(epochs,
                      output_path, chk_path,
                      train_dataloader, val_dataloader,
                      [test_1, test_2])

    # # Commit and Push new model
    # if flag_commit:
    #     origin = repo.remote(name='origin')
    #     origin.pull()
    #     repo.git.add(output_path)
    #     repo.index.commit(f"model {file_name} - {model.name} trained")
    #     origin.push()
