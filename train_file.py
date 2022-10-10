import argparse
import os
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import constants
from constants import ROOT_DIR
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.CNNs.APNN import APNN
from git import Repo


def train_model(model, epochs,
                loss_fn, optimizer, best_vloss,
                output_path,chk_path,
                train_dataloader, val_dataloader, test_dataloader=None,
                pretrained_epochs=0, device='cpu'):
    # TensorBoard
    writer = SummaryWriter(output_path + "/log/")

    # Early stopping
    patience = 250
    triggertimes = 0

    pretrained_epochs = pretrained_epochs + 1
    epoch = 0

    print(f"Training started for {output_path} at epoch {pretrained_epochs}")
    for epoch in range(epochs):
        train_loss = model.train_loop(train_dataloader, loss_fn, optimizer, device)
        if val_dataloader is not None:
            curr_loss = model.validation_loop(val_dataloader, loss_fn, device)
            print(f'Epoch {pretrained_epochs + epoch}\t'
                  f'\t train {train_loss :.2f}\t valid {curr_loss:.2f}')
            writer.add_scalars("Loss", {"train": train_loss, "validation": curr_loss}, pretrained_epochs + epoch)
        else:
            print(f'Epoch {pretrained_epochs + epoch}\t'
                  f'\t train {train_loss :.2f}')
            curr_loss = train_loss
            writer.add_scalar("Loss/train", train_loss, pretrained_epochs + epoch)

        # Test loss
        if test_dataloader is not None:
            test_loss = model.validation_loop(test_dataloader, loss_fn, device)
            writer.add_scalar("Loss/test", test_loss, pretrained_epochs + epoch)

        # Save Checkpoint
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_vloss
        }, f"{chk_path}/checkpoint_{pretrained_epochs + epoch}.pth")

        # Save Best Model
        if curr_loss < best_vloss:
            best_vloss = curr_loss

            torch.save({
                'best_epoch': pretrained_epochs + epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn': loss_fn,
                'best_loss': best_vloss
            }, output_path + "/model.pth")

            triggertimes = 0
        else:
            triggertimes += 1

            # Early Stopping
            if triggertimes >= patience:
                print("Early Stopping!")
                break

    m = torch.load(output_path + "/model.pth")
    m['tot_epochs'] = pretrained_epochs + epoch
    torch.save(m, output_path + "/model.pth")
    writer.flush()
    print(f"Training Completed at epoch {pretrained_epochs + epoch}. Saved in {output_path} folder")


if __name__ == '__main__':

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_model',
                        default='test',
                        help='Provide name of the model. Defaults to test',
                        type=str
                        )
    parser.add_argument('-d', '--dataset_path',
                        default=f'{constants.DATASET_DIR}',
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

    # Listing Patch Sizes Training
    patch_sizes = []
    file_num = -1
    for file in os.listdir(f"{dataset_path}/{satellite}"):
        if file.startswith("train"):
            if file.endswith(".h5"):
                patch_sizes.append(file[8:-3])
                file_num = file[6]

    # -------- Override of listing -------- #
    patch_sizes = [32]
    # ------------------------------------- #

    for patch_size in patch_sizes:
        train_dataset = f"train_{file_num}_{patch_size}.h5"
        val_dataset = f"val_{file_num}_{patch_size}.h5"
        test_dataset = f"test_1_256.h5"

        file_name = f"{satellite}_{file_num}_{patch_size}_{str(lr)[2:]}_mae"
        output_path = os.path.join(ROOT_DIR, 'pytorch_models', 'trained_models', 'W3_all', file_name)

        # Device Definition
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")

        # Data Loading
        train_dataloader = DataLoader(DatasetPytorch(f"{dataset_path}/{satellite}/{train_dataset}"), batch_size=64,
                                      shuffle=True)
        val_dataloader = DataLoader(DatasetPytorch(f"{dataset_path}/{satellite}/{val_dataset}"), batch_size=64,
                                    shuffle=True)
        test_dataloader = DataLoader(DatasetPytorch(f"{dataset_path}/{satellite}/{test_dataset}"), batch_size=64,
                                     shuffle=False)
        # Model Creation
        model = APNN(train_dataloader.dataset.channels)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        if pretrained_model_path is not None:
            trained_model = torch.load(f"{pretrained_model_path}/model.pth", map_location=torch.device(device))
            model.load_state_dict(trained_model['model_state_dict'])
            optimizer.load_state_dict(trained_model['optimizer_state_dict'])
            trained_epochs = trained_model['tot_epochs']
            loss_fn = trained_model['loss_fn']
            best_vloss = trained_model['best_loss']
        else:
            loss_fn = torch.nn.L1Loss(reduction='mean').to(device)
            trained_epochs = 0
            best_vloss = +np.inf
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

        # Model Training
        train_model(model, epochs,
                    loss_fn, optimizer, best_vloss,
                    output_path,chk_path,
                    train_dataloader, val_dataloader, test_dataloader,
                    pretrained_epochs=trained_epochs, device=device)

        # Commit and Push new model
        origin = repo.remote(name='origin')
        origin.pull()
        repo.git.add(output_path)
        repo.index.commit(f"model {file_name} trained")
        origin.push()
