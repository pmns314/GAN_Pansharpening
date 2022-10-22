import os
import shutil
from abc import ABC

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.DatasetPytorch import DatasetPytorch
from constants import ROOT_DIR
from pytorch_models.CNNs.CnnInterface import CnnInterface


class APNN(CnnInterface, ABC):
    def __init__(self, channels, device="cpu", name="APNN"):
        super(APNN, self).__init__(device, name)
        self._model_name = name
        self.channels = channels
        self.conv1 = nn.Conv2d(in_channels=channels + 1, out_channels=64, kernel_size=(9, 9), padding='same',
                               padding_mode='replicate', bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), padding='same',
                               padding_mode='replicate', bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=(5, 5), padding='same',
                               padding_mode='replicate', bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ms, pan):
        inputs = torch.cat([ms, pan], 1)
        rs = self.conv1(inputs)
        rs = self.relu(rs)
        rs = self.conv2(rs)
        rs = self.relu(rs)
        out = self.conv3(rs)

        # Skip connection converts the model in a residual model
        out = ms + out
        return out

    def train_step(self, dataloader):
        self.train(True)

        loss_batch = 0
        for batch, data in enumerate(dataloader):
            pan, ms, ms_lr, gt = data

            if len(pan.shape) == 3:
                pan = torch.unsqueeze(pan, 0)
            gt = gt.to(self.device)
            pan = pan.to(self.device)
            ms = ms.to(self.device)

            # Compute prediction and loss
            pred = self(ms, pan)
            loss = self.loss_fn(pred, gt)
            loss /= (32 * 32)

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss = loss.item()
            torch.cuda.empty_cache()

            loss_batch += loss

        return loss_batch / (len(dataloader))

    def validation_step(self, dataloader):
        self.train(False)
        self.eval()
        running_vloss = 0.0
        i = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                pan, ms, ms_lr, gt = data
                if len(pan.shape) == 3:
                    pan = torch.unsqueeze(pan, 0)
                gt = gt.to(self.device)
                pan = pan.to(self.device)
                ms = ms.to(self.device)

                voutputs = self(ms, pan)
                vloss = self.loss_fn(voutputs, gt)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)

        return avg_vloss

    def generate_output(self, pan, **kwargs):
        return self(kwargs['ms'], pan)




if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    file_name = "ppp"
    satellite = "W3"
    output_path = os.path.join(ROOT_DIR, 'pytorch_models', 'trained_models', file_name)
    dataset_path = os.path.join(ROOT_DIR, 'datasets/')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    train_data = DatasetPytorch(dataset_path + satellite + "/train.h5")
    train_dataloader = DataLoader(train_data, batch_size=64,
                                  shuffle=True)
    val_dataloader = DataLoader(DatasetPytorch(dataset_path + satellite + "/val.h5"), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(DatasetPytorch(dataset_path + satellite + "/test.h5"), batch_size=64,
                                 shuffle=True)

    model = APNN(train_data.channels)
    model.to(device)

    # Load Pre trained Model
    model.load_state_dict(
        torch.load(ROOT_DIR + "/pytorch_models/trained_models/pnn.pth", map_location=torch.device(device)))

    loss_fn = torch.nn.MSELoss(reduction='mean').to(device)
    lr = 0.0001 * 17 * 17 * train_data.channels
    optimizer = optim.Adam(model.parameters(), lr=.001)

    # TensorBoard
    writer = SummaryWriter(output_path + "/log/")
    # Early stopping
    patience = 30
    triggertimes = 0
    # Reduce Learning Rate on Plateaux
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

    best_vloss = +np.inf
    for epoch in range(1):
        train_loss = model.train_loop(train_dataloader, loss_fn, optimizer, device)
        if val_dataloader is not None:
            curr_loss = model.validation_loop(val_dataloader, loss_fn, device)
            print(f'Epoch {epoch + 1}\t'
                  f'\t train {train_loss :.2f}\t valid {curr_loss:.2f}')
            writer.add_scalars("Loss", {"train": train_loss, "validation": curr_loss}, epoch)
        else:
            print(f'Epoch {epoch + 1}\t'
                  f'\t train {train_loss :.2f}')
            curr_loss = train_loss
            writer.add_scalar("Loss/validation", train_loss, epoch)

        if curr_loss < best_vloss:
            best_vloss = curr_loss
            torch.save(model.state_dict(), output_path + "model.pth")
            triggertimes = 0
        else:
            triggertimes += 1
            if triggertimes >= patience:
                print("Early Stopping!")
                break
        scheduler.step(best_vloss)

    writer.flush()
    model.test_loop(test_dataloader, loss_fn)
