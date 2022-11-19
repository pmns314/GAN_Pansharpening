from abc import ABC

import torch
from torch import nn

from pytorch_models.CNNs.CnnInterface import CnnInterface


def frobenius_loss(y_true, y_pred):
    tensor = y_pred - y_true
    norm = torch.norm(tensor, p="fro")
    return torch.mean(torch.square(norm))


class DiCNN(CnnInterface, ABC):

    def __init__(self, channels, device="cpu", internal_channels=64, name="DiCNN"):
        super(DiCNN, self).__init__(device, name)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=channels+1, out_channels=internal_channels, kernel_size=3, stride=1, padding='same',
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=1,
                      padding='same',
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=internal_channels, out_channels=channels, kernel_size=3, stride=1, padding='same',
                      bias=True),
            nn.ReLU()
        )

    def forward(self, pan, ms):
        inputs = torch.cat([ms, pan], 1)
        output = self.backbone(inputs)
        output = torch.add(output, ms)
        return output

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
            ms_lr = ms_lr.to(self.device)

            # Compute prediction and loss
            pred = self.generate_output(pan, ms=ms, ms_lr=ms_lr)
            loss = self.loss_fn(pred, gt)

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss = loss.item()
            torch.cuda.empty_cache()

            loss_batch += loss
        return loss_batch / len(dataloader)

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
                ms_lr = ms_lr.to(self.device)

                # Compute prediction and loss
                voutputs = self.generate_output(pan, ms=ms, ms_lr=ms_lr)
                vloss = self.loss_fn(voutputs, gt)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)

        return avg_vloss

    def generate_output(self, pan, **kwargs):
        ms = kwargs['ms']
        return self(pan, ms)

    def compile(self, loss_fn=None, optimizer=None):
        self.loss_fn = loss_fn if loss_fn is not None else frobenius_loss
        self.opt = optimizer if optimizer is not None else torch.optim.Adam(self.parameters())
