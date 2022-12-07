from abc import abstractmethod

import numpy as np
import torch
from pytorch_models.NetworkInterface import NetworkInterface


class CnnInterface(NetworkInterface):
    def __init__(self, device, name):
        super().__init__(device, name)
        self.best_losses = [np.inf]
        self.opt = None
        self.loss_fn = None

    # ------------------ Abstract Methods -------------------------
    @abstractmethod
    def generate_output(self, pan, evaluation=True, **kwargs):
        pass

    @abstractmethod
    def compile(self, loss_fn=None, optimizer=None):
        pass

    # ------------------------- Concrete Methods ------------------------------

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
            pred = self.generate_output(pan, ms=ms, ms_lr=ms_lr, evaluation=False)
            loss = self.loss_fn(pred, gt)

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss = loss.item()
            torch.cuda.empty_cache()

            loss_batch += loss
        return {"Loss": loss_batch / len(dataloader)}

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

        return {"Loss": avg_vloss}

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss_fn': self.loss_fn,
            'best_losses': self.best_losses,
            'best_epoch': self.best_epoch,
            'tot_epochs': self.tot_epochs
        }, path)

    def load_model(self, path, weights_only=False):
        trained_model = torch.load(f"{path}", map_location=torch.device(self.device))
        self.load_state_dict(trained_model['model_state_dict'])
        if weights_only:
            return
        self.opt.load_state_dict(trained_model['optimizer_state_dict'])
        self.loss_fn = trained_model['loss_fn']
        self.tot_epochs = trained_model['tot_epochs']
        self.best_epoch = trained_model['best_epoch']
        try:
            self.best_losses = trained_model['best_losses']
        except KeyError:
            self.best_losses = [trained_model['best_loss']]

    def set_optimizers_lr(self, lr):
        for g in self.opt.param_groups:
            g['lr'] = lr
