from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
from constants import *
from utils import recompose


class CnnInterface(ABC, nn.Module):
    def __init__(self, device, name):
        super().__init__()
        self._model_name = name
        self.best_loss = np.inf
        self.best_epoch = 0
        self.tot_epochs = 0
        self.device = device
        self.opt = None
        self.loss_fn = None
        self.to(device)

    @property
    def name(self):
        return self._model_name

    # ------------------ Abstract Methods -------------------------
    @abstractmethod
    def generate_output(self, pan, **kwargs):
        pass

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

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss_fn': self.loss_fn,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'tot_epochs': self.tot_epochs
        }, path)

    def load_model(self, path):
        trained_model = torch.load(f"{path}", map_location=torch.device(self.device))
        self.load_state_dict(trained_model['model_state_dict'])
        self.opt.load_state_dict(trained_model['optimizer_state_dict'])
        self.loss_fn = trained_model['loss_fn']
        self.best_loss = trained_model['best_loss']
        self.tot_epochs = trained_model['tot_epochs']
        self.best_epoch = trained_model['best_epoch']

    def train_model(self, epochs,
                    output_path, chk_path,
                    train_dataloader, val_dataloader,
                    tests=None,
                    patience=30):

        # TensorBoard
        writer = SummaryWriter(output_path + "/log")
        # Early stopping
        triggertimes = 0

        # Reduce Learning Rate on Plateaux
        # scheduler_d = ReduceLROnPlateau(self.disc_opt, 'min', patience=10, verbose=True)
        # scheduler_g = ReduceLROnPlateau(self.gen_opt, 'min', patience=10, verbose=True)

        # Training
        print(f"Training started for {output_path} at epoch {self.tot_epochs + 1}")
        ending_epoch = self.tot_epochs + epochs
        for epoch in range(epochs):
            self.tot_epochs += 1
            print(f'\nEpoch {self.tot_epochs}/{ending_epoch}')

            # Compute Losses on Train Set
            train_loss = self.train_step(train_dataloader)
            if val_dataloader is not None:
                # Compute Losses on Validation Set if exsists
                curr_loss = self.validation_step(val_dataloader)

                print(f'\t Loss: train {train_loss :.3f}\t valid {curr_loss:.3f}\n')
                writer.add_scalars("Loss", {"train": train_loss, "validation": curr_loss},
                                   self.tot_epochs)
            else:
                # Otherwise keeps track only of train losses
                print(f'\t Loss: {train_loss :.3f}', end="\t")
                writer.add_scalar(f"Loss/Train", train_loss, self.tot_epochs)
                curr_loss = train_loss

            # Updates best losses
            # Saves the model if the loss of the generator ( position 0 ) improved
            if curr_loss < self.best_loss:
                self.best_loss = curr_loss
                self.best_epoch = self.tot_epochs
                self.save_model(f"{output_path}/model.pth")
                triggertimes = 0
            else:
                triggertimes += 1

            # Save Checkpoints
            if self.tot_epochs in TO_SAVE or epoch == epochs - 1:
                self.save_model(f"{chk_path}/checkpoint_{self.tot_epochs}.pth")

            # Generation Indexes
            for t in tests:
                df = pd.DataFrame(columns=["Epochs", "Q2n", "Q_avg", "SAM", "ERGAS"])

                gen = self.generate_output(pan=t['pan'].to(self.device),
                                           ms=t['ms'].to(self.device),
                                           ms_lr=t['ms_lr'].to(self.device))
                # gen = self.generator(t['ms'].to(device), t['pan'].to(device))
                gen = torch.permute(gen, (0, 2, 3, 1)).detach().to('cpu').numpy()
                gen = recompose(gen)
                gen = np.squeeze(gen) * 2048
                gt = np.squeeze(t['gt']) * 2048

                Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut,
                                                            th_values)
                df.loc[0] = [self.tot_epochs + epoch, Q2n, Q_avg, ERGAS, SAM]
                df.to_csv(t['filename'], index=False, header=True if self.tot_epochs == 1 else False,
                          mode='a', sep=";")

            # if triggertimes >= patience:
            #     print("Early Stopping!")
            #     break
            # scheduler_d.step(best_vloss_d)
            # scheduler_g.step(best_vloss_g)

        # Update number of trained epochs
        last_tot = self.tot_epochs
        self.load_model(f"{output_path}/model.pth")
        self.tot_epochs = last_tot
        self.save_model(f"{output_path}/model.pth")

        writer.flush()
        print(f"Training Completed at epoch {self.tot_epochs}.\n"
              f"Best Epoch:{self.best_epoch} Saved in {output_path} folder")

    def test_loop(self, dataloader):
        test_loss = self.validation_step(dataloader)
        print(f"Evaluation on Test Set: \n "
              f"\t Loss: {test_loss:>8f} \n")

    def set_optimizer_lr(self, lr):
        for g in self.opt.param_groups:
            g['lr'] = lr

