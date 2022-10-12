from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
from constants import *


class GanInterface(ABC, nn.Module):
    def __init__(self, device):
        super().__init__()
        # self.best_losses: list = NotImplemented  # Must be defined by subclasses
        self.pretrained_epochs = 0
        self.best_epoch = 0
        self.triggertimes = 0
        self.device = device
        self.to(device)

    @property
    @abstractmethod
    def best_losses(self) -> list:
        ...

    @best_losses.setter
    def best_losses(self, losses):
        self.best_losses = losses

    # ------------------ Abstract Methods -------------------------
    @abstractmethod
    def train_step(self, dataloader):
        pass

    @abstractmethod
    def validation_step(self, dataloader):
        pass

    @abstractmethod
    def save_checkpoint(self, path, curr_epoch):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def generate_output(self, ms, pan):
        pass

    # ------------------------- Concrete Methods ------------------------------
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

        self.pretrained_epochs = self.pretrained_epochs + 1
        epoch = 0

        # Training
        print(f"Training started for {output_path} at epoch {self.pretrained_epochs}")
        for epoch in range(epochs):
            print(f'\nEpoch {self.pretrained_epochs + epoch}')

            # Compute Losses on Train Set
            train_losses = self.train_step(train_dataloader)
            if val_dataloader is not None:
                # Compute Losses on Validation Set if exsists
                val_losses = self.validation_step(val_dataloader)
                for k in train_losses.keys():
                    print(f'\t {k}: train {train_losses[k] :.3f}\t valid {val_losses[k]:.3f}\n')
                    writer.add_scalars(k, {"train": train_losses[k], "validation": val_losses[k]},
                                       self.pretrained_epochs + epoch)
                losses = list(val_losses.values())
            else:
                # Otherwise keeps track only of train losses
                for item in train_losses.items():
                    key, value = item
                    print(f'\t {key}: {value :.3f}', end="\t")
                    writer.add_scalar(f"{key}/Train", value, self.pretrained_epochs + epoch)
                losses = list(train_losses.values())

            # Save Checkpoints
            if self.pretrained_epochs + epoch in TO_SAVE:
                self.save_checkpoint(chk_path, self.pretrained_epochs + epoch)

            # Updates best losses
            # Saves the model if the loss of the generator ( position 0 ) improved
            if losses[0] < self.best_losses[0]:
                self.best_losses[0] = losses[0]
                self.best_epoch = self.pretrained_epochs + epoch
                self.save_model(output_path)
                self.triggertimes = 0
            else:
                self.triggertimes += 1

            for i in range(1, len(losses)):
                if losses[i] < self.best_losses[i]:
                    self.best_losses[i] = losses[i]

            # Generation Indexes
            for t in tests:
                df = pd.DataFrame(columns=["Epochs", "Q2n", "Q_avg", "SAM", "ERGAS"])

                gen = self.generate_output(t['ms'].to(self.device), t['pan'].to(self.device))
                # gen = self.generator(t['ms'].to(device), t['pan'].to(device))
                gen = torch.permute(gen, (0, 2, 3, 1)).detach().to('cpu').numpy()
                gen = np.squeeze(gen) * 2048
                gt = np.squeeze(t['gt']) * 2048

                Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut,
                                                            th_values)
                df.loc[0] = [self.pretrained_epochs + epoch, Q2n, Q_avg, ERGAS, SAM]
                df.to_csv(t['filename'], index=False, header=True if self.pretrained_epochs + epoch == 1 else False,
                          mode='a', sep=";")

            if triggertimes >= patience:
                print("Early Stopping!")
                break
            # scheduler_d.step(best_vloss_d)
            # scheduler_g.step(best_vloss_g)

        # Update number of trained epochs
        self.load(output_path + "/model.pth")
        self.pretrained_epochs = self.pretrained_epochs + epoch
        self.save_model(output_path + "/model.pth")
        writer.flush()
        print(f"Training Completed at epoch {self.pretrained_epochs + epoch}. Saved in {output_path} folder")

    def test_loop(self, dataloader):
        disc_loss_avg, gen_loss_avg = self.validation_step(dataloader)
        print(f"Evaluation on Test Set: \n "
              f"\t Avg disc loss: {disc_loss_avg:>8f} \n"
              f"\t Avg gen loss: {gen_loss_avg:>8f} \n")
