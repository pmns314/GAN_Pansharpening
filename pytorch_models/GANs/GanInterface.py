from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
from constants import *
from utils import recompose
import matplotlib.pyplot as plt


class GanInterface(ABC, nn.Module):
    def __init__(self, device, name):
        super().__init__()
        self._model_name = name
        self.best_losses: list = NotImplemented  # Must be defined by subclasses
        self.best_epoch = 0
        self.tot_epochs = 0
        self.device = device
        self.to(device)

    @property
    def name(self):
        return self._model_name

    # ------------------ Abstract Methods -------------------------
    @abstractmethod
    def train_step(self, dataloader):
        pass

    @abstractmethod
    def validation_step(self, dataloader):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def generate_output(self, pan, **kwargs):
        pass

    @abstractmethod
    def set_optimizers_lr(self, lr):
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

        # Training
        print(f"Training started for {output_path} at epoch {self.tot_epochs + 1}")
        ending_epoch = self.tot_epochs + epochs
        for epoch in range(epochs):
            self.tot_epochs += 1
            print(f'\nEpoch {self.tot_epochs}/{ending_epoch}')

            # Compute Losses on Train Set
            train_losses = self.train_step(train_dataloader)
            if val_dataloader is not None:
                # Compute Losses on Validation Set if exsists
                val_losses = self.validation_step(val_dataloader)
                for k in train_losses.keys():
                    print(f'\t {k}: train {train_losses[k] :.3f}\t valid {val_losses[k]:.3f}\n')
                    writer.add_scalars(k, {"train": train_losses[k], "validation": val_losses[k]},
                                       self.tot_epochs)
                losses = list(val_losses.values())
            else:
                # Otherwise keeps track only of train losses
                for item in train_losses.items():
                    key, value = item
                    print(f'\t {key}: {value :.3f}', end="\t")
                    writer.add_scalar(f"{key}/Train", value, self.tot_epochs)
                losses = list(train_losses.values())

            # Updates the best losses
            # Saves the model if the loss of the generator ( position 0 ) improved
            if losses[0] < self.best_losses[0]:
                self.best_losses[0] = losses[0]
                self.best_epoch = self.tot_epochs
                self.save_model(f"{output_path}/model.pth")
                print(f"New Best Loss {self.best_losses[0]:.3f} at epoch {self.best_epoch}")
                triggertimes = 0
            else:
                triggertimes += 1

            for i in range(1, len(losses)):
                if losses[i] < self.best_losses[i]:
                    self.best_losses[i] = losses[i]

            if self.tot_epochs in TO_SAVE or epoch == epochs-1:
                # Save Checkpoints
                self.save_model(f"{chk_path}/checkpoint_{self.tot_epochs}.pth")

                # Generation Indexes
                for idx_test in range(len(tests)):
                    t = tests[idx_test]
                    df = pd.DataFrame(columns=["Epochs", "Q2n", "Q_avg", "SAM", "ERGAS"])

                    gen = self.generate_output(pan=t['pan'].to(self.device),
                                               ms=t['ms'].to(self.device),
                                               ms_lr=t['ms_lr'].to(self.device))
                    gen = torch.permute(gen, (0, 2, 3, 1)).detach().cpu().numpy()
                    gen = recompose(gen)
                    np.clip(gen, 0, 1, out=gen)
                    gen = np.squeeze(gen) * 2048.0
                    gt = np.squeeze(t['gt']) * 2048.0

                    Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut,
                                                                th_values)
                    df.loc[0] = [self.tot_epochs, Q2n, Q_avg, ERGAS, SAM]
                    df.to_csv(t['filename'], index=False, header=True if self.tot_epochs == 1 else False,
                              mode='a', sep=";")

                    writer.add_image(f'gen_img_test_{idx_test}', gen[:, :, 2:0:-1] / 2048, self.tot_epochs,
                                     dataformats='HWC')

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
        results: dict = self.validation_step(dataloader)
        print(f"Evaluation on Test Set: \n ")
        for k in results.keys():
            print(f"\t {k}: {results[k]:>8f} \n")

