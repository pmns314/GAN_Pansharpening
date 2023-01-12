from abc import ABC, abstractmethod
import pandas as pd
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from constants import *
from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
from utils import adjust_image


class NetworkInterface(ABC, nn.Module):
    """ Common Interface for all the networks of the framework """

    def __init__(self, device, name):
        """ Constructor of the class

        Parameters
        ----------
        device : str
            the device onto which train the network (either cpu or a cuda visible device)
        name : str
            the name of the network
        """

        super().__init__()
        self._model_name = name
        self.best_losses: list = NotImplemented  # Must be defined by subclasses
        self.best_epoch = 0
        self.tot_epochs = 0
        self.device = device
        self.use_ms_lr = False
        self.to(device)

    @property
    def name(self):
        """ Returns the name of the network"""
        return self._model_name

    # ------------------ Abstract Methods -------------------------
    @abstractmethod
    def train_step(self, dataloader):
        """ Defines the operations to be carried out during the training step

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            the dataloader that loads the training data
        """
        pass

    @abstractmethod
    def validation_step(self, dataloader):
        """ Defines the operations to be carried out during the validation step

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            the dataloader that loads the validation data
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """ Saves the model as a .pth file

        Parameters
        ----------

        path : str
            the path where the model has to be saved into
        """
        pass

    @abstractmethod
    def load_model(self, path, weights_only=False):
        """ Loads the network model

        Parameters
        ----------

        path : str
            the path of the model
        weights_only : bool, optional
            True if only the weights of the generator must be loaded, False otherwise (default is False)

        """
        pass

    @abstractmethod
    def generate_output(self, pan, ms, evaluation=True):
        """
        Generates the output image of the network

        Parameters
        ----------
        pan : tensor
            the panchromatic image fed to the network
        ms : tensor
            the multi spectral image fed to the network
        evaluation: bool, optional
            True if the network must be switched into evaluation mode, False otherwise (default is True)

        """
        pass

    @abstractmethod
    def set_optimizers_lr(self, lr):
        """ Sets the learning rate of the optimizers

        Parameter
        ---------
        lr : int
            the new learning rate of the optimizers
        """
        pass

    # ------------------------- Concrete Methods ------------------------------
    def train_model(self, epochs,
                    output_path, chk_path,
                    train_dataloader, val_dataloader,
                    tests=None):
        """
        Method for fitting the model.

        For each epoch, the following operations are carried out:
            1. Compute the Losses of Training Step ( and the validation step, if any).
            2. Update the Tensorboard Log.
            3. Update the Best losses of the network.

            4. If checkpoint epoch:
                1. Save model as checkpoint.
                2. Evaluate Network for each of the provided tests saving results in csv file.
        Finally, it evaluates the best model and prints the output.

        Parameters
        ----------
        epochs : int
            number of training epochs
        output_path : str
            model saving path
        chk_path : str
            checkpoints saving path
        train_dataloader : torch.utils.data.DataLoader
            Data loader of the Training Data
        val_dataloader : torch.utils.data.DataLoader
            Data loader of Validation Data
        tests : dict, optional
            Dictionary of test data
        """

        # TensorBoard
        writer = SummaryWriter(output_path + "/log")

        # Training
        print(f"Training started for {output_path} at epoch {self.tot_epochs + 1}")
        ending_epoch = self.tot_epochs + epochs
        for epoch in range(epochs):
            self.tot_epochs += 1
            print(f'\nEpoch {self.tot_epochs}/{ending_epoch}')

            # Compute Losses on Train Set
            train_losses = self.train_step(train_dataloader)
            if val_dataloader is not None:
                # Compute Losses on Validation Set if exists
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
            # Saves the model if the loss in position 0 improved.
            # If CNN, that's the only loss; if GAN, that's the loss of the generator
            if losses[0] < self.best_losses[0]:
                self.best_losses[0] = losses[0]
                self.best_epoch = self.tot_epochs
                self.save_model(f"{output_path}/model.pth")
                print(f"New Best Loss {self.best_losses[0]:.3f} at epoch {self.best_epoch}")

            # This is ignored for CNNs
            for i in range(1, len(losses)):
                if losses[i] < self.best_losses[i]:
                    self.best_losses[i] = losses[i]

            if self.tot_epochs in TO_SAVE or epoch == epochs - 1:
                # Save Checkpoints
                self.save_model(f"{chk_path}/checkpoint_{self.tot_epochs}.pth")

                # Generation Evaluation Indexes
                for idx_test in range(len(tests)):
                    t = tests[idx_test]
                    df = pd.DataFrame(columns=["Epochs", "Q2n", "Q_avg", "ERGAS", "SAM"])

                    gen = self.generate_output(pan=t['pan'].to(self.device),
                                               ms=t['ms'].to(self.device) if self.use_ms_lr is False else t['ms_lr'].to(
                                                   self.device),
                                               evaluation=True)
                    try:
                        gen = adjust_image(gen, t['ms_lr'])
                        gt = adjust_image(t['gt'])

                        Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds,
                                                                    dim_cut,
                                                                    th_values)
                        df.loc[0] = [self.tot_epochs, Q2n, Q_avg, ERGAS, SAM]
                        df.to_csv(t['filename'], index=False, header=True if self.tot_epochs == 1 else False,
                                  mode='a', sep=";")
                    except:
                        print("error in calculating outputs")

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

        t = tests[-1]
        gen = self.generate_output(pan=t['pan'].to(self.device),
                                   ms=t['ms'].to(self.device) if self.use_ms_lr is False else t['ms_lr'].to(
                                       self.device),
                                   evaluation=True)
        gen = adjust_image(gen, t['ms_lr'])
        gt = adjust_image(t['gt'])

        Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds,
                                                    dim_cut,
                                                    th_values)

        print(f"Best Model Results:\n"
              f"\t Q2n: {Q2n :.4f}  Q_avg:{Q_avg:.4f}"
              f" ERGAS:{ERGAS:.4f} SAM:{SAM:.4f}")

    def test_model(self, dataloader):
        """
        Tests the model calling the validation step on the input data

        Parameters
        ---------
        dataloader : torch.utils.data.DataLoader
            Data loader of Testing Data
        """
        results: dict = self.validation_step(dataloader)
        print(f"Evaluation on Test Set: \n ")
        for k in results.keys():
            print(f"\t {k}: {results[k]:>8f} \n")
