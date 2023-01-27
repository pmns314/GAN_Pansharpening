from abc import abstractmethod

import torch

from pytorch_models.NetworkInterface import NetworkInterface


class GanInterface(NetworkInterface):
    """ Common Interface for GANs of the framework """

    def __init__(self, device, name):
        """ Constructor of the class

        Parameters
        ----------
        device : str
            the device onto which train the network (either cpu or a cuda visible device)
        name : str
            the name of the network
        """
        super().__init__(device, name)
        self.adv_loss = None
        self.rec_loss = None

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
    def validation_step(self, dataloader, evaluate_indexes=False):
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
        if evaluation:
            self.generator.eval()
            with torch.no_grad():
                return self.generator(pan, ms)
        return self.generator(pan, ms)

    @abstractmethod
    def set_optimizers_lr(self, lr):
        """ Sets the learning rate of the optimizers

        Parameter
        ---------
        lr : int
            the new learning rate of the optimizers
        """
        pass

    @abstractmethod
    def define_losses(self, rec_loss=None, adv_loss=None):
        """ Defines the losses used during training

        Parameters
        ----------
        rec_loss : str, optional
            reconstruction loss. If None, default is used
        adv_loss: adversarial_loss, optional
            adversarial loss. If None, default is used

        """
        pass
