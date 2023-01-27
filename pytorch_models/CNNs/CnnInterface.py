from abc import abstractmethod

import numpy as np
import torch

from pytorch_models.NetworkInterface import NetworkInterface
from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation


class CnnInterface(NetworkInterface):
    """ Common Interface for the CNN networks of the framework """

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
        self.best_losses = [np.inf]
        self.opt = None
        self.loss_fn = None

    # ------------------ Abstract Methods -------------------------
    @abstractmethod
    def compile(self, loss_fn=None, optimizer=None):
        """ Compiles CNN model

        Parameters
        ----------
        loss_fn : Loss, optional
            loss function used for calculating losses. If None, default is used
        optimizer : Optimizer, optional
            optimizer used during training. If None, default is used
        """
        pass

    # ------------------------- Concrete Methods ------------------------------

    def train_step(self, dataloader):
        """ Defines the operations to be carried out during the training step

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            the dataloader that loads the training data
        """
        self.train(True)

        loss_batch = 0
        for batch, data in enumerate(dataloader):
            pan, ms, ms_lr, gt = data

            if len(pan.shape) == 3:
                pan = torch.unsqueeze(pan, 0)
            gt = gt.to(self.device)
            pan = pan.to(self.device)
            if self.use_ms_lr is False:
                multi_spectral = ms.to(self.device)
            else:
                multi_spectral = ms_lr.to(self.device)

            # Compute prediction and loss
            pred = self.generate_output(pan, multi_spectral, evaluation=False)
            loss = self.loss_fn(pred, gt)

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss = loss.item()
            torch.cuda.empty_cache()

            loss_batch += loss
        try:
            self.loss_fn.reset()
        except:
            pass
        return {"Loss": loss_batch / len(dataloader)}

    def validation_step(self, dataloader, evaluate_indexes=False):
        """ Defines the operations to be carried out during the validation step

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            the dataloader that loads the validation data
        """
        self.train(False)
        self.eval()
        running_vloss = 0.0

        running_q2n = 0.0
        running_q = 0.0
        running_sam = 0.0
        running_ergas = 0.0
        i = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                pan, ms, ms_lr, gt = data

                if len(pan.shape) == 3:
                    pan = torch.unsqueeze(pan, 0)
                gt = gt.to(self.device)
                pan = pan.to(self.device)

                if self.use_ms_lr is False:
                    multi_spectral = ms.to(self.device)
                else:
                    multi_spectral = ms_lr.to(self.device)

                # Compute prediction and loss
                voutputs = self.generate_output(pan, multi_spectral)
                vloss = self.loss_fn(voutputs, gt)
                running_vloss += vloss.item()

                # Compute indexes
                if evaluate_indexes:
                    voutputs = torch.permute(voutputs, (0, 2, 3, 1)).detach().cpu().numpy()
                    gt_all = torch.permute(gt, (0, 2, 3, 1)).detach().cpu().numpy()
                    for i in range(voutputs.shape[0]):
                        gt = gt_all[i, :, :, :]
                        gen = voutputs[i, :, :, :]
                        indexes = indexes_evaluation(gt, gen, 4, 11, 31, False, None, True)
                        running_q2n += indexes[0]
                        running_q += indexes[1]
                        running_ergas += indexes[2]
                        running_sam += indexes[3]

        avg_vloss = running_vloss / (i + 1)
        q2n_tot = running_q2n / (i + 1)
        q_tot = running_q / (i + 1)
        ergas_tot = running_ergas / (i + 1)
        sam_tot = running_sam / (i + 1)
        try:
            self.loss_fn.reset()
        except:
            pass
        return {"Loss": avg_vloss}, [q2n_tot, q_tot, ergas_tot, sam_tot]

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
            self.eval()
            with torch.no_grad():
                return self(pan, ms)
        return self(pan, ms)

    def save_model(self, path):
        """ Saves the model as a .pth file

        Parameters
        ----------

        path : str
            the path where the model has to be saved into
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss_fn': self.loss_fn,
            'best_losses': self.best_losses,
            'best_epoch': self.best_epoch,
            'tot_epochs': self.tot_epochs,
            'metrics': [self.best_q, self.best_q_avg, self.best_sam, self.best_ergas]
        }, path)

    def load_model(self, path, weights_only=False):
        """ Loads the network model

        Parameters
        ----------

        path : str
            the path of the model
        weights_only : bool, optional
            True if only the weights of the generator must be loaded, False otherwise (default is False)

        """
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

        try:
            self.best_q, self.best_q_avg, self.best_sam, self.best_ergas = trained_model['metrics']
        except KeyError:
            pass

    def set_optimizers_lr(self, lr):
        """ Sets the learning rate of the optimizers

        Parameter
        ---------
        lr : int
            the new learning rate of the optimizers
        """
        for g in self.opt.param_groups:
            g['lr'] = lr
