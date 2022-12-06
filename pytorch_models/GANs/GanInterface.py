from abc import abstractmethod
from pytorch_models.NetworkInterface import NetworkInterface


class GanInterface(NetworkInterface):
    def __init__(self, device, name):
        super().__init__(device, name)

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
    def generate_output(self, pan, evaluation=True, **kwargs):
        pass

    @abstractmethod
    def set_optimizers_lr(self, lr):
        pass
