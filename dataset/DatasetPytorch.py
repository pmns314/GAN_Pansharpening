import h5py
from torch.utils.data import Dataset, DataLoader

from utils import *


class DatasetPytorch(Dataset):
    def __init__(self, dataset_name):
        """ Loads data into memory """
        self.file = h5py.File(dataset_name, 'r')
        self.channels = self.file['gt'].shape[1]

    def __len__(self):
        """ Returns number of elements of the dataset """
        return self.file['gt'].shape[0]

    def __getitem__(self, index):
        """ Retrieves element at given index """
        # Load Data
        gt = np.array(self.file["gt"][index], dtype=np.float32)
        pan = np.array(self.file["pan"][index], dtype=np.float32)
        ms = np.array(self.file["ms"][index], dtype=np.float32)
        ms_lr = np.array(self.file["ms_lr"][index], dtype=np.float32)


        # Normalization
        gt = norm_max_val(gt)
        pan = norm_max_val(pan)
        ms = norm_max_val(ms)
        ms_lr = norm_max_val(ms_lr)

        # Transform to Pytorch Tensor
        gt = torch.from_numpy(gt)
        pan = torch.from_numpy(pan)
        ms = torch.from_numpy(ms)
        ms_lr = torch.from_numpy(ms_lr)

        if len(pan.shape) != len(ms.shape):
            pan = torch.unsqueeze(pan, 0)

        return pan, ms, ms_lr, gt

    def close(self):
        self.file.close()


if __name__ == '__main__':
    satellite = "W3"

    train_data = DatasetPytorch("../datasets/FR/W3/train_1_64.h5")
    train_dataloader = DataLoader(train_data, batch_size=32,
                                  shuffle=True)

    print(len(train_data))

    pan, ms, ms_lr, gt = (train_dataloader.dataset[0])

    print(ms.shape)

