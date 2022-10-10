import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv20 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=(7, 7), stride=(1, 1), padding=3,
                                bias=True)

        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.conv20(x)
        rs = self.relu1(rs1)
        return rs


class DRPNN(nn.Module):
    def __init__(self, spectral_num, channels):
        super(DRPNN, self).__init__()
        self.conv2_pre = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channels,
                                   kernel_size=(7, 7), stride=(1, 1), padding=3,
                                   bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(
            ConvBlock(channels),
            ConvBlock(channels),
            ConvBlock(channels),
            ConvBlock(channels),
            ConvBlock(channels),
            ConvBlock(channels),
            ConvBlock(channels),
            ConvBlock(channels),
        )

        self.conv2_post = nn.Conv2d(in_channels=channels, out_channels=spectral_num + 1,
                                    kernel_size=(7, 7), stride=(1, 1), padding=3,
                                    bias=True)

        self.conv2_final = nn.Conv2d(in_channels=spectral_num + 1, out_channels=spectral_num,
                                     kernel_size=(7, 7), stride=(1, 1), padding=3,
                                     bias=True)

    def forward(self, ms, pan):

        input_data = torch.cat([ms, pan], 1)

        rs1 = self.conv2_pre(input_data)
        rs1 = self.relu(rs1)

        rs1 = self.backbone(rs1)

        rs1 = self.conv2_post(rs1)

        rs = torch.add(input_data, rs1)

        rs = self.conv2_final(rs)
        return rs

    def train_loop(self, dataloader, model, loss_fn, optimizer, device='cpu'):
        size = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        # model.train(True)
        loss = 0
        for batch, data in enumerate(dataloader):
            gt, pan, ms = data

            gt = gt.to(device)
            pan = pan.to(device)
            ms = ms.to(device)

            # Compute prediction and loss
            pred = model(ms, pan)
            loss = loss_fn(pred, gt)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            torch.cuda.empty_cache()
            # print(f"loss: {loss:>7f}  [{batch:>5d}/{size//batch_size:>5d}]")

        return loss

    def validation_loop(self, dataloader, model, loss_fn, device='cpu'):
        # model.train(False)

        running_vloss = 0.0
        for i, data in enumerate(dataloader):
            gt, pan, ms = data

            gt = gt.to(device)
            pan = pan.to(device)
            ms = ms.to(device)

            voutputs = model(pan, ms)
            vloss = loss_fn(voutputs, gt)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        return avg_vloss

    def test_loop(self, dataloader, model, loss_fn):
        model.train(False)
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for data in dataloader:
                gt, pan, ms = data

                gt = gt.to(device)
                pan = pan.to(device)
                ms = ms.to(device)

                pred = model(ms, pan)
                test_loss += loss_fn(pred, gt).item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")


    class myData(torch.utils.data.Dataset):
        def __init__(self, dataset_name):
            """ Loads data into memory """
            self.file = h5py.File(dataset_name, 'r')

        def __len__(self):
            """ Returns number of elements of the dataset """
            return self.file['gt'].shape[0]

        def __getitem__(self, index):
            """ Retrieves element at given index """
            gt = torch.from_numpy(np.array(self.file["gt"][index], dtype=np.float32))
            pan = torch.from_numpy(np.array(self.file["pan"][index], dtype=np.float32))
            ms = torch.from_numpy(np.array(self.file["lms"][index], dtype=np.float32))

            return gt, pan, ms

        def close(self):
            self.file.close()


    model = DRPNN(8, 64)
    #torchsummary.summary(model, [(64,64,1), (64,64,8)])

    gg =torch.ones((1, 8,64,64))
    pp = torch.zeros((1, 1,64,64))
    x = model(gg, pp)

    print(x.shape)
    train_dataloader = DataLoader(myData("..\\..\\datasets\\train.h5"), batch_size=64, shuffle=True)

    val_dataloader = DataLoader(myData("..\\..\\datasets\\valid.h5"), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(myData("..\\..\\datasets\\test.h5"), batch_size=64, shuffle=True)
    # gt_test, ms_test, ms_lr_test, pan_test = next(iter(train_dataloader))

    loss_fn = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=.001, weight_decay=0)

    best_vloss = +np.inf

    for epoch in range(5):
        train_loss = model.train_loop(train_dataloader, model, loss_fn, optimizer, device)
        val_loss_avg = model.validation_loop(train_dataloader, model, loss_fn, device)
        print(f'Epoch {epoch + 1}\t'
              f'\t train {train_loss :.2f}\t valid {val_loss_avg:.2f}')

        if val_loss_avg < best_vloss:
            best_vloss = val_loss_avg
            model_path = '../trained_models/drpnn.pth'
            torch.save(model.state_dict(), model_path)

    exit(0)
    model = model.load_state_dict(torch.load(model_path))

    for i, data in enumerate(train_dataloader):
        gt = data['gt'][0]
        ms = data['ms'][0]
        pan = data['pan'][0]

        print(gt.shape)
        print(ms.shape)
        print(pan.shape)

        output = model(torch.unsqueeze(ms, 0), torch.unsqueeze(pan, 0))
        print(output.shape)
        output = torch.squeeze(output)
        print(output.shape)
        output = output.detach().numpy()
        plt.imshow(output[1, :, :], 'gray')
        plt.show()
        plt.imshow(gt[1, :, :], 'gray')
        plt.show()
