import argparse
import shutil

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from constants import *
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.CNNs import *
from utils import recompose


def create_model(name: str, channels, device="cpu"):
    name = name.strip().upper()
    if name == "APNN":
        return APNN(channels, device)
    if name == "BDPN":
        return BDPN(channels, device)
    if name == "DRPNN":
        return DRPNN(channels, device)
    if name == "PANNET":
        return PanNet(channels, device)
    if name == "PNN":
        return PNN(channels, device)
    if name == "DICNN":
        return DiCNN(channels, device)
    else:
        raise KeyError("Model not Defined")


def create_test_dict(path, filename):
    test_dict = {}
    test_dataloader1 = DataLoader(DatasetPytorch(path), batch_size=64, shuffle=False)
    pan, ms, ms_lr, gt = next(enumerate(test_dataloader1))[1]
    if len(pan.shape) == 3:
        pan = torch.unsqueeze(pan, 0)
    gt = torch.permute(gt, (0, 2, 3, 1))
    test_dict['pan'] = pan
    test_dict['ms'] = ms
    test_dict['ms_lr'] = ms_lr
    test_dict['gt'] = recompose(torch.squeeze(gt).detach().numpy())
    test_dict['filename'] = filename
    return test_dict


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_model',
                        default='test',
                        help='Provide name of the model. Defaults to test',
                        type=str
                        )
    parser.add_argument('-t', '--type_model',
                        default='apnn',
                        help='Provide type of the model. Defaults to APNN',
                        type=str
                        )
    parser.add_argument('-d', '--dataset_path',
                        default=f'{DATASET_DIR}',
                        help='Provide name of the model. Defaults to ROOT/datasets',
                        type=str
                        )
    parser.add_argument('-s', '--satellite',
                        default='W3',
                        help='Provide satellite to use as training. Defaults to W3',
                        type=str
                        )
    parser.add_argument('-e', '--epochs',
                        default=2,
                        help='Provide number of epochs. Defaults to 1000',
                        type=int
                        )
    parser.add_argument('-lr', '--learning_rate',
                        default=0.01,
                        help='Provide learning rate. Defaults to 0.001',
                        type=float
                        )
    parser.add_argument('-r', '--resume',
                        action='store_true',
                        help='Boolean indicating if resuming the training or starting a new one deleting the one '
                             'already existing, if any'
                        )
    parser.add_argument('-o', '--output_path',
                        default="pytorch_models/trained_models",
                        help='Path of the output folder',
                        type=str
                        )
    parser.add_argument('-f', '--force',
                        action='store_true',
                        help='Boolean indicating if forcing GPU Max Memory allowed'
                        )
    parser.add_argument('--rr',
                        action='store_true',
                        help='Boolean indicating if using Reduced Resolution'
                        )
    parser.add_argument('--no_val',
                        action='store_true',
                        help='Boolean indicating if avoid using the validation set'
                        )

    args = parser.parse_args()

    file_name = args.name_model
    type_model = args.type_model
    satellite = args.satellite
    dataset_path = args.dataset_path
    epochs = args.epochs
    lr = args.learning_rate
    resume_flag = args.resume
    output_base_path = args.output_path
    use_rr = args.rr
    no_val = args.no_val

    type_model = "DICNN"
    data_resolution = "RR" if use_rr else "FR"

    train_dataset = f"test_3_64.h5"
    val_dataset = f"val_1_64.h5"
    test_dataset1 = f"test_3_64.h5"
    test_dataset2 = f"test_3_512.h5"
    test_dataset_FR = f"test_3_512.h5"

    # Device Definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Force 8 Gb max GPU usage
    if args.force and device == "cuda":
        total_memory = torch.cuda.mem_get_info()[1]
        torch.cuda.set_per_process_memory_fraction(8192 / (total_memory // 1024 ** 2))

    # Data Loading
    train_dataloader = DataLoader(DatasetPytorch(f"{dataset_path}/{data_resolution}/{satellite}/{train_dataset}"),
                                  batch_size=64, shuffle=True)
    val_dataloader = DataLoader(DatasetPytorch(f"{dataset_path}/{data_resolution}/{satellite}/{val_dataset}"),
                                batch_size=64, shuffle=True)

    # Model Creation
    model = create_model(type_model, train_dataloader.dataset.channels, device)
    model.to(device)

    output_path = os.path.join(output_base_path, model.name, file_name)

    # Checkpoint path definition
    chk_path = f"{output_path}/checkpoints"

    # Model Loading if resuming training
    if resume_flag and os.path.exists(chk_path) and len(os.listdir(chk_path)) != 0:
        latest_checkpoint = max([int((e.split("_")[1]).split(".")[0]) for e in os.listdir(chk_path)])
        model.load_model(f"{output_path}/model.pth")
        best_losses = model.best_losses
        model.load_model(f"{chk_path}/checkpoint_{latest_checkpoint}.pth")
        model.best_losses = best_losses
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        os.makedirs(chk_path)
        model.compile()

    model.set_optimizer_lr(lr)

    # Setting up index evaluation
    tests = [create_test_dict(f"{dataset_path}/{data_resolution}/{satellite}/{test_dataset1}",
                              f"{output_path}/test_0.csv"),
             create_test_dict(f"{dataset_path}/{data_resolution}/{satellite}/{test_dataset2}",
                              f"{output_path}/test_1.csv"),
             create_test_dict(f"{dataset_path}/FR/{satellite}/{test_dataset_FR}",
                              f"{output_path}/test_FR.csv")]

    # Model Training
    model.train_model(epochs,
                      output_path, chk_path,
                      train_dataloader, None if no_val else val_dataloader,
                      tests)

    # Report
    with open(f"{output_path}/report.txt", "w") as f:
        f.write(f"Network Type : {type_model}\n")
        f.write(f"Datasets Used: \n")
        f.write(f"\t Training: {train_dataset}\n")
        if not no_val:
            f.write(f"\t Validation: {val_dataset}\n")
        f.write(f"\t Test From Training: {test_dataset1}\n")
        f.write(f"\t External Test: {test_dataset2}\n")

        if use_rr:
            f.write(f"\nTrained at Reduced Resolution."
                    f"\n\tDataset for testing at Full Resolution: {train_dataset}\n")
        f.write(f"Number of Trained Epochs: {model.tot_epochs}\n")
        f.write(f"Best Epoch: {model.best_epoch}\n")
