import argparse
import shutil

import torch
from torch.utils.data import DataLoader

from constants import *
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.GANs import *
from utils import recompose


def create_model(name: str, channels, device="cpu", **kwargs):
    name = name.strip().upper()
    if name == "PSGAN":
        return PSGAN(channels, device)
    elif name == "FUPSGAN":
        return FUPSGAN(channels, device)
    elif name == "STPSGAN":
        return STPSGAN(channels, device)
    elif name == "PANGAN":
        train_spat_disc = kwargs['train_spat_disc']
        use_highpass = kwargs['use_highpass']
        return PanGan(channels, device, train_spat_disc=train_spat_disc, use_highpass=use_highpass)
    elif name == "PANCOLORGAN":
        return PanColorGan(channels, device)
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
                        default='psgan',
                        help='Provide type of the model. Defaults to PSGAN',
                        type=str
                        )
    parser.add_argument('-d', '--dataset_path',
                        default=f'{DATASET_DIR}',
                        help='Provide name of the model. Defaults to ROOT/datasets',
                        type=str
                        )
    parser.add_argument('-s', '--satellite',
                        default='W2',
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
    parser.add_argument('-c', '--commit',
                        action='store_true',
                        help='Boolean indicating if commit is to git is needed',
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
    parser.add_argument('--train_spat_disc',
                        action='store_true',
                        help='Boolean indicating if training the spatial discriminator of the PanGan Network'
                        )
    parser.add_argument('--use_highpass',
                        action='store_true',
                        help='Boolean indicating if training using the spatial details of the PanGan Network'
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
    flag_commit = args.commit
    use_rr = args.rr
    no_val = args.no_val
    train_spat_disc = args.train_spat_disc
    use_highpass = args.use_highpass

    data_resolution = "RR" if use_rr else "FR"

    train_dataset = f"train_1_64.h5"
    val_dataset = f"val_1_64.h5"
    test_dataset1 = f"test_1_64.h5"
    test_dataset2 = f"test_2_64.h5"
    test_dataset_FR = f"test_2_512.h5"

    # Device Definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Force 8 Gb max GPU usage
    if args.force and device == "cuda":
        print("Forcing to 8Gb")
        total_memory = torch.cuda.mem_get_info()[1]
        torch.cuda.set_per_process_memory_fraction(8192 / (total_memory // 1024 ** 2))

    # Data Loading
    train_dataloader = DataLoader(DatasetPytorch(f"{dataset_path}/{data_resolution}/{satellite}/{train_dataset}"),
                                  batch_size=64, shuffle=True)
    val_dataloader = DataLoader(DatasetPytorch(f"{dataset_path}/{data_resolution}/{satellite}/{val_dataset}"),
                                batch_size=64, shuffle=False)

    # Model Creation
    model = create_model(type_model, train_dataloader.dataset.channels, device,
                         train_spat_disc=train_spat_disc,
                         use_highpass=use_highpass)
    model.to(device)

    output_path = f"{output_base_path}/{satellite}/{model.name}/{file_name}"

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

    model.set_optimizers_lr(lr)

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
                    f"\n\tDataset for testing at Full Resolution: {test_dataset_FR}\n")
        f.write(f"Number of Trained Epochs: {model.tot_epochs}\n")
        f.write(f"Best Epoch: {model.best_epoch}\n")
        f.write(f"Best Loss: {model.best_losses[0]}\n")
        f.write(f"Learning Rate: {lr}\n")
