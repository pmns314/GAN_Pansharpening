import argparse
import shutil

import torch
from torch.utils.data import DataLoader

from constants import *
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models.CNNS import CNNS
from pytorch_models.GANS import GANS
from pytorch_models.AdvLosses import AdvLosses
from pytorch_models.Losses import Losses


def create_model(name: str, channels, device="cpu", **kwargs):
    name = name.strip().upper()
    try:
        model = GANS[name].value(channels, device)

        # Adversarial Loss Definition
        adv_loss = kwargs['adv_loss_fn'].strip().upper() if "adv_loss_fn" in kwargs else None
        if adv_loss is not None:
            adv_loss = AdvLosses[adv_loss].value()

        # Reconstruction Loss Definition
        rec_loss = kwargs['loss_fn'].strip().upper() if "loss_fn" in kwargs else None
        if rec_loss is not None:
            print(rec_loss)
            rec_loss = Losses[rec_loss].value()

        model.define_losses(rec_loss=rec_loss, adv_loss=adv_loss)
        return model
    except KeyError:
        pass

    try:
        model = CNNS[name].value(channels, device)
        # Reconstruction Loss Definition
        loss_fn = kwargs['loss_fn'].strip().upper() if "loss_fn" in kwargs else None
        if loss_fn is not None:
            print(loss_fn)
            loss_fn = Losses[loss_fn].value()

        optimizer = kwargs['optimizer'] if "optimizer" in kwargs else None
        model.compile(loss_fn, optimizer)
        return model
    except KeyError:
        pass

    raise KeyError("Model not defined!")


def create_test_dict(path, filename):
    test_dict = {}
    test_dataloader1 = DataLoader(DatasetPytorch(path), batch_size=64, shuffle=False)
    pan, ms, ms_lr, gt = next(enumerate(test_dataloader1))[1]
    if len(pan.shape) == 3:
        pan = torch.unsqueeze(pan, 0)
    test_dict['pan'] = pan
    test_dict['ms'] = ms
    test_dict['ms_lr'] = ms_lr
    test_dict['gt'] = gt
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
                        help=f'Provide type of the model. Select one of the followings.\n'
                             f'\tGANs Choices: {[e.name for e in GANS]}\n'
                             f'\tCNNs Choices: {[e.name for e in CNNS]}\n'
                             f'Defaults to PSGAN',
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
    parser.add_argument('-b', '--base_path',
                        default=None,
                        help='Path of the model to use as basis of the training',
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

    parser.add_argument('--source_dataset',
                        help='Choose from Train, Train&Val, Train&Val&Test and Test',
                        type=str,
                        nargs="+",
                        default=["Train", "Test", "Test"]
                        )
    parser.add_argument('--index_images',
                        help='Indexes',
                        type=int,
                        nargs="+",
                        default=[1, 1, 1]
                        )
    parser.add_argument('--patch_size',
                        help='Set Patch Sizes',
                        type=int,
                        nargs="+",
                        default=[64, 64, 512]
                        )
    parser.add_argument('-adv', '--adv_loss_fn',
                        help=f'Provide type of adversarial loss. Select one of the followings.\n'
                             f'\t minmax, lsgan, ragan. If unset, uses default',
                        type=str
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
    base_path = args.base_path
    source_dataset = args.source_dataset
    index_images = args.index_images
    patch_size = args.patch_size

    assert len(patch_size) == len(index_images) == len(source_dataset)

    # Device Definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Force 8 Gb max GPU usage
    if args.force and device == "cuda":
        print("Forcing to 8Gb")
        total_memory = torch.cuda.mem_get_info()[1]
        torch.cuda.set_per_process_memory_fraction(8192 / (total_memory // 1024 ** 2))

    dataset_settings = list(zip(source_dataset, index_images, patch_size))
    data_resolution = "RR3" if use_rr else "FR3"

    # Data Loading
    cnt = 0
    train_dataset = f"train_{dataset_settings[cnt][1]}_{dataset_settings[cnt][2]}.h5"
    train_dataloader = DataLoader(
        DatasetPytorch(f"{dataset_path}/{data_resolution}/{dataset_settings[cnt][0]}/{satellite}/{train_dataset}"),
        batch_size=64, shuffle=True)
    cnt += 1
    if no_val:
        val_dataset = None
        val_dataloader = None
    else:
        val_dataset = f"val_{dataset_settings[cnt][1]}_{dataset_settings[cnt][2]}.h5"
        val_dataloader = DataLoader(
            DatasetPytorch(f"{dataset_path}/{data_resolution}/{dataset_settings[cnt][0]}/{satellite}/{val_dataset}"),
            batch_size=64, shuffle=False)
        cnt += 1

    # Model Creation
    model = create_model(type_model, train_dataloader.dataset.channels, device, **vars(args))
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
        if base_path is not None:
            print(f"Using weights from {base_path}")
            model.load_model(base_path, weights_only=True)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        os.makedirs(chk_path)

    model.set_optimizers_lr(lr)

    # Setting up index evaluation
    tests = []
    for cnt in range(cnt, len(dataset_settings) - 1):
        test_dataset1 = f"test_{dataset_settings[cnt][1]}_{dataset_settings[cnt][2]}.h5"
        tests.append(
            create_test_dict(f"{dataset_path}/{data_resolution}/{dataset_settings[cnt][0]}/{satellite}/{test_dataset1}",
                             f"{output_path}/test_{dataset_settings[cnt][1]}_{dataset_settings[cnt][2]}.csv"))
    cnt += 1
    test_dataset1 = f"test_{dataset_settings[cnt][1]}_{dataset_settings[cnt][2]}.h5"
    tests.append(
        create_test_dict(f"{dataset_path}/FR3/{dataset_settings[cnt][0]}/{satellite}/{test_dataset1}",
                         f"{output_path}/test_FR.csv"))

    # Model Training
    model.train_model(epochs,
                      output_path, chk_path,
                      train_dataloader, val_dataloader,
                      tests)

    # Report
    with open(f"{output_path}/report.txt", "w") as f:
        f.write(f"Network Type : {type_model}\n")
        f.write(f"Datasets Used: \n")
        f.write(f"\tSatellite: {satellite} \n")
        for ds in dataset_settings:
            f.write(f"\t{ds[0]} - Image: {ds[1]} - Patch Size: {ds[2]}\n")

        if use_rr:
            f.write(f"\nTrained at Reduced Resolution.\n")
        f.write(f"Number of Trained Epochs: {model.tot_epochs}\n")
        f.write(f"Best Epoch: {model.best_epoch}\n")
        f.write(f"Best Loss: {model.best_losses[0]}\n")
        f.write(f"Learning Rate: {lr}\n")
