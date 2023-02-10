import argparse
import shutil

from torch.utils.data import DataLoader, ConcatDataset

from constants import *
from dataset.DatasetPytorch import DatasetPytorch
from pytorch_models import *
from util2 import create_model


def create_test_dict(data_path: str, filename: str):
    """ Creates the dictionary of data for testing the network during the training

    Parameters
    ---------
    data_path : str
        path of the dataset to user for testing
    filename : str
        name of the .csv file used to store the results
    """
    test_dict = {}
    test_dataloader1 = DataLoader(DatasetPytorch(data_path), batch_size=64, shuffle=False)
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
                             f'Defaults to APNN',
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
                        default=1000,
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
                        default="Train"
                        )
    parser.add_argument('--index_image',
                        help='Indexes',
                        type=int,
                        default=1
                        )
    parser.add_argument('--patch_size',
                        help='Set Patch Sizes',
                        type=int,
                        default=64
                        )
    parser.add_argument('-adv', '--adv_loss_fn',
                        help=f'Provide type of adversarial loss. Select one of the followings.\n'
                             f'\t {[e.name for e in AdvLosses]} If unset, uses default',
                        type=str
                        )
    parser.add_argument('-loss', '--loss_fn',
                        help=f'Provide type of reconstruction loss. Select one of the followings.\n'
                             f'\t {[e.name for e in Losses]} . If unset, uses default',
                        type=str
                        )
    parser.add_argument('-opt', '--optimizer',
                        help=f'Provide type of reconstruction loss. Select one of the followings.\n'
                             f'\t ... . If unset, uses default',
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
    source_dataset = args.source_dataset.strip()
    index_image = args.index_image
    patch_size = args.patch_size

    # Device Definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Force 8 Gb max GPU usage
    if args.force and device == "cuda":
        print("Forcing to 8Gb")
        total_memory = torch.cuda.mem_get_info()[1]
        torch.cuda.set_per_process_memory_fraction(8192 / (total_memory // 1024 ** 2))

    # dataset_settings = list(zip(source_dataset, index_images, patch_size))
    data_resolution = "RR3" if use_rr else "FR3"

    # Data Loading
    cnt = 0
    prefix = "train" if source_dataset != "Test" else "test"
    train_dataset = f"train_{index_image}_64.h5"
    train_data1 = DatasetPytorch(f"{dataset_path}/{data_resolution}/{source_dataset}/W3/{train_dataset}")

    x = 1
    if x == 1:
        train_data_all = train_data1
    elif x == 2:
        # train_data_all = ConcatDataset([train_data1, train_data2])
        pass
    else:
        # train_data_all = ConcatDataset([train_data1, train_data2, train_data3, train_data4, train_data5, train_data6])
        pass
    train_dataloader = DataLoader(train_data1, batch_size=64, shuffle=True)
    cnt += 1

    if prefix == "test":
        no_val = True

    if no_val:
        val_dataset = None
        val_dataloader = None
    else:
        val_dataset = f"train_1_2_64.h5"
        val_data1 = DatasetPytorch(f"{dataset_path}/{data_resolution}/W3/{val_dataset}")
        val_dataset = f"train_2_2_64.h5"
        val_data2 = DatasetPytorch(f"{dataset_path}/{data_resolution}/W3/{val_dataset}")
        val_dataset = f"train_3_2_64.h5"
        val_data3 = DatasetPytorch(f"{dataset_path}/{data_resolution}/W3/{val_dataset}")

        if x == 1:
            val_data_all = val_data1
        elif x == 2:
            val_data_all = ConcatDataset([val_data1, val_data2])
        else:
            val_data_all = ConcatDataset([val_data1, val_data2, val_data3])

        val_data_all = DatasetPytorch(f"{dataset_path}/{data_resolution}/Test/W3/{val_dataset}")
        val_dataloader = DataLoader(val_data_all, batch_size=64, shuffle=False)
        cnt += 1

    val_dataset = f"test_{index_image}_128.h5" if use_rr else f"test_{index_image}_512.h5"
    val_data_all = DatasetPytorch(f"{dataset_path}/{data_resolution}/Test/{satellite}/{val_dataset}")
    val_dataloader = DataLoader(val_data_all, batch_size=64, shuffle=False)
    # Model Creation

    model = create_model(type_model, 8, device, **vars(args))
    model.to(device)

    output_path = f"{output_base_path}/{satellite}/{model.name}/{file_name}"
    model.output_path = output_path

    # Checkpoint path definition
    chk_path = f"{output_path}/checkpoints"

    # Model Loading if resuming training
    if resume_flag and os.path.exists(chk_path) and len(os.listdir(chk_path)) != 0:
        latest_checkpoint = max([int((e.split("_")[1]).split(".")[0]) for e in os.listdir(chk_path)])
        model.load_model(f"{output_path}/model.pth")
        best_losses = model.best_losses
        best_q = model.best_q
        model.load_model(f"{chk_path}/checkpoint_{latest_checkpoint}.pth")
        model.best_losses = best_losses
        model.best_q = best_q
    else:
        if base_path is not None:
            print(f"Using weights from {base_path}")
            model.load_model(base_path, weights_only=True)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        os.makedirs(chk_path)

    model.set_optimizers_lr(lr)

    RR_test_dict = create_test_dict(f"{dataset_path}/RR3/Test/{satellite}/test_{index_image}_128.h5",
                                    f"{output_path}/test_{index_image}_RR.csv") if use_rr else None

    FR_test_dict1 = create_test_dict(f"{dataset_path}/FR3/Test/{satellite}/test_{index_image}_512.h5",
                                     f"{output_path}/test_{index_image}_FR.csv")


    # FR_test_dict4 = create_test_dict(f"{dataset_path}/{data_resolution}/W3/test_1_3_64.h5",
    #                                  f"{output_path}/test_img_1_quad_3.csv")
    # FR_test_dict5 = create_test_dict(f"{dataset_path}/{data_resolution}/W3/test_2_3_64.h5",
    #                                  f"{output_path}/test_img_2_quad_3.csv")
    # FR_test_dict6 = create_test_dict(f"{dataset_path}/{data_resolution}/W3/test_3_3_64.h5",
    #                                 f"{output_path}/test_img_3_quad_3.csv")
    # Model Training
    model.train_model(epochs,
                      output_path, chk_path,
                      train_dataloader, val_dataloader, [FR_test_dict1],
                      rr_test=None)

    # Report
    with open(f"{output_path}/report.txt", "w") as f:
        f.write(f"Network Type : {type_model}\n")
        f.write(f"Datasets Used: \n")
        f.write(f"\tSatellite: {satellite} \n")
        if use_rr:
            f.write(f"\nTrained at Reduced Resolution.\n")
        f.write(f"Number of Trained Epochs: {model.tot_epochs}\n")
        f.write(f"Best Epoch: {model.best_epoch}\n")
        f.write(f"Best Loss: {model.best_losses[0]}\n")
        f.write(f"Learning Rate: {lr}\n")
