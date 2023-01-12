""" Loads the model, generates the high resolution image and saves it a .mat file"""
import argparse

from scipy.io import loadmat
from skimage import io as io

from constants import *
from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
from utils import *

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--satellite',
                        default='W3',
                        help='Provide satellite to use as training. Defaults to W3',
                        type=str
                        )
    parser.add_argument('-r', '--results_path',
                        default=f"{ROOT_DIR}/results/GANs",
                        help='Path of the output folder',
                        type=str
                        )

    args = parser.parse_args()

    satellite = args.satellite
    result_folder = args.results_path
    index_test = 1
    imgs = []
    to_show = []
    names = []
    satellite = "W2"
    force_model = None
    main_title = "W2_Miam_Urb"

    names = ["apnn", "PanGan", "PanGan TF", "PanColorGan", "PanColorGan TF",
             "PSGAN", "PSGAN TF",
             "PSGAN con GT"]
    to_show = ["apnn_v2.b",
               "pangan_v2.6",
               "pangan_v3.7",
               "pancolorgan_v2.5",
               "pancolorgan_v3.4",
               "psgan_v3.a",
               "psgan_v3.9",
               ]
    imgs = [[] for _ in range(len(names))]
    assert len(names) == len(to_show) == len(imgs)

    gt = io.imread(f"{result_folder}/{satellite}/gt_{index_test}.tif")
    gt = np.array(gt).astype("float32")
    imgs.insert(0, gt)
    names.insert(0, "GT")

    index = 1
    for type in os.listdir(f"{result_folder}/{satellite}"):
        if force_model is not None and type != force_model:
            continue

        if type[:2] == f"gt":
            continue
        for img_name in os.listdir(f"{result_folder}/{satellite}/{type}"):
            if img_name[-5] != str(index_test):
                continue
            if len(to_show) > 0:
                if img_name[:-11] not in to_show:
                    continue
                else:
                    index = to_show.index(img_name[:-11]) + 1
            img_path = f"{result_folder}/{satellite}/{type}/{img_name}"
            if img_path.split(".")[-1] == "mat":
                mat = loadmat(img_path)
                gen = np.array(mat['gen'])
            else:
                gen = io.imread(img_path)

            if len(to_show) > 0:
                imgs[index] = gen
            else:
                imgs.append(gen)

            Q2n, Q_avg, ERGAS, SAM = indexes_evaluation(gen, gt, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut,
                                                        th_values)

            if len(to_show) > 0:
                new_str = f"{names[index]} Q2n:{Q2n:.4f} SAM:{SAM:.4f}"
                names[index] = new_str
            else:
                new_str = f"{img_name[:-11]} Q2n:{Q2n:.4f} SAM:{SAM:.4f}"
                names.append(new_str)
                index += 1
            print(new_str)

    num_rows = int(np.ceil(np.sqrt(len(imgs))))
    num_cols = num_rows - 1 if len(imgs) <= num_rows ** 2 - num_rows else num_rows

    for img in imgs:
        assert img.shape == gt.shape

    # lin_stretched = linear_strech(np.concatenate(imgs, 1)[:, :, (0, 2, 4)])
    # cnt = 1
    # plt.figure()
    # dim_img = gt.shape[0]
    # for i in range(len(imgs)):
    #     plt.subplot(num_cols, num_rows, cnt)
    #     img_to_show = lin_stretched[:, dim_img * i:dim_img * i + dim_img, :]
    #     plt.imshow(img_to_show[:, :, ::-1])
    #     plt.title(names[i])
    #     plt.axis('off')
    #     cnt += 1
    #
    # plt.suptitle(main_title)

    ##################################################
    cnt = 1
    plt.figure()
    dim_img = gt.shape[0]
    for i in range(len(imgs)):
        plt.subplot(num_cols, num_rows, cnt)
        img_to_show = linear_strech(imgs[i][:, :, (0, 2, 4)], i == 0)
        plt.imshow(img_to_show[:, :, ::-1])
        plt.title(names[i])
        plt.axis('off')
        cnt += 1

    plt.suptitle(main_title)

    plt.show()
