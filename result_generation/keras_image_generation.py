import argparse
import os.path

import numpy as np
import scipy
from PIL import Image
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from dataset.DataGeneratorKeras import DataGenerator, norm_min_max, inv_norm_mean, norm_mean
from keras_models.CNNs.PNN import PNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_name", help="Name of the model")
    args = parser.parse_args()

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "PSgan_max_val"

    satellite = "W3"
    train_gen = DataGenerator("./datasets/" + satellite + "/train.h5")
    test_gen = DataGenerator("./datasets/" + satellite + "/test.h5", shuffle=False)
    pan, ms, _ = next(train_gen.generate_batch())[0]
    output_path = "./results/" + satellite + "/"
    model_path1 = "./keras_models/trained_models/" + model_name

    model = load_model(model_path1)
    model.summary()
    [pan, ms, _], gt = next(test_gen.generate_batch())

    gen = model.predict([pan, ms, _])

    scipy.io.savemat(output_path + model_name + ".mat", dict(g=gen))

    plt.imshow(gt[0, :, :, :3])
    plt.figure()
    plt.imshow(gen[0, :, :, :3])
    plt.show()

    #
