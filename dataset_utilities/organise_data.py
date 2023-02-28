import os.path
import shutil

import skimage.io as skio
from scipy.io import savemat

from quality_indexes_toolbox import *

if __name__ == '__main__':
    folder_path = '..\\original_images\\'
    folder_path = "C:\\Users\\pmans\\Documents\\Magistrale\\Remote Sensing\\Progetto\\PAirMax\\"
    base_output_path = "..\\data\\"
    ratio = 4
    index = -1

    if os.path.exists(base_output_path):
        shutil.rmtree(base_output_path)
    os.mkdir(base_output_path)

    for resolution in ["FR", "RR"]:
        # Create Output Folder
        output_path = f"{base_output_path}\\{resolution}"
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

        # Loop Over Images of the Dataset
        for img in os.listdir(folder_path):
            spl_str = img.split("_")
            if len(spl_str) != 3:
                continue

            # Extract Sensor Name
            sensor_img = spl_str[0]

            if sensor_img == "W2":
                sensor = "WV2"
            elif sensor_img == "W3":
                sensor = "WV3"
            elif sensor_img == "W4":
                sensor = "WV4"
            elif sensor_img == "GE":
                sensor = "GeoEye1"
            else:
                sensor = "none"

            # Create satellite output folder
            satellite_folder = f"{output_path}\\{sensor_img}"
            if not os.path.exists(satellite_folder):
                os.mkdir(satellite_folder)
                index = 1
            else:
                index += 1

            # Read Full Resolution Images (According to Reduced Resolution Protocol)
            I_PAN = skio.imread(f"{folder_path}\\{img}\\RR/PAN.tif").astype("double")
            I_MS = skio.imread(f"{folder_path}\\{img}\\RR/MS.tif").astype("double")
            I_MS_LR = skio.imread(f"{folder_path}\\{img}\\RR/MS_LR.tif").astype("double")
            I_GT = skio.imread(f"{folder_path}\\{img}\\RR/GT.tif").astype("double")

            # Generate Downgraded Version
            if resolution == "RR":
                I_GT = I_MS_LR
                [I_MS_LR, I_PAN] = resize_images(I_MS_LR, I_PAN, ratio, sensor)
                I_MS = interp23(I_MS_LR, ratio)

            savemat(f"{satellite_folder}\\{img}.mat", dict(I_PAN=I_PAN, I_GT=I_GT, I_MS=I_MS, I_MS_LR=I_MS_LR))

    print("Images Organization Completed")
