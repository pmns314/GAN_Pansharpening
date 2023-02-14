# GAN_Pansharpening

Pytorch framework for training and testing CNN and GAN networks
for Pansharpening

Developed by Paolo Mansi
***

## How to use
### Dataset setup
1. Download PAirMax dataset from https://resources.maxar.com/product-samples/pansharpening-benchmark-dataset
and place it inside the _original_data_ folder

Alternatively
1. Make sure the data  inside the _original_data_ folder are organised with the following structure
where *SENSOR TAG* is one of GE, W2, W3, W4
    ```
    ├───*SENSOR TAG*_image name 1
    │   ├───FR
    │   └───RR
    ├───*SENSOR TAG*_image name 2
    │   ├───FR
    │   └───RR
     ...
    ```

2. Run MATLAB script _generate_data.m_
3. Run _generate_patch_dataset.py_

### Train Network
Run

    python train_file.py
For information  about parameters run:

    python train_file.py -h

### Generate Output
Run

    python generate_image.py
For information about parameters, run:
    
    python generate_image.py -h
***
## How To Contribute to the Project

### Add Loss Function
* Inside the _Losses.py_ create a New Loss class extending _torch.nn.Module_
* Inside _Losses.py_ add entry to the Enum **Losses** like
    > KEY = NewLossClass 
    >- KEY must be a capital letter word


### Add Adversarial Loss Function
1. Inside _adversarial_losses_ package create a new Adversarial loss class extending _torch.nn.Module_
2. Inside ```adversarial_losses/__init__.py``` :
  * add import
    > from .new_file import newClass
  * add entry to the Enum ***AdvLosses*** like
    >   KEY = NewAdversarialLossClass 
    >  - KEY must be a capital letter word


### Add CNN
1. Inside _CNNs_ package create a new CNN class extending _CnnInterface_
2. Inside ```CNNs_/__init__.py``` :
  * add import
    > from .new_file import newClass
  * add entry to the Enum ***CNNS*** like
    >   KEY = NewCNNClass 
    >  - KEY must be a capital letter word

### Add GAN
1. Inside _GANs_ package create a new GAN class extending _GanInterface_
2. Inside ```GANs_/__init__.py``` :
  * add import
    > from .new_file import newClass
  * add entry to the Enum ***GANS*** like
    >   KEY = NewGANClass 
    >  - KEY must be a capital letter word
***

## References
* Python Quality index Toolbox : https://sites.google.com/site/vivonegemine/code?authuser=0
 