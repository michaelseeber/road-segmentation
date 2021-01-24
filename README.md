![](https://github.com/michaelseeber/road-segmentation/blob/main/overlays.png)

# Road Segmentation from Aerial images

This repo containts the projects source code of Bastian Morath, Daniel Peter, Michael Seeber and Kam-Ming Mark Tam which is part of a Kaggle competition that is available under [this link](https://www.kaggle.com/c/cil-road-segmentation-2020/).

## Report
The written report can be found [here](https://github.com/michaelseeber/road-segmentation/blob/main/report.pdf).

## Project structure

Following list describes the directories and their use:

| Folder          | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| helpers          | This folder contains all various helper methods such as metrics, plots, post and preprocessing |
| models             | This folder contains the source code for our main models used in the report               |
| backlog| Contains various things that were developed and used during the process of this project, but not really needed anymore |

## Prerequisites

In order to run this project following requirements need to be met.

### Environment Setup

This project is based on Python 3.7 and Tensorflow (tested with version 2.2). Therefore the environment needs to be set up with following packages:

- opencv_python==4.3.0.36
- scikit_image==0.17.2
- numpy==1.19.0
- pip==20.1.1
- tensorflow==2.2.0
- matplotlib==3.3.0
- wheel==0.34.2
- pandas==1.1.0
- Pillow==7.2.0
- skimage==0.0

### How to run on Leonhard (ETHZ GPU cluster)
```
ssh <nethz-name>@login.leonhard.ethz.ch
module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1

cd cil-road-segmentation-2020/models/
bsub -I -n 1 -R "rusage[mem=9000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python report_unet.py
```

# Training the model

## Executing the training

1. download the training dataset from: https://polybox.ethz.ch/index.php/s/F0F81UIdITCQa6K
2. download the testing dataset from: https://polybox.ethz.ch/index.php/s/TJq3KqorQk0Pppy
3. extract the downloaded folders in `/cil-road-segmentation-2020/training` and `/cil-road-segmentation-2020/testing`


### UNet
The UNet models can be found in `'/cil-road-segmentation-2020/models/'`

| Report         | filename in repository                                              |
| --------------- | ------------------------------------------------------------ |
| U-net (baseline)         | report_unet.py                                      |
| U-net with cropping        | report_unet_5x5.py                                      |
| U-net with augmentation        | report_unet_5x5_augmented.py                                      |
| U-net with cropping and dilation        | report_unet_dilated_5x5.py                                      |
| U-net with augmentation adn dilation        | report_unet_dilated_5x5_augmented.py                                      |
| U-net snapshot        | report_unet_dilated_5x5_augmented_snapshot_sgd_4000.py                                      |

### Pix2Pix
The Pix2Pix model can be found as Jupyter Notebook also in `'/cil-road-segmentation-2020/models/'`
More detailed information can be found inside the Jupyter Notebooks.
-   `pix2pix.ipynb`
-   `pix2pix_snapshot.ipynb`  (based on previous but includes snapshot blending)

| Report         | filename in repository                                              |
| --------------- | ------------------------------------------------------------ |
| Pix2Pix with augmentation         | pix2pix.ipynb                                     |
| Pix2Pix with snapshot       | pix2pix_snapshot.ipynb                                      |


### Blending
Once the Pix2Pix and the UNet predictions are available one can use the notebook `'mergePredictions.ipynb'` to receive the final blended prediction.


## Helpers
Some experiments such as postprocessing were done after saving the predictions to disk, i.e. not included directly into the model pipeline. Those files can be found in the folder `'/helpers/'`.
