# CIL Road Segmentation 2020

## how to run on Leonhard

```
ssh <nethz-name>@login.leonhard.ethz.ch
module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1

cd cil-road-segmentation-2020/
bsub -I -n 1 -R "rusage[mem=9000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python <file you want to run>
```

## Repository Structure

Here are some guidelines on how tu structure the code / folders and submissions. Please see models/fcn_example.py for an example file.

1. Put your python files into the 'models/' folder
2. Copy-paste the following imports, which gives access to helper methods etc.

```
import sys
sys.path.append('../') # Adjust this if your file is not in the '/model' folder
from helpers.metrics import *
from helpers.augment import *
from helpers.display import *
from helpers.paths import *
from helpers.submission import *
import tensorflow as tf
import numpy as np
import os
import argparse
```
3. Try to use the provided helper functions for the most common methods. If needed, add new methods (e.g. metrics)
4. When submitting a file to kaggle, name the csv file (and the corresponding python and tensorboard file)  
  "submission_yyyy_mm_dd_hh__mm" such as "submission_2020_06_07_20_13.csv / .py / .tensorboard" and add them to the kaggle folder



## how to run TensorBoard locally to monitor training on Leonhard
Connect to Leonhard to access files on your local machine. (On ubuntu: in file manager go to 'Other Locations' and connect to server using `sftp://<nethzname>@login.leonhard.ethz.ch`)
Then run in a local terminal:
```
cd ./cluster/home/<nethzname>/cil-road-segmentation-2020/
tensorboard --logdir=logs
```
It's a good idea to delete the logs folder (or rename in case you want to keep the logs of a different run) before submitting your job on Leonhard.



## TODOs
* Decide on which metrics we keep track of. We can then use them to compare models and list them in the report
  - The metric from kaggle? (= Accuracy patchwise?)
  - Accuracy (pixelwise)
  - F1-score
  - IoU
  - Cross Entropy
  - Precision?
  - Recall?
* Start with the list of results. Decide on what we want to keep track of:
  - predicted images?
  - TensorBorad logs?
  - kaggle leader board: public score
  - training/validation scores of the metrics mentioned above
  - the exact .py file that was used for that run, such that we can easily reproduce the scores?

## Some ideas for Fully Convolutional Networks (FCN)

### Preprocessing?
* Not sure, maybe some sort of Edge Detection as a 4. channel?
  - Canny Edge Detection: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
  - Hough Line Transform (could be really cool, but the lines I got did sometimes not align with the roads :( The sudoku example looked promising...): https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
* something else?

### Data Augmentation!
* random horizontal flip  
* random rotations
* randomly changing hue
* randomly changing brightness
* randomly changing contrast
* random crop
  - ~~TODO: so far: fixed size. It is not clear which size is optimal (trade-off: small crop: many different looking training images BUT might be zoomed in too far...) Maybe use random crop size instead?~~
  - BETTER: independent of whether you crop or not: make sure that your predict on patches that are of the same size as the crops/full images that you use in training.
* there are also many more augmentation techniques such as projective transforms, elastic deformations, adding noise, etc. But we have to be careful which ones we use (e.g. shearing might decrease accuracy since the 2 borders of roads are usually parallel and have a constant distance between eachother)
albumentations: https://github.com/albumentations-team/albumentations

### Network
* Basic parameters
  - Parameters
    - number of filters?
    - number of layers?
  - Normalisation and regularisation...
    - Batch Normalization?
  - Activation functions
    - (ReLU, Leaky ReLU, ELU SELU, Swish, ...)

* Architecture and strategy... in increasing complexity
    -
    - decoder + encoder (DE): Segnet
    - DE + skip connections: UNet, segnet, etc
    - DE + skip connections ++
    - As generative problem (with discriminator as loss function)

* Loss functions

### Cost function
* (Binary) Cross Entropy
* Balanced Cross Entropy
* Weighted Boundary loss
* Soft Dice loss
* Lovasz loss https://github.com/bermanmaxim/LovaszSoftmax https://arxiv.org/abs/1705.08790
* Tversky loss
* Focal loss  
* Combination of the above

Really good blog post:
https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/


### References
* U-Net
https://arxiv.org/pdf/1505.04597.pdf
* UNet++
https://arxiv.org/abs/1807.10165
https://github.com/MrGiovanni/UNetPlusPlus
* UNet3+
https://ieeexplore.ieee.org/abstract/document/9053405
* Attention U-Net
https://arxiv.org/abs/1804.03999
* U-Net with dilated convolutions in the bottelneck layers:
https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199
https://github.com/lyakaap/Kaggle-Carvana-3rd-place-solution
* Residual U-Net
https://arxiv.org/abs/1711.10684
* ANU-Net
https://www.sciencedirect.com/science/article/pii/S0097849320300546
* Something else than U-Net? Tiramisu? Seg-Net? Link-Net? simpler FCN?
https://openaccess.thecvf.com/content_cvpr_2017_workshops/w13/html/Jegou_The_One_Hundred_CVPR_2017_paper.html  
https://github.com/qubvel/segmentation_models

### Multitask Learning?
* Distance Transform
https://arxiv.org/abs/1709.05932

### Regularization
* Dropout layers
* Kernel regularization?


### Optimizers & Training
* Adam
* SGD with momentum
* RMSprop
* learning rate??
* batch size??
* learning rate reduce on plateau?
* learning rate scheduling?
* Early Stopping?
* Number of epochs?

### Test Time Augmentation
* predict on rot0, rot90, rot180, rot270, hflip_rot0, hflip_rot90, hflip_rot180, hflip_rot270 and then average predictions

### Postprocessing?
* Custom Threshold?
* Using Morphological Operators such as dilation and/or erosion? (https://www.hindawi.com/journals/je/2013/243021/)
* Skeletonization?
* Something else?


### Some general tips and tricks for image segmentation
* https://neptune.ai/blog/image-segmentation-tips-and-tricks-from-kaggle-competitions

## useful links for TensorFlow
TensorFlow 2.0 tutorial: tf.data.Dataset  
https://www.tensorflow.org/tutorials/load_data/images

TensorFlow 2.0 tutorial: data augmentation:  
https://www.tensorflow.org/tutorials/images/data_augmentation

TensorFlow 2.0 tutorial: image segmentation:  
https://www.tensorflow.org/tutorials/images/segmentation

Segmentation of Roads in Aerial Images:  
https://towardsdatascience.com/road-segmentation-727fb41c51af

Learn about Fully Convolutional Networks for semantic segmentation:  
https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.html
