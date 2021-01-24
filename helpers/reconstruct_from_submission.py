#!/usr/bin/python
import os
import sys
from PIL import Image
import math
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
#label_file = 'dummy_submission.csv'
#label_file = '../submissions/submission_2020-07-03 10:42.csv'

#baseline_fcn.py (dani)
#label_file = '../results/csv/submission_2020-07-09_10:22.csv'

#baseline_fcn.py (basti)
#label_file = '../results/csv/submission_2020-07-05_1657.csv'

#(mark)
#label_file = '../results/csv/submission_2020-07-25-00-58.csv'

#snapshot ensemble (dani)
#label_file = '../results/csv/submission_2020-07-25_08:28.csv'

#0.92264
#label_file = '../results/csv/submission_2020-07-26_17_43.csv'

#random erasure, BCE 2000 epochs
#label_file = '../results/csv/submission_2020-07-27_03:59.csv'

#0.91980
#label_file = '../results/csv/submission_2020-07-27_21_48_threshold0.4.csv'

#baseline
# label_file = '../results/csv/submission_baseline.csv'

#best
# label_file = '../results/csv/submission_2020-07-26_17_43_0.35threshold-BEST.csv'
#final model
label_file = '../results/csv/submission_2020-07-31_10_09_threshold0.35.csv'

h = 16
w = h
imgwidth = int(math.ceil((600.0/w))*w)
imgheight = int(math.ceil((600.0/h))*h)
nc = 3

# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def reconstruct_from_labels(image_id, name = label_file):
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(name)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    image_id_found = False
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue
        image_id_found = True

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)
    path = '../testing/predictions_submission/'
    Path(path).mkdir(parents=True, exist_ok=True)

    if image_id_found:
        Image.fromarray(im).save(path + 'test_' + '%.3d' % image_id + '.png')

    return im

def main():
    for i in range(7, 224):
        reconstruct_from_labels(i)
   
if __name__ == '__main__':
    main()


