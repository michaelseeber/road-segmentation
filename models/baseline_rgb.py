


import sys
sys.path.append('../')
import numpy as np 
import cv2 
from matplotlib import pyplot as plt
import os
import glob
from pathlib import Path
import tensorflow as tf
from helpers.submission import *
from helpers.threshold import *
from helpers.morph import *
from helpers.postprocessing import plot_comparison
def pixel_threshold(img):
    w, h = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lowerBound = (95, 95, 95);
    upperBound = (120, 120, 120);

    mask = cv2.inRange(img, lowerBound, upperBound)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # morph = morph_closing(mask, filter_size = 4)
    # morph = morph_erosion(morph, filter_size=3)
    # morph = morph_area_opening(morph, area_size=40)

    # plot_comparison([img, mask, morph],
    #                 ['satellite', 'prediction', 'morphed'],
    #                 [None] + [plt.cm.gray] * 20,
    #                 -1,
    #                 'morphed/thresholds_area_300_opening_10_closing_1_area_500/',
    #                 show = True)

    # plt.imshow(mask)
    
    # plt.show()

    return mask

def baseline_color():
    satellite_path = '../testing/images/'
    new_images_path = '../testing/predictions_baseline/'

    satellite_images_names = sorted(os.listdir(satellite_path))
    filenames = sorted(os.listdir(satellite_path))
    satellite_images_list = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(
    glob.glob(satellite_path + '*.png'))]

    Path(new_images_path).mkdir(parents=True, exist_ok=True)

    for i, im in enumerate(satellite_images_list):
            new_im = pixel_threshold(im)
            #new_im = np.expand_dims(new_im, axis=2)
          #   encoded = tf.image.encode_png(new_im)
            name = new_images_path  + "test_" + str(filenames[i][5:8]) + ".png"
            cv2.imwrite(name, new_im)
            print("Written to " + new_images_path + "test_" + str(filenames[i][5:8]) + ".png")

    new_filenames = os.listdir(new_images_path)
    if '.DS_Store' in new_filenames:
        new_filenames.remove('.DS_Store')
    submit_predictions(new_filenames, path = new_images_path)


baseline_color()



