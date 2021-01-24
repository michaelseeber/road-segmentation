import sys
sys.path.append('../')
import pandas as pd
from helpers.display import *
from helpers.paths import *
import cv2
import os
import glob
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


def main():

    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    images_path = '../training/images/'
    groundtruth_path = '../training/groundtruth/'

    images_list = [cv2.imread(file) for file in sorted(
        glob.glob(images_path + '*.png'))]
    filenames = sorted([file for file in os.listdir(
        images_path) if file.endswith('.png')])

    groundtruth_list = [cv2.imread(file) for file in sorted(
        glob.glob(groundtruth_path + '*.png'))]

    results = pd.DataFrame({ 'image_name': filenames })

    results['percentage_road'] = percentage_of_roads_for_image_mask(groundtruth_list)
    
    rgb = average_rgb_color(images_list)
    results['mean_rgb_r'] = rgb[0]
    results['mean_rgb_g'] = rgb[1]
    results['mean_rgb_b'] = rgb[2]
    results['mean_rgb_r'] = results['mean_rgb_r'].astype('int') 
    results['mean_rgb_g'] = results['mean_rgb_g'].astype('int') 
    results['mean_rgb_b'] = results['mean_rgb_b'].astype('int') 

    print(results)


def percentage_of_roads_for_image_mask(images):
    values = [None] * len(images)
    for i, img in enumerate(images):
        height, width = img.shape
        cntPixels = height*width
        n_black_pixels = np.sum(img == 0)
        values[i] = (cntPixels - n_black_pixels) / cntPixels * 100

    
    return values

def average_rgb_color(images):
    values_r = [None] * len(images)
    values_g = [None] * len(images)
    values_b = [None] * len(images)

    # cv2 imread uses BGR instead of RGB
    for i, img in enumerate(images):
        avg_color = np.mean(img, axis=(0, 1))
        values_b[i] = round(avg_color[0])
        values_g[i] = round(avg_color[1])
        values_r[i] = round(avg_color[2])

    return (values_r, values_g, values_b)

if __name__ == '__main__':
    main()
