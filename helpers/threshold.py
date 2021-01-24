import os
from pathlib import Path
import datetime 
import matplotlib.pyplot as plt
from skimage import io, util, filters
from skimage.filters import *
import numpy as np
import sys
sys.path.append('../')
from helpers.submission import masks_to_submission


def th_otsu(im):
    t = threshold_otsu(im)
    otsu = im > t
    return  (np.clip(otsu*255, 0, 255)).astype(np.uint8)

def th_minimum(im):
    t = threshold_minimum(im)
    minimum = im > t
    return  (np.clip(minimum*255, 0, 255)).astype(np.uint8)

def th_li(im):
    t = threshold_li(im) + 0.07 * 255
    li = im > t
    return  (np.clip(li*255, 0, 255)).astype(np.uint8)

def th_manual(im, t):
    manual = im > t * 255
    return  (np.clip(manual*255, 0, 255)).astype(np.uint8)

def plot_different_thresholds():

    #load predicted images
    input_folder = '../testing/predictions'
    input_files = [os.path.join(root, filename)
            for root, dirs, files in os.walk(input_folder)
            for filename in files
            if filename.lower().endswith('.png')]
    images = []
    for input_file in input_files:
        images.append(io.imread(input_file))

    # load satellite images for comaparison
    satellite_path = '../testing/images/'
    satellite_input_files = [os.path.join(root, filename)
            for root, dirs, files in os.walk(satellite_path)
            for filename in files
            if filename.lower().endswith('.png')]
    satellite_images = []
    for input_file in satellite_input_files:
        satellite_images.append(io.imread(input_file))

    output_folder = '../testing/predictions_threshold_test/'
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    output_files = [os.path.join(output_folder, filename)
            for root, dirs, files in os.walk(input_folder)
            for filename in files
            if filename.lower().endswith('.png')]
    for idx, image in enumerate(images):
        fig, ax = try_all_threshold(image, figsize=(10, 6), verbose=False)
        plt.savefig(output_folder + '_img_' + str(idx)  + '.png')
        plt.clf()




# def submit_predictions(filenames, path='../testing/predictions_threshold/'):
#   Path("../results/csv").mkdir(parents=True, exist_ok=True)
#   submission_filename = '../results/csv/submission_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M') + '.csv'
#   image_filenames = []
#   for i in range(0, 94):
#     number = filenames[i]
#     filename = path + 'test_' + str(number[38:41]) + ".png"
#     if not os.path.isfile(filename):
#         print(filename + " not found")
#         continue
#     image_filenames.append(filename)
    
#   masks_to_submission(submission_filename, *image_filenames)

def normal_thresholding():


    #load predicted images
    input_folder = '../testing/predictions'
    input_files = [os.path.join(root, filename)
            for root, dirs, files in os.walk(input_folder)
            for filename in files
            if filename.lower().endswith('.png')]
    images = []
    for input_file in input_files:
        images.append(io.imread(input_file))

    # load satellite iamges for comaparison
    satellite_path = '../testing/images/'
    satellite_input_files = [os.path.join(root, filename)
            for root, dirs, files in os.walk(satellite_path)
            for filename in files
            if filename.lower().endswith('.png')]
    satellite_images = []
    for input_file in satellite_input_files:
        satellite_images.append(io.imread(input_file))

    #apply threshold
    output_folder = '../testing/predictions_threshold'
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    output_files = [os.path.join(output_folder, filename)
            for root, dirs, files in os.walk(input_folder)
            for filename in files
            if filename.lower().endswith('.png')]
    for idx, image in enumerate(images):
        #threshold = threshold_minimum(image)
        threshold = 0.375 * 255
        binary = image > threshold
        binary = (np.clip(binary*255, 0, 255)).astype(np.uint8)
        io.imsave(output_files[idx], binary)
        
    submit_predictions(output_files)

