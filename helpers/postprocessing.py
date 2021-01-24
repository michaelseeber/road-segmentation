import sys
sys.path.append('../')
import cv2
import numpy as np
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from helpers.submission_to_mask import *
from helpers.submission import *
import glob
import tensorflow as tf
from pathlib import Path
from skimage import io, util
from helpers.paths import *
from helpers.metrics import *
from helpers.augment import *
from helpers.threshold import *
from helpers.morph import *
from helpers.reconstruct_from_submission import reconstruct_from_labels
FILTER_SIZE = 50
CLOSING_FILTER_SIZE = 20
DILATED_FILTER_SIZE = 10

OPENING_FILTER_SIZE = 15
AREA_SIZE = 900
AUTOTUNE = tf.data.experimental.AUTOTUNE

def plot_comparison(im_list, name_list, cmap_list, save_plot_image_index = -1, save_plot_path = 'image', show = True):
      fig, axes = plt.subplots(ncols=len(im_list), figsize=(3 * len(im_list), 4), sharex=True,
                                    sharey=True)
      for i, el in enumerate(axes):
            el.imshow(im_list[i], cmap=cmap_list[i])
            el.set_title(name_list[i])
            el.axis('off')

      if (save_plot_image_index != -1):
            Path(save_plot_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_plot_path + 'test_morphed_' + str(save_plot_image_index) + '.png')
      if show:
            plt.show()

      plt.close(fig)

def plot_image(im):
      plt.imshow(im)
      plt.show()

def postprocess_and_submit(redo=True, predictions_path='../testing/predictions/'):
      satellite_path = '../testing/images/'
      output_path = '../testing/postprocessed/'
      pred_thresh_path = '../testing/predictions_threshold/'

      satellite_images_names = sorted(os.listdir(satellite_path))
      filenames = sorted(os.listdir(predictions_path))
      if '.DS_Store' in filenames:
            filenames.remove('.DS_Store')
      satellite_images_list = [cv2.imread(file) for file in sorted(
        glob.glob(satellite_path + '*.png'))]

      predictions_list = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(
        glob.glob(predictions_path + '*.png'))]

      pred_threshold_list = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(
        glob.glob(pred_thresh_path + '*.png'))]
        
      Path(output_path).mkdir(parents=True, exist_ok=True)
      manual_t = 0.3

      manual_images = []
      li_images = []
      otsu_images = []

      final_images = []
      final_images_submitted = []
      if redo:
            for i, im in enumerate(predictions_list):
                  # 1. Do thresholding
                  #im_otsu = th_otsu(im)
                  im_li = th_li(im)
                  li_images.append(im_li)

                 # otsu_images.append(im_otsu)

                  # mask1 = morph_closing(im, filter_size = 16)
                  # # Remove everything that now still has big area ->
                  # mask2 = morph_area_opening(mask1, area_size = 256*3+1)

                  # #
                  # # diff = mask2 - mask1
                  # # diff[diff!=1] = 0
                  # # diff[diff==1] = 1

                  # # first = morph_closing(im, filter_size = 15)
                  # first[diff==1] = 0 
                  
                  im_manual = th_manual(im, manual_t)
                  manual_images.append(im_manual)

                  first = morph_closing(im_li, filter_size = 6)
                  first = morph_area_opening(first, area_size = 400)
                  new_im = first                

                  new_im = np.expand_dims(new_im, axis=2)
                  name = output_path  + "test_" + str(filenames[i][5:8]) + ".png"
                  cv2.imwrite(name, new_im)
                  final_images.append(new_im)
                  print("Written to " + name)

      new_filenames = os.listdir(output_path)
      if '.DS_Store' in new_filenames:
            new_filenames.remove('.DS_Store')
      csv_filename = submit_predictions(new_filenames, path = output_path)
      for i in range(7, 224):
            reconstruct_from_labels(i, name = csv_filename)

      name_list = range(7, 224)
      for i, im in enumerate(satellite_images_list):
            image_id = int(filenames[i][5:8])
            plot_comparison([im,predictions_list[i], manual_images[i], li_images[i],  final_images[i], reconstruct_from_labels(image_id, name= csv_filename)],
            ['Satellite', 'Prediction', 't=0.3', 'Li method', 'After morphing', 'Submission',],
            [None] + [plt.cm.gray] * 20,
            i,
            'postprocessed/li_close_area_sub0.35/',
            show = False)
    

def postprocess_and_submit_final(redo=True, predictions_path='../testing/predictions_pixelated/'):
      satellite_path = '../testing/images/'
      output_path = '../testing/postprocessed/'
      pred_thresh_path = '../testing/predictions_threshold/'

      satellite_images_names = sorted(os.listdir(satellite_path))
      filenames = sorted(os.listdir(predictions_path))
      if '.DS_Store' in filenames:
            filenames.remove('.DS_Store')
      satellite_images_list = [cv2.imread(file) for file in sorted(
        glob.glob(satellite_path + '*.png'))]

      predictions_list = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(
        glob.glob(predictions_path + '*.png'))]

      pred_threshold_list = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(
        glob.glob(pred_thresh_path + '*.png'))]
        
      Path(output_path).mkdir(parents=True, exist_ok=True)
      manual_t = 0.3

      thresholded = []
      li_images = []
      final_images = []
      final_images_submitted = []
      if redo:
            for i, im in enumerate(predictions_list):
                  # # 1. Do thresholding
                  # im_otsu = th_otsu(im)
                  # im_li = th_li(im)

                  # li_images.append(im_li)

                  # mask1 = morph_closing(im, filter_size = 16)
                  # # Remove everything that now still has big area ->
                  # mask2 = morph_area_opening(mask1, area_size = 256*3+1)

                  
                  # diff = mask2 - mask1
                  # diff[diff!=1] = 0
                  # diff[diff==1] = 1

                  first = morph_area_opening(im, area_size = 256*2)
                  # first[diff==1] = 0 
                  
                  new_im = first                

                  new_im = np.expand_dims(new_im, axis=2)
                  name = output_path  + "test_" + str(filenames[i][5:8]) + ".png"
                  cv2.imwrite(name, new_im)
                  final_images.append(new_im)
                  print("Written to " + name)
      new_filenames = sorted(os.listdir(output_path))

      if '.DS_Store' in new_filenames:
            new_filenames.remove('.DS_Store')
      csv_filename = submit_predictions(new_filenames, path = output_path)
      # for i, im in enumerate(satellite_images_list):
      #       image_id = int(filenames[i][5:8])
            # print(image_id)
            # plot_comparison([im,predictions_list[i], final_images[i]],
            # ['Satellite', 'Prediction', 'final'],
            # [None] + [plt.cm.gray] * 20,
            # i,
            # 'postprocessed/_area_opening_on_final_35/',
            # show = True)


def main():
      postprocess_and_submit(redo=True)

if __name__ == '__main__':
    main()
