from pathlib import Path
import datetime 
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('../')

from helpers.mask_to_submission import *
from helpers.submission_to_mask import *

#### Applies the "mask_to_submission file that converts the predicted images to our output format 
#   Output format: Each image is split into patches of 16 x 16 pixels, and then a 0 or 1 label is assigned to it 
#   based on our predicted pixel-wise label
#   The public test score is based on those patch-wise predictions 
#  ####

def submit_predictions(filenames, path='../testing/predictions/'):
  Path("../results/csv").mkdir(parents=True, exist_ok=True)
  submission_filename = '../results/csv/submission_' + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M') + '.csv'
  image_filenames = []
  print(filenames)
  for i in range(0, 94):
    number = filenames[i]
    filename = path + 'test_' + str(number[5:8]) + ".png"
    print(filename)
    if not os.path.isfile(filename):
        print(filename + " not found")
        continue
    image_filenames.append(filename)
    
  masks_to_submission(submission_filename, *image_filenames)
  return submission_filename

  # reconstructed = reconstruct_from_labels(3, submission_filename)
  # print(reconstructed)
  # plt.imshow(reconstructed)
  # plt.show()