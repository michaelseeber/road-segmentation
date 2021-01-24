# THIS FILE CONSTRUCTS A smaller U-Net like network

import sys
sys.path.append('../')
from helpers.metrics import *
from helpers.display import *
from helpers.paths import *
from helpers.augment import *
from helpers.predict_on_patches import *
from helpers.submission import *
import tensorflow as tf
import numpy as np
import os
import argparse

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Adjust!
NUM_VALIDATION_IMAGES = 6 * 6
TRAIN_LENGTH = 576
EPOCHS = 2000
BATCH_SIZE = 32

# Probably not changing 
BUFFER_SIZE = 600
OUTPUT_CHANNELS = 3
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = STEPS_PER_EPOCH

def main():
    global NUM_VALIDATION_IMAGES, TRAIN_LENGTH, EPOCHS, BATCH_SIZE, BUFFER_SIZE, OUTPUT_CHANNELS, STEPS_PER_EPOCH, VALIDATION_STEPS,IMG_WIDTH, IMG_HEIGHT
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="Number of epochs we want", type=int)
    args = parser.parse_args()
    if args.epochs:
      EPOCHS = args.epochs

    training_images_path = '../training/images_rotated/'
    test_images_path = '../testing/images/'
    list_ds = tf.data.Dataset.list_files(training_images_path + '*')
    test_list_ds = tf.data.Dataset.list_files(test_images_path + '*', shuffle=False)
    train =    list_ds.skip(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE)
    train_dataset = train.cache().shuffle(BUFFER_SIZE).map(lambda img, label: augment(img, label, horizontal_flip=True, random_crop=True, rotate90=True, color=True, block_noise=False), num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_dataset = list_ds.take(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE).map(lambda img, label: augment(img, label, color=False)).batch(1).repeat()
    test_dataset = test_list_ds.map(test_process_path)

    from matplotlib import pyplot as plt

    model = tf.keras.models.load_model('saved_models/model.h5', compile=False)
    model_erasure = tf.keras.models.load_model('saved_models/model_2020-07-29_20:08.h5', compile=False)

    images = []
    predictions = []
    predictions_erasure = []
    groundtruths = []

    for _ in range(10):
      for img, gt in validation_dataset.take(1):
        images.append(tf.squeeze(img))
        predictions.append(tf.squeeze(model.predict(img, verbose=1)))
        predictions_erasure.append(tf.squeeze(model_erasure.predict(img, verbose=1)))
        groundtruths.append(tf.squeeze(gt))

    plt.figure(figsize=(25, 25))
    for i in range(5):
      plt.subplot(4, 5, i+1)
      plt.imshow(images[i], cmap="gray" )
      plt.axis('off')
      plt.subplot(4, 5, 5+i+1)
      # plt.title(title[i])
      plt.imshow(groundtruths[i], cmap="gray" )
      plt.axis('off')
      plt.subplot(4, 5, 10+i+1)
      # plt.title(title[i])
      plt.imshow(predictions[i], cmap="gray" )
      plt.axis('off')
      plt.subplot(4, 5, 15+i+1)
      # plt.title(title[i])
      plt.imshow(predictions_erasure[i], cmap="gray" )
      plt.axis('off')
    plt.show()
    #plt.savefig("result.png")

    
    # #### If you want to show the prediction images, uncomment this line ####
    # # show_predictions(train_dataset, 1)

    # predictions = predict_on_patches5x5_with_probabilities(model, test_dataset)
    # filenames = sorted(os.listdir(test_images_path))
    # Path("../testing/predictions").mkdir(parents=True, exist_ok=True)

    # #### Save the predicted images to the folder testing/predictions ####
    # for i, prediction in enumerate(predictions):
    #   prediction = tf.image.resize(prediction, [608, 608])
    #   prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
    #   img_prediction = tf.image.encode_png(prediction)
    #   number = filenames[i]
    #   tf.io.write_file("../testing/predictions/test_" + str(number[5:8]) + ".png", img_prediction)
      
    # #### If you want to show the predicted images on the testset, uncomment this line ####
    # # test_show_predictions(test_dataset, 10)
    # submit_predictions(filenames)
      
if __name__ == '__main__':
    main()
