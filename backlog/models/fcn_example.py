### THIS IS AN EXAMPLE FILE ON HOW TO USE THE EXISTING CODE BASE AND
#   ON HOW TO IDEALLY STRUCTURE THE CODE, FOLDERS, AND SUBMISSIONS
#   
#   1. COPY-PASTE THE IMPORTS OF THIS FILE, WHICH GIVES ACCESS TO HELPER METHODS ETC.
#   2. TRY TO USE THE PROVIDED HELPER FUNCTIONS FOR THE MOST COMMON METHODS
#      IF NEEDED, ADD NEW METHODS (E.G. METRICS)
#   3. WHEN SUBMITTING A FILE TO KAGGLE, NAME THE CSV FILE (AND THE CORRESPONDING PYTHON AND TENSORBOARD FILE)  
#      "SUBMISSION_YYYY_MM_DD_HH__MM" SUCH AS "SUBMISSION_2020_06_07_20_13.CSV / .PY / .TENSORBOARD"
#       AND ADD THEM TO THE KAGGLE FOLDER
#
#   4. PUT COMMENTS ON TOP OF EACH FILE THAT EXPLAINS REAL QUICK WHAT THIS FILE DOES AND THE SCORES
#        OR WHAT WE COULD DO AS A FURTHER STEP
###


# THIS FILE CONSTRUCTS A SIMPLE FCN USING DATA AUGMENTATION. IT USES ACCURACY AS A METRIC AND SOFT DICE LOSS
# .......

import sys
sys.path.append('../')
from helpers.metrics import *
from helpers.augment import *
from helpers.display import *
from helpers.paths import *
from helpers.submission import *
import tensorflow as tf
import numpy as np
import os
import argparse

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Adjust!
NUM_VALIDATION_IMAGES = 16
TRAIN_LENGTH = 128
EPOCHS = 2
BATCH_SIZE = 16

# Probably not changing 
BUFFER_SIZE = 200
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

    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    training_images_path = '../training/images/'
    test_images_path = '../testing/images/'
    list_ds = tf.data.Dataset.list_files(training_images_path + '*')
    test_list_ds = tf.data.Dataset.list_files(test_images_path + '*', shuffle=False)
    train =    list_ds.skip(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images", "groundtruth"), num_parallel_calls=AUTOTUNE)
    train_dataset = train.shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_dataset = list_ds.take(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images", "groundtruth"), num_parallel_calls=AUTOTUNE).map(resize_validation).map(augment_validation, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
    test_dataset = test_list_ds.map(test_process_path).map(resize_test).batch(1)

    for image, mask in train.take(1):
      sample_image, sample_mask = image, mask

    #### Create FCN model ####

    from tensorflow.keras import layers
    
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(filters=8, kernel_size=4, strides=4, padding='same'))
    model.add(layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid'))

    model.summary()

    #### Train FCN model ####

    model.compile(optimizer='adam',
                  loss=soft_dice_loss,
                  metrics=[f1, bce_loss, soft_dice_loss, 'accuracy'])
    model_history = model.fit(train_dataset,
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=validation_dataset,
                              callbacks=[tf.keras.callbacks.TensorBoard()])
    plot_model_history(model_history, save=True, show=False)

    #### If you want to show the prediction images, uncomment this line ####
    # show_predictions(train_dataset, 1)

    predictions = model.predict(test_dataset, verbose=1)
    filenames = sorted(os.listdir(test_images_path))
    Path("../testing/predictions").mkdir(parents=True, exist_ok=True)

    #### Save the predicted images to the folder testing/predictions ####
    for i, prediction in enumerate(predictions):
      prediction = tf.image.resize(prediction, [608, 608])
      prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
      img_prediction = tf.image.encode_png(prediction)
      number = filenames[i]
      tf.io.write_file("../testing/predictions/test_" + str(number[5:8]) + ".png", img_prediction)
      
    #### If you want to show the predicted images on the testset, uncomment this line ####
    # test_show_predictions(test_dataset, 10)
    submit_predictions(filenames)
      
if __name__ == '__main__':
    main()
