# THIS FILE CONSTRUCTS A U-Net like network, uses dice loss and no augmentation

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
NUM_VALIDATION_IMAGES = 6 * 6
TRAIN_LENGTH = 128
EPOCHS = 500
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

    training_images_path = '../training/images_rotated/'
    test_images_path = '../testing/images/'
    list_ds = tf.data.Dataset.list_files(training_images_path + '*')
    test_list_ds = tf.data.Dataset.list_files(test_images_path + '*', shuffle=False)
    train =    list_ds.skip(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE)
    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_dataset = list_ds.take(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE).map(resize_validation).batch(BATCH_SIZE).repeat()
    test_dataset = test_list_ds.map(test_process_path).map(resize_test).batch(1)

    #### Create FCN model ####

    from tensorflow.keras import layers
    
    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, 3))
    
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (inputs)
    conv1 = tf.keras.layers.BatchNormalization() (conv1)
    conv1 = tf.keras.layers.Dropout(0.1) (conv1)
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (conv1)
    conv1 = tf.keras.layers.BatchNormalization() (conv1)
    pooling1 = tf.keras.layers.MaxPooling2D((2, 2)) (conv1)
    
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (pooling1)
    conv2 = tf.keras.layers.BatchNormalization() (conv2)
    conv2 = tf.keras.layers.Dropout(0.1) (conv2)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (conv2)
    conv2 = tf.keras.layers.BatchNormalization() (conv2)
    pooling2 = tf.keras.layers.MaxPooling2D((2, 2)) (conv2)
    
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (pooling2)
    conv3 = tf.keras.layers.BatchNormalization() (conv3)
    conv3 = tf.keras.layers.Dropout(0.2) (conv3)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (conv3)
    conv3 = tf.keras.layers.BatchNormalization() (conv3)
    pooling3 = tf.keras.layers.MaxPooling2D((2, 2)) (conv3)
    
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (pooling3)
    conv4 = tf.keras.layers.BatchNormalization() (conv4)
    conv4 = tf.keras.layers.Dropout(0.2) (conv4)
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (conv4)
    conv4 = tf.keras.layers.BatchNormalization() (conv4)
    pooling4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (conv4)
    
    conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same') (pooling4)
    conv5 = tf.keras.layers.BatchNormalization() (conv5)
    conv5 = tf.keras.layers.Dropout(0.3) (conv5)
    conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same') (conv5)
    conv5 = tf.keras.layers.BatchNormalization() (conv5)
    
    upsample6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
    upsample6 = tf.keras.layers.concatenate([upsample6, conv4])
    conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (upsample6)
    conv6 = tf.keras.layers.BatchNormalization() (conv6)
    #conv6 = tf.keras.layers.Dropout(0.2) (conv6)
    conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (conv6)
    conv6 = tf.keras.layers.BatchNormalization() (conv6)
    
    upsample7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
    upsample7 = tf.keras.layers.concatenate([upsample7, conv3])
    conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (upsample7)
    conv7 = tf.keras.layers.BatchNormalization() (conv7)
    #conv7 = tf.keras.layers.Dropout(0.2) (conv7)
    conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (conv7)
    conv7 = tf.keras.layers.BatchNormalization() (conv7)
    
    upsample8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
    upsample8 = tf.keras.layers.concatenate([upsample8, conv2])
    conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (upsample8)
    conv8 = tf.keras.layers.BatchNormalization() (conv8)
    #conv8 = tf.keras.layers.Dropout(0.1) (conv8)
    conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (conv8)
    conv8 = tf.keras.layers.BatchNormalization() (conv8)
    
    upsample9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv8)
    upsample9 = tf.keras.layers.concatenate([upsample9, conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (upsample9)
    conv9 = tf.keras.layers.BatchNormalization() (conv9)
    #conv9 = tf.keras.layers.Dropout(0.1) (conv9)
    conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (conv9)
    conv9 = tf.keras.layers.BatchNormalization() (conv9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv9)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.summary()

    #### Train FCN model ####

    model.compile(optimizer='adam',
                  loss=soft_dice_loss,
                  metrics=['accuracy', meanIoU_with_probabilities(num_classes=2)])

    model_history = model.fit(train_dataset,
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=validation_dataset,
                              callbacks=[tf.keras.callbacks.TensorBoard()])

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
