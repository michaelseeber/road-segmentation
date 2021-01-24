# THIS FILE CONSTRUCTS A U-Net like network with dilated convolutions in the bottleneck, uses BaclancedCE + Lovasz loss, data augmentation, test time augmentation, prediction on 25 images patches

import sys
sys.path.append('../')
from helpers.metrics import *
from helpers.augment import *
from helpers.display import *
from helpers.paths import *
from helpers.submission import *
from helpers.postprocessing import *
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
import os
import argparse
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import tensorflow_datasets as tfds


AUTOTUNE = tf.data.experimental.AUTOTUNE

# Adjust!
NUM_VALIDATION_IMAGES = 25
NUM_THRESHOLD_IMAGES = 6

TRAIN_LENGTH = 128
EPOCHS = 1
#EPOCHS = 500
BATCH_SIZE = 32

# Probably not changing 
BUFFER_SIZE = 200
OUTPUT_CHANNELS = 3
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = STEPS_PER_EPOCH


def main(retrain=True):
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
    train =    list_ds.skip(NUM_VALIDATION_IMAGES).skip(NUM_THRESHOLD_IMAGES).map(lambda x: process_path(x, "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE)
    train_dataset = train.cache().shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_dataset = list_ds.take(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE).map(augment_validation_with_rand_crop).batch(BATCH_SIZE).repeat()

    test_dataset = test_list_ds.map(test_process_path)

    #### Create FCN model ####

    def loss(y_true, y_pred):
        return BCE_and_lovasz_hinge(y_true, y_pred, batch_size=BATCH_SIZE, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    accuracy = tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.0
    )

    if retrain: 
        from tensorflow.keras import layers
        
        inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, 3))
        
        conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (inputs)
        conv1 = tf.keras.layers.BatchNormalization() (conv1)
        conv1 = tf.keras.layers.Dropout(0.1) (conv1)
        conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv1)
        conv1 = tf.keras.layers.BatchNormalization() (conv1)
        pooling1 = tf.keras.layers.MaxPooling2D((2, 2)) (conv1)
        
        conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (pooling1)
        conv2 = tf.keras.layers.BatchNormalization() (conv2)
        conv2 = tf.keras.layers.Dropout(0.1) (conv2)
        conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv2)
        conv2 = tf.keras.layers.BatchNormalization() (conv2)
        pooling2 = tf.keras.layers.MaxPooling2D((2, 2)) (conv2)
        
        conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (pooling2)
        conv3 = tf.keras.layers.BatchNormalization() (conv3)
        conv3 = tf.keras.layers.Dropout(0.2) (conv3)
        conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv3)
        conv3 = tf.keras.layers.BatchNormalization() (conv3)
        pooling3 = tf.keras.layers.MaxPooling2D((2, 2)) (conv3)
        
        conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (pooling3)
        conv4 = tf.keras.layers.BatchNormalization() (conv4)
        conv4 = tf.keras.layers.Dropout(0.2) (conv4)
        conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv4)
        conv4 = tf.keras.layers.BatchNormalization() (conv4)
        pooling4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (conv4)
        
        #conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (pooling4)
        #conv5 = tf.keras.layers.BatchNormalization() (conv5)
        #conv5 = tf.keras.layers.Dropout(0.3) (conv5)
        #conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (conv5)
        #conv5 = tf.keras.layers.BatchNormalization() (conv5)
        conv5_1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=1, padding='same', kernel_regularizer=regularizers.l2(0.001)) (pooling4)
        conv5_2 = tf.keras.layers.BatchNormalization() (conv5_1)
        conv5_2 = tf.keras.layers.Dropout(0.1) (conv5_2)
        conv5_2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=2, padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv5_2)
        conv5_4 = tf.keras.layers.BatchNormalization() (conv5_2)
        conv5_4 = tf.keras.layers.Dropout(0.1) (conv5_4)
        conv5_4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=4, padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv5_4)
        conv5_8 = tf.keras.layers.BatchNormalization() (conv5_4)
        conv5_8 = tf.keras.layers.Dropout(0.1) (conv5_8)
        conv5_8 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=8, padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv5_8)
        conv5_16 = tf.keras.layers.BatchNormalization() (conv5_8)
        conv5_16 = tf.keras.layers.Dropout(0.1) (conv5_16)
        conv5_16 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=16, padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv5_16)
        conv5_32 = tf.keras.layers.BatchNormalization() (conv5_16)
        conv5_32 = tf.keras.layers.Dropout(0.1) (conv5_32)
        conv5_32 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', dilation_rate=32, padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv5_32)
        conv5_sum = tf.keras.layers.Add() ([conv5_1, conv5_2, conv5_4, conv5_8, conv5_16, conv5_32])
        
        #upsample6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
        upsample6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5_sum)
        upsample6 = tf.keras.layers.concatenate([upsample6, conv4])
        conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (upsample6)
        conv6 = tf.keras.layers.BatchNormalization() (conv6)
        #conv6 = tf.keras.layers.Dropout(0.2) (conv6)
        conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv6)
        conv6 = tf.keras.layers.BatchNormalization() (conv6)
        
        upsample7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
        upsample7 = tf.keras.layers.concatenate([upsample7, conv3])
        conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (upsample7)
        conv7 = tf.keras.layers.BatchNormalization() (conv7)
        #conv7 = tf.keras.layers.Dropout(0.2) (conv7)
        conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv7)
        conv7 = tf.keras.layers.BatchNormalization() (conv7)
        
        upsample8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
        upsample8 = tf.keras.layers.concatenate([upsample8, conv2])
        conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (upsample8)
        conv8 = tf.keras.layers.BatchNormalization() (conv8)
        #conv8 = tf.keras.layers.Dropout(0.1) (conv8)
        conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv8)
        conv8 = tf.keras.layers.BatchNormalization() (conv8)
        
        upsample9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv8)
        upsample9 = tf.keras.layers.concatenate([upsample9, conv1], axis=3)
        conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (upsample9)
        conv9 = tf.keras.layers.BatchNormalization() (conv9)
        #conv9 = tf.keras.layers.Dropout(0.1) (conv9)
        conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)) (conv9)
        conv9 = tf.keras.layers.BatchNormalization() (conv9)
        
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation=None) (conv9)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        
        model.summary()

     

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH*1000,
            decay_rate=1,
            staircase=False)

        def get_optimizer():
            return tf.keras.optimizers.Adam(lr_schedule)
        model.compile(optimizer=get_optimizer(),
                    loss=loss,
                    metrics=[accuracy, Precision_with_logits(), Recall_with_logits(), meanIoU_with_logits(num_classes=2)])

        model_history = model.fit(train_dataset,
                                epochs=EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_steps=VALIDATION_STEPS,
                                validation_data=validation_dataset,
                                callbacks=[tf.keras.callbacks.TensorBoard(),
                                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=200)
                                ])

        plot_model_history(model_history, save=True, show=False)
        Path('saved_models/').mkdir(parents=True, exist_ok=True)
        model.save('saved_models/model_' + datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm') + '.h5')
    else:
        model = tf.keras.models.load_model('saved_models/saved.h5', custom_objects={'loss': loss, 'Precision_with_logits': Precision_with_logits(),
        'Recall_with_logits': Recall_with_logits(), 'meanIoU_with_logits': meanIoU_with_logits(num_classes=2)})


    # write files to folder
    path1 = '../thresholds/closing/'

    ds =  list_ds.skip(NUM_VALIDATION_IMAGES).take(NUM_THRESHOLD_IMAGES)
    threshold_dataset = ds.map(lambda x: tf.py_function(apply_threshold(x)), num_parallel_calls=AUTOTUNE)#ds.map(apply_threshold, num_parallel_calls=AUTOTUNE).map(augment_validation_with_rand_crop).batch(BATCH_SIZE).repeat()
    res = model.evaluate(threshold_dataset, steps=STEPS_PER_EPOCH, verbose=1)

# def apply_threshold(img, label):
#     # file_path_mask = tf.strings.regex_replace(file_path, "images_rotated", "groundtruth_rotated")
#     # mask = tf.io.read_file(file_path_mask)
#     # mask = decode_mask(mask)
#     # print(file_path)
#     # img = cv2.imread(file_path)
#     # # load the raw data from the file as a string
#     # # img = tf.io.read_file(file_path)
#     # # img = decode_img(img)
#     # print(img)

#     # img = tf.keras.preprocessing.image.img_to_array(img)
#     # # print(img)
#     # #img = morph_closing(img)
#     # print(img)
#     print(img)
#     print(label)
#     img_new = tfds.as_numpy(img)
#     print(img_new)
#     img2 = morph_closing(img_new, square(15))
# #     return  img2, label
def apply_threshold(file_path):
    file_path_mask = tf.strings.regex_replace(file_path, "images_rotated", "groundtruth_rotated")
    mask = tf.io.read_file(file_path_mask)
    mask = decode_mask(mask)
    #img = cv2.imread(tf.strings.as_string(file_path))
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    print(img)
    img = decode_img(img)
    img = morph_closing(img)
    # img = closing(img.numpy(), square(15))
    return  img, mask

      
if __name__ == '__main__':
    main(retrain=False)
