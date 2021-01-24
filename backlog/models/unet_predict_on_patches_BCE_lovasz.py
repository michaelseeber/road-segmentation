# THIS FILE CONSTRUCTS A U-Net like network, uses lovasz loss, data augmentation, test time augmentation, prediction on 9 images patches

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

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Adjust!
NUM_VALIDATION_IMAGES = 6 * 6
TRAIN_LENGTH = 128
EPOCHS = 20000
BATCH_SIZE = 32

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
    train_dataset = train.cache().shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_dataset = list_ds.take(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE).map(augment_validation_with_rand_crop).batch(BATCH_SIZE).repeat()
    test_dataset = test_list_ds.map(test_process_path)

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
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation=None) (conv9)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.summary()

    #### Train FCN model ####
    accuracy = tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.0
    )

    def loss(y_true, y_pred):
        return BCE_and_lovasz_hinge(y_true, y_pred, batch_size=BATCH_SIZE, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

    model.compile(optimizer='adam',
                  #loss=soft_dice_loss,
                  #loss=lovasz_hinge,
                  loss=loss,
                  #metrics=[accuracy, meanIoU_with_logits(num_classes=2)])
                  #metrics=[accuracy, Precision_with_logits(), Recall_with_logits(), F1Score_with_logits(), meanIoU_with_logits(num_classes=2)])
                  metrics=[accuracy, Precision_with_logits(), Recall_with_logits(), meanIoU_with_logits(num_classes=2)])
                  #metrics=[F1Score_with_logits()])

    model_history = model.fit(train_dataset,
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=validation_dataset,
                              callbacks=[tf.keras.callbacks.TensorBoard()])

    #### If you want to show the prediction images, uncomment this line ####
    # show_predictions(train_dataset, 1)
    filenames = sorted(os.listdir(test_images_path))
    predictions = []
    for idx in range (94):
        predictions.append(tf.zeros([608, 608, 1], tf.float32))
    print("Start: predict on patch [0, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_0).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.sigmoid(prediction_patch)
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [176, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[176, 0], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction = tf.pad(prediction_patch, [[0, 352], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_1).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.sigmoid(prediction_patch)
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [0, 176], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [176, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[176, 0], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction = tf.pad(prediction_patch, [[0, 352], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_2).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.sigmoid(prediction_patch)
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [0, 176], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[176, 0], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction = tf.pad(prediction_patch, [[0, 352], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_3).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.sigmoid(prediction_patch)
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [176, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[0, 176], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[176, 0], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction = tf.pad(prediction_patch, [[176, 176], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_4).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.sigmoid(prediction_patch)
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [0, 176], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [176, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[0, 176], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[176, 0], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction = tf.pad(prediction_patch, [[176, 176], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_5).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.sigmoid(prediction_patch)
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [0, 176], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[0, 176], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[176, 0], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction = tf.pad(prediction_patch, [[176, 176], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_6).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.sigmoid(prediction_patch)
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [176, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[0, 176], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction = tf.pad(prediction_patch, [[352, 0], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_7).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.sigmoid(prediction_patch)
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [0, 176], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [176, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[0, 176], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction = tf.pad(prediction_patch, [[352, 0], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(model.predict(test_dataset.map(crop_test_8).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (94):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.sigmoid(prediction_patch)
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([256, 80, 1], 0.5), [[0, 0], [0, 176], [0, 0]], "CONSTANT", constant_values=1))
        prediction_patch = tf.math.multiply(prediction_patch, tf.pad(tf.fill([80, 256, 1], 0.5), [[0, 176], [0, 0], [0, 0]], "CONSTANT", constant_values=1))
        prediction = tf.pad(prediction_patch, [[352, 0], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")

    #### Save the predicted images to the folder testing/predictions ####
    print("Start: write predicted images to ../testing/predictions")
    for idx in range(94):
        prediction = predictions[idx]
        prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
        img_prediction = tf.image.encode_png(prediction)
        number = filenames[idx]
        tf.io.write_file("../testing/predictions/test_" + str(number[5:8]) + ".png", img_prediction)
    print("Done")
    print("Start: create kaggle submission file")
    submit_predictions(filenames)
    print("Done")
      
if __name__ == '__main__':
    main()
