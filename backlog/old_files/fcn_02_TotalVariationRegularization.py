import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from mask_to_submission import *
import datetime 
from pathlib import Path
import argparse

AUTOTUNE = tf.data.experimental.AUTOTUNE

NUM_VALIDATION_IMAGES = 6

IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 16
BUFFER_SIZE = 200
OUTPUT_CHANNELS = 3

TRAIN_LENGTH = 128
EPOCHS = 6000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = STEPS_PER_EPOCH

@tf.function
def f1(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    return 2 * (tf.keras.backend.sum(y_true * y_pred)+ tf.keras.backend.epsilon()) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + tf.keras.backend.epsilon())

@tf.function
def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.function
def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

@tf.function
def custom_loss(y_true, y_pred):
    #Total Variation as a loss function to suppress noise in images: https://www.tensorflow.org/api_docs/python/tf/image/total_variation
    #return soft_dice_loss + tf.reduce_mean(tf.image.total_variation(y_pred))
    return soft_dice_loss(y_true, y_pred) + tf.reduce_sum(tf.image.total_variation(y_pred))

def augment(image, label):
    temp = tf.concat([image, label], axis=-1)
    temp = tf.image.resize(temp, [400, 400])
    temp = tf.image.random_crop(temp, size=[256, 256, 4])
    temp = tf.image.resize(temp, [IMG_WIDTH, IMG_HEIGHT])
    temp = tf.image.random_flip_left_right(temp)
    temp = tf.image.rot90(temp, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image, label = tf.split(temp, num_or_size_splits=[3, 1], axis=-1)

    return image, label

def augment_validation(image, label):
    temp = tf.concat([image, label], axis=-1)
    temp = tf.image.random_flip_left_right(temp)
    #not necessary if there is also rotation
    #temp = tf.image.random_flip_up_down(temp)
    temp = tf.image.rot90(temp, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image, label = tf.split(temp, num_or_size_splits=[3, 1], axis=-1)

    return image, label
  
def resize_validation(image, label):
  return tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT]),  tf.image.resize(label, [IMG_WIDTH, IMG_HEIGHT])

def resize_test(image):
    return tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
  
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.

  #return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
  return img

def decode_mask(mask):
  # convert the compressed string to a 3D uint8 tensor
  mask = tf.image.decode_png(mask, channels=1)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  mask = tf.image.convert_image_dtype(mask, tf.float32)

  #return tf.image.resize(mask, [IMG_WIDTH, IMG_HEIGHT])
  return mask

def process_path(file_path):
  file_path_mask = tf.strings.regex_replace(file_path, "images", "groundtruth")
  mask = tf.io.read_file(file_path_mask)
  mask = decode_mask(mask)
  
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)

  return img, mask

def test_process_path(file_path):
  
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  
  return img

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


def create_mask(pred_mask):
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

def test_show_predictions(dataset=None, num=1):
  for image in dataset.take(num):
    pred_mask = model.predict(image)
    display([image[0], create_mask(pred_mask)])


def submit_predictions(dataset=None):
  Path("submissions").mkdir(parents=True, exist_ok=True)
  submission_filename = 'submissions/submission_' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + '.csv'
  image_filenames = []
  for i in range(0, 94):
    number = filenames[i]
    filename = 'testing/predictions/test_' + str(number[5:8]) + ".png"
    if not os.path.isfile(filename):
        print(filename + " not found")
        continue
    image_filenames.append(filename)
    
  masks_to_submission(submission_filename, *image_filenames)

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="Number of epochs we want", type=int)
args = parser.parse_args()
if args.epochs:
  EPOCHS = args.epochs

training_images_path = 'training/images/'
test_images_path = 'testing/images/'
list_ds = tf.data.Dataset.list_files(training_images_path + '*')
test_list_ds = tf.data.Dataset.list_files(test_images_path + '*', shuffle=False)
train = list_ds.skip(NUM_VALIDATION_IMAGES).map(process_path, num_parallel_calls=AUTOTUNE)
train_dataset = train.cache().shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = list_ds.take(NUM_VALIDATION_IMAGES).map(process_path, num_parallel_calls=AUTOTUNE).map(resize_validation).map(augment_validation, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
test_dataset = test_list_ds.map(test_process_path).map(resize_test).batch(1)

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask

# Create FCN model
from tensorflow.keras import layers
 
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, 3))

conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
conv1 = tf.keras.layers.BatchNormalization() (conv1)
conv1 = tf.keras.layers.Dropout(0.1) (conv1)
conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1)
conv1 = tf.keras.layers.BatchNormalization() (conv1)
pooling1 = tf.keras.layers.MaxPooling2D((2, 2)) (conv1)

conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling1)
conv2 = tf.keras.layers.BatchNormalization() (conv2)
conv2 = tf.keras.layers.Dropout(0.1) (conv2)
conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2)
conv2 = tf.keras.layers.BatchNormalization() (conv2)
pooling2 = tf.keras.layers.MaxPooling2D((2, 2)) (conv2)

conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling2)
conv3 = tf.keras.layers.BatchNormalization() (conv3)
conv3 = tf.keras.layers.Dropout(0.2) (conv3)
conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3)
conv3 = tf.keras.layers.BatchNormalization() (conv3)
pooling3 = tf.keras.layers.MaxPooling2D((2, 2)) (conv3)

conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling3)
conv4 = tf.keras.layers.BatchNormalization() (conv4)
conv4 = tf.keras.layers.Dropout(0.2) (conv4)
conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv4)
conv4 = tf.keras.layers.BatchNormalization() (conv4)
pooling4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (conv4)

conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling4)
conv5 = tf.keras.layers.BatchNormalization() (conv5)
conv5 = tf.keras.layers.Dropout(0.3) (conv5)
conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv5)
conv5 = tf.keras.layers.BatchNormalization() (conv5)

upsample6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
upsample6 = tf.keras.layers.concatenate([upsample6, conv4])
conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample6)
conv6 = tf.keras.layers.BatchNormalization() (conv6)
conv6 = tf.keras.layers.Dropout(0.2) (conv6)
conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv6)
conv6 = tf.keras.layers.BatchNormalization() (conv6)

upsample7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
upsample7 = tf.keras.layers.concatenate([upsample7, conv3])
conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample7)
conv7 = tf.keras.layers.BatchNormalization() (conv7)
conv7 = tf.keras.layers.Dropout(0.2) (conv7)
conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv7)
conv7 = tf.keras.layers.BatchNormalization() (conv7)

upsample8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
upsample8 = tf.keras.layers.concatenate([upsample8, conv2])
conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample8)
conv8 = tf.keras.layers.BatchNormalization() (conv8)
conv8 = tf.keras.layers.Dropout(0.1) (conv8)
conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv8)
conv8 = tf.keras.layers.BatchNormalization() (conv8)

upsample9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv8)
upsample9 = tf.keras.layers.concatenate([upsample9, conv1], axis=3)
conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample9)
conv9 = tf.keras.layers.BatchNormalization() (conv9)
conv9 = tf.keras.layers.Dropout(0.1) (conv9)
conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv9)
conv9 = tf.keras.layers.BatchNormalization() (conv9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.summary()

model.compile(optimizer='adam',
              loss=soft_dice_loss,
              #loss=custom_loss,
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

model_history = model.fit(train_dataset,
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=validation_dataset,
                          callbacks=[tf.keras.callbacks.TensorBoard()])

# show_predictions(train_dataset, 1)

predictions = model.predict(test_dataset, verbose=1)
filenames = os.listdir(test_images_path)
Path("predictions").mkdir(parents=True, exist_ok=True)

for i, prediction in enumerate(predictions):
  prediction = tf.image.resize(prediction, [608, 608])
  prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
  img_prediction = tf.image.encode_png(prediction)
  number = filenames[i]
  tf.io.write_file("testing/predictions/test_" + str(number[5:8]) + ".png", img_prediction)
# test_show_predictions(test_dataset, 10)
submit_predictions(test_dataset)
