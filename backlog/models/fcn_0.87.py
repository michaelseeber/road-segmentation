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

NUM_VALIDATION_IMAGES = 20

IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32
BUFFER_SIZE = 200
OUTPUT_CHANNELS = 3

TRAIN_LENGTH = 128
EPOCHS = 500
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = STEPS_PER_EPOCH


def augment(image, label):
    temp = tf.concat([image, label], axis=-1)
    temp = tf.image.resize(temp, [400, 400])
    temp = tf.image.random_crop(temp, size=[256, 256, 4])
    temp = tf.image.resize(temp, [IMG_WIDTH, IMG_HEIGHT])
    temp = tf.image.random_flip_left_right(temp)
    temp = tf.image.rot90(temp, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image, label = tf.split(temp, num_or_size_splits=[3, 1], axis=-1)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.4)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)

    return image, label


def augment_test_time_1(image):
    image = tf.image.rot90(image, 1)
    return image
def augment_test_time_2(image):
    image = tf.image.rot90(image, 2)
    return image
def augment_test_time_3(image):
    image = tf.image.rot90(image, 3)
    return image
def augment_test_time_4(image):
    image = tf.image.flip_left_right(image)
    return image
def augment_test_time_5(image):
    image = tf.image.flip_left_right(image)
    image = tf.image.rot90(image, 1)
    return image
def augment_test_time_6(image):
    image = tf.image.flip_left_right(image)
    image = tf.image.rot90(image, 2)
    return image
def augment_test_time_7(image):
    image = tf.image.flip_left_right(image)
    image = tf.image.rot90(image, 3)
    return image
  
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


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="Number of epochs we want", type=int)
args = parser.parse_args()
if args.epochs:
  EPOCHS = args.epochs

training_images_path = '../training/images_rotated/'
test_images_path = '../testing/images/'
list_ds = tf.data.Dataset.list_files(training_images_path + '*')
test_list_ds = tf.data.Dataset.list_files(test_images_path + '*', shuffle=False)
train = list_ds.skip(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x,  "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE)
train_dataset = train.shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = list_ds.take(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x,  "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE).map(resize_validation).map(augment_validation, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
#test_dataset = test_list_ds.map(test_process_path).map(resize_test).batch(1)

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

#conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling4)
#conv5 = tf.keras.layers.BatchNormalization() (conv5)
#conv5 = tf.keras.layers.Dropout(0.3) (conv5)
#conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv5)
#conv5 = tf.keras.layers.BatchNormalization() (conv5)
conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', dilation_rate=1) (pooling4)
conv5 = tf.keras.layers.Conv2D(192, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', dilation_rate=2) (conv5)
conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', dilation_rate=4) (conv5)
conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', dilation_rate=8) (conv5)
#conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', dilation_rate=16) (conv5)
#conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', dilation_rate=32) (conv5)

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

#outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv9)
#logits = tf.keras.layers.Conv2D(1, (1, 1), activation=None) (conv9)
#outputs = tf.keras.layers.Activation('sigmoid') (logits)
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation=None) (conv9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.summary()

#metric_MeanIoU = tf.keras.metrics.MeanIoU(num_classes=2)
accuracy = tf.keras.metrics.BinaryAccuracy(
    name='accuracy', dtype=None, threshold=0.0
)
crossentropy = tf.keras.metrics.BinaryCrossentropy(
    name='crossentropy', dtype=None, from_logits=True, label_smoothing=0
)


model.compile(optimizer='adam',
              #loss=soft_dice_loss,
              #loss=lovasz_hinge,
              loss=BCE_and_lovasz_hinge,
              metrics=[f1, accuracy, crossentropy])

model_history = model.fit(train_dataset,
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=validation_dataset,
                          callbacks=[tf.keras.callbacks.TensorBoard()])

# show_predictions(train_dataset, 1)
test_dataset = test_list_ds.map(test_process_path).map(resize_test)
#predictions = model.predict(test_dataset, verbose=1)
predictions = []
predictions.append(model.predict(test_dataset.batch(1), verbose=1))
predictions.append(model.predict(test_dataset.map(augment_test_time_1).batch(1), verbose=1))
predictions.append(model.predict(test_dataset.map(augment_test_time_2).batch(1), verbose=1))
predictions.append(model.predict(test_dataset.map(augment_test_time_3).batch(1), verbose=1))
predictions.append(model.predict(test_dataset.map(augment_test_time_4).batch(1), verbose=1))
predictions.append(model.predict(test_dataset.map(augment_test_time_5).batch(1), verbose=1))
predictions.append(model.predict(test_dataset.map(augment_test_time_6).batch(1), verbose=1))
predictions.append(model.predict(test_dataset.map(augment_test_time_7).batch(1), verbose=1))
filenames = sorted(os.listdir(test_images_path))
Path("predictions").mkdir(parents=True, exist_ok=True)

#print(tf.shape(predictions))
predictions = tf.stack(predictions, axis=1)
#print(tf.shape(predictions))
for idx in range (94):
    prediction = tf.stack([predictions[idx][0],
            tf.image.rot90(predictions[idx][1], 3),
            tf.image.rot90(predictions[idx][2], 2),
            tf.image.rot90(predictions[idx][3], 1),
            tf.image.flip_left_right(predictions[idx][4]),
            tf.image.flip_left_right(tf.image.rot90(predictions[idx][5], 3)),
            tf.image.flip_left_right(tf.image.rot90(predictions[idx][6], 2)),
            tf.image.flip_left_right(tf.image.rot90(predictions[idx][7], 1))])
    prediction = tf.math.reduce_mean(tf.math.sigmoid(prediction), axis=0)
    prediction = tf.image.resize(prediction, [608, 608])
    prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
    img_prediction = tf.image.encode_png(prediction)
    number = filenames[idx]
    tf.io.write_file("testing/predictions/test_" + str(number[5:8]) + ".png", img_prediction)
submit_predictions(filenames)
