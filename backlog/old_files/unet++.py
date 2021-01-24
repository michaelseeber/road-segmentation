import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from mask_to_submission import *
import datetime 
from pathlib import Path
import argparse

AUTOTUNE = tf.data.experimental.AUTOTUNE

NUM_VALIDATION_IMAGES = 6 * 6

IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32
BUFFER_SIZE = 200
OUTPUT_CHANNELS = 3

TRAIN_LENGTH = 128
EPOCHS = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = STEPS_PER_EPOCH

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
def BCE_and_soft_dice_loss(y_true, y_pred):
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return BCE(y_true, y_pred) + 1-dice_coef(y_true, y_pred)

@tf.function
def BCE_and_soft_dice_loss_deep_supervision(y_true, y_pred):
    outputs01, outputs02, outputs03, outputs04 = tf.split(y_pred, num_or_size_splits=[1, 1, 1, 1], axis=-1)
    return BCE_and_soft_dice_loss(y_true, outputs01) + BCE_and_soft_dice_loss(y_true, outputs02) + BCE_and_soft_dice_loss(y_true, outputs03) + BCE_and_soft_dice_loss(y_true, outputs04)

@tf.function
def lovasz_hinge(labels, logits, per_image=True, ignore=None):
#def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

@tf.function
def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   #strict=True,
                   name="loss"
                   )
    return loss

@tf.function
def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

@tf.function
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

@tf.function
def BCE_and_lovasz_hinge(labels, logits):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    #return 0.5 * BCE(labels, logits) + 0.5 * lovasz_hinge(labels, logits)
    return 0.5 * BCE(labels, logits) + 5 * lovasz_hinge(labels, logits)

@tf.function
def MSE_and_gradient_loss(groundtruth, regression):

    dy_true, dx_true = tf.image.image_gradients(groundtruth)
    dy_pred, dx_pred = tf.image.image_gradients(regression)

    return (0.50 * tf.keras.losses.MSE(groundtruth, regression) + 
           0.25 * tf.keras.losses.MSE(dx_true, dx_pred) +
           0.25 * tf.keras.losses.MSE(dy_true, dy_pred))

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

def augment_validation(image, label):
    temp = tf.concat([image, label], axis=-1)
    temp = tf.image.random_flip_left_right(temp)
    #not necessary if there is also rotation
    #temp = tf.image.random_flip_up_down(temp)
    temp = tf.image.rot90(temp, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image, label, = tf.split(temp, num_or_size_splits=[3, 1], axis=-1)

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

def process_path(file_path):
  file_path_mask = tf.strings.regex_replace(file_path, "images_rotated", "groundtruth_rotated")
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

training_images_path = 'training/images_rotated/'
test_images_path = 'testing/images/'
list_ds = tf.data.Dataset.list_files(training_images_path + '*')
test_list_ds = tf.data.Dataset.list_files(test_images_path + '*', shuffle=False)
train = list_ds.skip(NUM_VALIDATION_IMAGES).map(process_path, num_parallel_calls=AUTOTUNE)
#train_dataset = train.cache().shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
train_dataset = train.shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = list_ds.take(NUM_VALIDATION_IMAGES).map(process_path, num_parallel_calls=AUTOTUNE).map(resize_validation).map(augment_validation, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat()
#test_dataset = test_list_ds.map(test_process_path).map(resize_test).batch(1)

# Create FCN model
from tensorflow.keras import layers
 
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, 3))

##256
#conv00 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (inputs)
#conv00 = tf.keras.layers.BatchNormalization() (conv00)
##additional conv?
#conv00 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv00)
#conv00 = tf.keras.layers.BatchNormalization() (conv00)
#pooling00 = tf.keras.layers.MaxPooling2D((2, 2)) (conv00)
#
##128
#conv10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (pooling00)
#conv10 = tf.keras.layers.BatchNormalization() (conv10)
##additional conv?
#conv10 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (conv10)
#conv10 = tf.keras.layers.BatchNormalization() (conv10)
#pooling10 = tf.keras.layers.MaxPooling2D((2, 2)) (conv10)
#upsample10 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv10)
#
##64
#conv20 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (pooling10)
#conv20 = tf.keras.layers.BatchNormalization() (conv20)
##additional conv?
#conv20 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (conv20)
#conv20 = tf.keras.layers.BatchNormalization() (conv20)
#pooling20 = tf.keras.layers.MaxPooling2D((2, 2)) (conv20)
#upsample20 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv20)
#
##32
#conv30 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (pooling20)
#conv30 = tf.keras.layers.BatchNormalization() (conv30)
##additional conv?
#conv30 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (conv30)
#conv30 = tf.keras.layers.BatchNormalization() (conv30)
#pooling30 = tf.keras.layers.MaxPooling2D((2, 2)) (conv30)
#upsample30 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv30)
#
##16
#conv40 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same') (pooling30)
#conv40 = tf.keras.layers.BatchNormalization() (conv40)
##additional conv?
#conv40 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (conv40)
#conv40 = tf.keras.layers.BatchNormalization() (conv40)
#upsample40 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv40)
#
##32
#concat31 = tf.keras.layers.concatenate([upsample40, conv30])
#conv31 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (concat31)
#conv31 = tf.keras.layers.BatchNormalization() (conv31)
##additional conv?
#conv31 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (conv31)
#conv31 = tf.keras.layers.BatchNormalization() (conv31)
#upsample31 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv31)
#
##64
#concat21 = tf.keras.layers.concatenate([upsample30, conv20])
#conv21 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (concat21)
#conv21 = tf.keras.layers.BatchNormalization() (conv21)
##additional conv?
#conv21 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (conv21)
#conv21 = tf.keras.layers.BatchNormalization() (conv21)
#upsample21 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv21)
#
##concat22 = tf.keras.layers.concatenate([upsample31, conv20])
#concat22 = tf.keras.layers.concatenate([upsample31, conv20, conv21])
#conv22 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (concat22)
#conv22 = tf.keras.layers.BatchNormalization() (conv22)
##additional conv?
#conv22 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (conv22)
#conv22 = tf.keras.layers.BatchNormalization() (conv22)
#upsample22 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv22)
#
##128
#concat11 = tf.keras.layers.concatenate([upsample20, conv10])
#conv11 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (concat11)
#conv11 = tf.keras.layers.BatchNormalization() (conv11)
##additional conv?
#conv11 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (conv11)
#conv11 = tf.keras.layers.BatchNormalization() (conv11)
#upsample11 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv11)
#
#concat12 = tf.keras.layers.concatenate([upsample21, conv10, conv11])
#conv12 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (concat12)
#conv12 = tf.keras.layers.BatchNormalization() (conv12)
##additional conv?
#conv12 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (conv12)
#conv12 = tf.keras.layers.BatchNormalization() (conv12)
#upsample12 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv12)
#
##concat13 = tf.keras.layers.concatenate([upsample22, conv10])
#concat13 = tf.keras.layers.concatenate([upsample22, conv10, conv11, conv12])
#conv13 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (concat13)
#conv13 = tf.keras.layers.BatchNormalization() (conv13)
##additional conv?
#conv13 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (conv13)
#conv13 = tf.keras.layers.BatchNormalization() (conv13)
#upsample13 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv13)
#
##256
#concat01 = tf.keras.layers.concatenate([upsample10, conv00])
#conv01 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (concat01)
#
#concat02 = tf.keras.layers.concatenate([upsample11, conv00, conv01])
#conv02 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (concat02)
#
#concat03 = tf.keras.layers.concatenate([upsample12, conv00, conv01, conv02])
#conv03 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (concat03)
#
##concat04 = tf.keras.layers.concatenate([upsample13, conv00])
#concat04 = tf.keras.layers.concatenate([upsample13, conv00, conv01, conv02, conv03])
#conv04 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (concat04)
#
##outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv04)
#outputs01 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv01)
#outputs02 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv02)
#outputs03 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv03)
#outputs04 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv04)
#outputs = tf.keras.layers.concatenate([outputs01, outputs02, outputs03, outputs04], axis=-1)


#256
conv00 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
conv00 = tf.keras.layers.BatchNormalization() (conv00)
#additional conv?
conv00 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv00)
conv00 = tf.keras.layers.BatchNormalization() (conv00)
pooling00 = tf.keras.layers.MaxPooling2D((2, 2)) (conv00)

#128
conv10 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pooling00)
conv10 = tf.keras.layers.BatchNormalization() (conv10)
#additional conv?
conv10 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv10)
conv10 = tf.keras.layers.BatchNormalization() (conv10)
pooling10 = tf.keras.layers.MaxPooling2D((2, 2)) (conv10)
upsample10 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv10)

#64
conv20 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pooling10)
conv20 = tf.keras.layers.BatchNormalization() (conv20)
#additional conv?
conv20 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv20)
conv20 = tf.keras.layers.BatchNormalization() (conv20)
pooling20 = tf.keras.layers.MaxPooling2D((2, 2)) (conv20)
upsample20 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv20)

#32
conv30 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pooling20)
conv30 = tf.keras.layers.BatchNormalization() (conv30)
#additional conv?
conv30 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv30)
conv30 = tf.keras.layers.BatchNormalization() (conv30)
pooling30 = tf.keras.layers.MaxPooling2D((2, 2)) (conv30)
upsample30 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv30)

#16
conv40 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pooling30)
conv40 = tf.keras.layers.BatchNormalization() (conv40)
#additional conv?
conv40 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv40)
conv40 = tf.keras.layers.BatchNormalization() (conv40)
upsample40 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv40)

#32
concat31 = tf.keras.layers.concatenate([upsample40, conv30])
conv31 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat31)
conv31 = tf.keras.layers.BatchNormalization() (conv31)
#additional conv?
conv31 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv31)
conv31 = tf.keras.layers.BatchNormalization() (conv31)
upsample31 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv31)

#64
concat21 = tf.keras.layers.concatenate([upsample30, conv20])
conv21 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat21)
conv21 = tf.keras.layers.BatchNormalization() (conv21)
#additional conv?
conv21 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv21)
conv21 = tf.keras.layers.BatchNormalization() (conv21)
upsample21 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv21)

#concat22 = tf.keras.layers.concatenate([upsample31, conv20])
concat22 = tf.keras.layers.concatenate([upsample31, conv20, conv21])
conv22 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat22)
conv22 = tf.keras.layers.BatchNormalization() (conv22)
#additional conv?
conv22 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv22)
conv22 = tf.keras.layers.BatchNormalization() (conv22)
upsample22 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv22)

#128
concat11 = tf.keras.layers.concatenate([upsample20, conv10])
conv11 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat11)
conv11 = tf.keras.layers.BatchNormalization() (conv11)
#additional conv?
conv11 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv11)
conv11 = tf.keras.layers.BatchNormalization() (conv11)
upsample11 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv11)

concat12 = tf.keras.layers.concatenate([upsample21, conv10, conv11])
conv12 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat12)
conv12 = tf.keras.layers.BatchNormalization() (conv12)
#additional conv?
conv12 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv12)
conv12 = tf.keras.layers.BatchNormalization() (conv12)
upsample12 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv12)

#concat13 = tf.keras.layers.concatenate([upsample22, conv10])
concat13 = tf.keras.layers.concatenate([upsample22, conv10, conv11, conv12])
conv13 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat13)
conv13 = tf.keras.layers.BatchNormalization() (conv13)
#additional conv?
conv13 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv13)
conv13 = tf.keras.layers.BatchNormalization() (conv13)
upsample13 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv13)

#256
concat01 = tf.keras.layers.concatenate([upsample10, conv00])
conv01 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat01)

concat02 = tf.keras.layers.concatenate([upsample11, conv00, conv01])
conv02 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat02)

concat03 = tf.keras.layers.concatenate([upsample12, conv00, conv01, conv02])
conv03 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat03)

#concat04 = tf.keras.layers.concatenate([upsample13, conv00])
concat04 = tf.keras.layers.concatenate([upsample13, conv00, conv01, conv02, conv03])
conv04 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (concat04)

#outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv04)
outputs01 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv01)
outputs02 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv02)
outputs03 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv03)
outputs04 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (conv04)
outputs = tf.keras.layers.concatenate([outputs01, outputs02, outputs03, outputs04], axis=-1)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.summary()

accuracy = tf.keras.metrics.BinaryAccuracy(
    name='accuracy', dtype=None, threshold=0.5
)
crossentropy = tf.keras.metrics.BinaryCrossentropy(
    name='crossentropy', dtype=None, from_logits=False, label_smoothing=0
)


model.compile(optimizer='adam',
              #loss=soft_dice_loss,
              #loss=lovasz_hinge,
              #loss=BCE_and_lovasz_hinge,
              loss=BCE_and_soft_dice_loss_deep_supervision,
              metrics=[accuracy, crossentropy])

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
    #prediction = tf.math.reduce_mean(prediction, axis=-1, keepdims=True)
    _, prediction = tf.split(prediction, [3, 1], axis=-1)
    prediction = tf.math.reduce_mean(prediction, axis=0)
    prediction = tf.image.resize(prediction, [608, 608])
    prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
    img_prediction = tf.image.encode_png(prediction)
    number = filenames[idx]
    tf.io.write_file("testing/predictions/test_" + str(number[5:8]) + ".png", img_prediction)
submit_predictions(test_dataset)
