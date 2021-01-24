import sys
sys.path.append('../')
from helpers.metrics import *
from helpers.augment import *
from helpers.normalize import *
from helpers.display import *
from helpers.paths import *
from helpers.submission import *
import tensorflow as tf
import numpy as np
import os
import argparse
import time
import datetime
from matplotlib import pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

PATH = "/content/drive/My Drive/CIL Project Images"

NUM_VALIDATION_IMAGES = 16
EPOCHS = 200
BATCH_SIZE = 1
BUFFER_SIZE = 400

IMG_WIDTH = 256
IMG_HEIGHT = 256

OUTPUT_CHANNELS = 1

NUM_TEST = 94


##########################################
# INCOMPELTE WORK IN PROGRESS
##########################################
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="Number of epochs we want", type=int)
    args = parser.parse_args()
    if args.epochs:
      EPOCHS = args.epochs

    ## Load the Data
    # Explain agmentation etc... TODO

    training_images_path = PATH + '/images_rotated/'
    test_images_path = PATH + '/test_images/'

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    list_ds = tf.data.Dataset.list_files(training_images_path + '*')
    test_list_ds = tf.data.Dataset.list_files(test_images_path + '*', shuffle=False)
    train =    list_ds.skip(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE)
    train_dataset = train.cache().shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).map(normalize).batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_dataset = list_ds.take(NUM_VALIDATION_IMAGES).map(lambda x: process_path(x, "images_rotated", "groundtruth_rotated"), num_parallel_calls=AUTOTUNE).map(augment_validation_with_rand_crop).map(normalize).batch(BATCH_SIZE)
    test_dataset = test_list_ds.map(test_process_path).map(normalize)


    ## Generator
    # The architecture of generator is a modified U-Net.
    # Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
    # Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
    # There are skip connections between the encoder and decoder (as in U-Net).
    generator = Generator()

    ## Discriminator
    # The Discriminator is a PatchGAN.
    # Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
    # The shape of the output after the last layer is (batch_size, 30, 30, 1)
    # Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
    # Discriminator receives 2 inputs.
    # Input image and the target image, which it should classify as real.
    # Input image and the generated image (output of generator), which it should classify as fake.
    # We concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))
    discriminator = Discriminator()

    ## Optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)

    ## Checkpoints and logs
    checkpoint_dir = '/content/drive/My Drive/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    log_dir="logs/"
    summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    ## Train
    fit(train_dataset, EPOCHS, validation_dataset)

    ## Restore last checkpoint
    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    ## Predict on Test Image using patching
    predict_using_patches(test_dataset)



def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result
        
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    return result


## Generator loss
# It is a sigmoid cross entropy loss of the generated images and an array of ones.
# The paper also includes L1 loss which is MAE between the generated image and the target image.
# The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the paper.
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


## Discriminator loss
# The discriminator loss function takes 2 inputs; real images, generated images
# real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since these are the real images)
# generated_loss is a sigmoid balanced cross entropy loss of the generated images and an array of zeros(since these are the fake images)
# Then the total_loss is the sum of real_loss and the generated_loss
# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def loss_object(y_true, y_pred):
    return BCE_with_logits(y_true, y_pred)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss


# Note: The training=True is intentional here since we want the batch statistics while running the model on the test dataset.
# If we use training=False, we will get the accumulated statistics learned from the training dataset (which we don't want)
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tf.squeeze(tar[0]), tf.squeeze(prediction[0])]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it. * 0.5 + 0.5
    plt.imshow(display_list[i], cmap="gray" )
    plt.axis('off')
  plt.show()

# Sae as generate_images but no visual output
def make_prediction(input, verbose=1):
  predictions = []
  for el in input:
    predictions.append(tf.squeeze(generator(el, training=True), [0]))
  return predictions


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)


def predict_using_patches(test_dateset, test_images_path):
    filenames = sorted(os.listdir(test_images_path))
    predictions = []
    for idx in range (NUM_TEST):
    predictions.append(tf.zeros([608, 608, 1], tf.float32))
    print("Start: predict on patch [0, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_0).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_0).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_0).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_0).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_0).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_0).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_0).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_0).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_01).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_01).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_01).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_01).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_01).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_01).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_01).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_01).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_1).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_1).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_1).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_1).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_1).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_1).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_1).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_12).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_12).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_12).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_12).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_12).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_12).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_12).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_12).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [0, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_2).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_2).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_2).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_2).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_2).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_2).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_2).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 1.0), tf.fill([88, 256, 1], 0.75), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[0, 352], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_03).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_03).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_03).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_03).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_03).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_03).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_03).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_03).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_04).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_04).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_04).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_04).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_04).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_04).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_04).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_04).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_14).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_14).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_14).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_14).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_14).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_14).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_14).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_14).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_24).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_24).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_24).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_24).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_24).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_24).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_24).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_24).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [1, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_25).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_25).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_25).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_25).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_25).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_25).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_25).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_25).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([88, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[88, 264], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_3).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_3).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_3).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_3).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_3).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_3).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_3).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_34).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_34).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_34).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_34).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_34).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_34).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_34).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_34).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_4).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_4).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_4).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_4).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_4).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_4).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_4).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_45).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_45).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_45).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_45).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_45).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_45).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_45).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_45).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [2, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_5).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_5).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_5).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_5).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_5).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_5).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_5).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([96, 256, 1], 0.5), tf.fill([80, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[176, 176], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_36).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_36).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_36).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_36).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_36).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_36).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_36).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_36).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_37).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_37).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_37).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_37).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_37).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_37).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_37).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_37).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_47).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_47).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_47).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_47).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_47).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_47).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_47).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_47).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_48).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_48).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_48).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_48).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_48).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_48).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_48).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_48).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [3, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_58).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_58).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_58).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_58).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_58).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_58).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_58).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_58).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.5), tf.fill([88, 256, 1], 0.25)]))
        prediction = tf.pad(prediction_patch, [[264, 88], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 0] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_6).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_6).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_6).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_6).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_6).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_6).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_6).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 1.0), tf.fill([256, 88, 1], 0.75), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [0, 352], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 1] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_67).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_67).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_67).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_67).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_67).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_67).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_67).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_67).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 88, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [88, 264], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 2] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_7).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_7).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_7).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_7).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_7).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_7).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_7).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_7).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 96, 1], 0.5), tf.fill([256, 80, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [176, 176], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 3] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_78).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_78).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_78).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_78).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_78).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_78).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_78).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test5x5_78).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.5), tf.fill([256, 88, 1], 0.25)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [264, 88], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")
    print("Start: predict on patch [4, 4] using 8 transformations as test time augmentation")
    predictions_patch = []
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_8).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_8).map(augment_test_time_1).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_8).map(augment_test_time_2).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_8).map(augment_test_time_3).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_8).map(augment_test_time_4).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_8).map(augment_test_time_5).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_8).map(augment_test_time_6).batch(1), verbose=1))
    predictions_patch.append(make_prediction(test_dataset.map(crop_test_8).map(augment_test_time_7).batch(1), verbose=1))
    predictions_patch = tf.stack(predictions_patch, axis=1)
    for idx in range (NUM_TEST):
        prediction_patch = tf.stack([predictions_patch[idx][0],
                tf.image.rot90(predictions_patch[idx][1], 3),
                tf.image.rot90(predictions_patch[idx][2], 2),
                tf.image.rot90(predictions_patch[idx][3], 1),
                tf.image.flip_left_right(predictions_patch[idx][4]),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][5], 3)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][6], 2)),
                tf.image.flip_left_right(tf.image.rot90(predictions_patch[idx][7], 1))])
        prediction_patch = tf.math.reduce_mean(prediction_patch, axis=0)
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=1, values=[
            tf.fill([256, 80, 1], 0.25), tf.fill([256, 88, 1], 0.75), tf.fill([256, 88, 1], 1.0)]))
        prediction_patch = tf.math.multiply(prediction_patch, tf.concat(axis=0, values=[
            tf.fill([80, 256, 1], 0.25), tf.fill([88, 256, 1], 0.75), tf.fill([88, 256, 1], 1.0)]))
        prediction = tf.pad(prediction_patch, [[352, 0], [352, 0], [0, 0]], "CONSTANT")
        predictions[idx] = tf.math.add(predictions[idx], prediction)
    print("Done")

    #write to disk
    print("Start: write predicted images to ../testing/predictions")
    for idx in range(NUM_TEST):
        prediction = predictions[idx]
        
        #back to 0-1 range
        prediction = prediction * 0.5 + 0.5

        prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
        img_prediction = tf.image.encode_png(prediction)
        plt.imshow(tf.squeeze(prediction))
        plt.show()
        plt.axis('off')
        number = filenames[idx]
        tf.io.write_file(PATH + '/predictions_michi/' + str(number[5:8]) + ".png", img_prediction)
    print("Done")
    import datetime
    print("Start: create kaggle submission file") 
    submit_predictions(filenames)
    print("Done")

      
if __name__ == '__main__':
    main()
