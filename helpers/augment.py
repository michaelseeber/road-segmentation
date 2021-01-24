import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import cv2
import random

IMG_WIDTH = 256
IMG_HEIGHT = 256

def augment(image, label, resize=400, scale=1, horizontal_flip=True, random_crop=True, rotate90=True, color=True, block_noise=True):
    temp = tf.concat([image, label], axis=-1)
    if resize is not None:
        temp = tf.image.resize(temp, [resize, resize])
    if random_crop:
        temp = tf.image.random_crop(temp, size=[256, 256, 4])
        
    if horizontal_flip:
        temp = tf.image.random_flip_left_right(temp)
    if rotate90:
        temp = tf.image.rot90(temp, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    temp = tf.image.resize(temp, [IMG_WIDTH, IMG_HEIGHT])

    image, label = tf.split(temp, num_or_size_splits=[3, 1], axis=-1)
    
    if color:
        color_ordering = random.randint(0,3)
        if color_ordering == 0:
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
          image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
    
    if block_noise:
        #https://stackoverflow.com/questions/57374732/how-to-apply-imgaug-augmentation-to-tf-datadataset-in-tensorflow-2-0
        image_shape = image.shape
        [image,] = tf.py_function(apply_gaussian_block, [image], [tf.float32])
        image.set_shape(image_shape)

    return image, label


def augmentAutomatically(images):
    augmentation_size = len(images)
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=360,
        shear_range=0.1,
        zoom_range=0.2,
        fill_mode="reflect"
    )

    seed = np.random.choice(100000, 1)
    d_images = datagen.flow(images, batch_size=augmentation_size, shuffle=False, seed=seed)

    images_augmented = next(d_images)

    return images_augmented

def augment_validation(image, label):
    temp = tf.concat([image, label], axis=-1)
    temp = tf.image.random_flip_left_right(temp)
    #not necessary if there is also rotation
    #temp = tf.image.random_flip_up_down(temp)
    temp = tf.image.rot90(temp, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image, label = tf.split(temp, num_or_size_splits=[3, 1], axis=-1)

    return image, label

def augment_validation_with_rand_crop(image, label):
    temp = tf.concat([image, label], axis=-1)
    temp =  tf.image.random_crop(temp, size=[256, 256, 4])
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

def crop_test_0(image):
    #3x3: patch[0, 0]
    #5x5: patch[0, 0]
    return tf.image.crop_to_bounding_box(image, 0, 0, 256, 256)

def crop_test_1(image):
    #3x3: patch[0, 1]
    #5x5: patch[0, 2]
    return tf.image.crop_to_bounding_box(image, 0, 176, 256, 256)

def crop_test_2(image):
    #3x3: patch[0, 2]
    #5x5: patch[0, 4]
    return tf.image.crop_to_bounding_box(image, 0, 352, 256, 256)

def crop_test_3(image):
    #3x3: patch[1, 0]
    #5x5: patch[2, 0]
    return tf.image.crop_to_bounding_box(image, 176, 0, 256, 256)

def crop_test_4(image):
    #3x3: patch[1, 1]
    #5x5: patch[2, 2]
    return tf.image.crop_to_bounding_box(image, 176, 176, 256, 256)

def crop_test_5(image):
    #3x3: patch[1, 2]
    #5x5: patch[2, 4]
    return tf.image.crop_to_bounding_box(image, 176, 352, 256, 256)

def crop_test_6(image):
    #3x3: patch[2, 0]
    #5x5: patch[4, 0]
    return tf.image.crop_to_bounding_box(image, 352, 0, 256, 256)

def crop_test_7(image):
    #3x3: patch[2, 1]
    #5x5: patch[4, 2]
    return tf.image.crop_to_bounding_box(image, 352, 176, 256, 256)

def crop_test_8(image):
    #3x3: patch[2, 2]
    #5x5: patch[4, 2]
    return tf.image.crop_to_bounding_box(image, 352, 352, 256, 256)

def crop_test5x5_01(image):
    #5x5: patch[0, 1]
    return tf.image.crop_to_bounding_box(image, 0, 88, 256, 256)

def crop_test5x5_12(image):
    #5x5: patch[0, 2]
    return tf.image.crop_to_bounding_box(image, 0, 264, 256, 256)

def crop_test5x5_03(image):
    #5x5: patch[1, 0]
    return tf.image.crop_to_bounding_box(image, 88, 0, 256, 256)

def crop_test5x5_04(image):
    #5x5: patch[1, 1]
    return tf.image.crop_to_bounding_box(image, 88, 88, 256, 256)

def crop_test5x5_14(image):
    #5x5: patch[1, 2]
    return tf.image.crop_to_bounding_box(image, 88, 176, 256, 256)

def crop_test5x5_24(image):
    #5x5: patch[1, 3]
    return tf.image.crop_to_bounding_box(image, 88, 264, 256, 256)

def crop_test5x5_25(image):
    #5x5: patch[1, 4]
    return tf.image.crop_to_bounding_box(image, 88, 352, 256, 256)

def crop_test5x5_34(image):
    #5x5: patch[2, 1]
    return tf.image.crop_to_bounding_box(image, 176, 88, 256, 256)

def crop_test5x5_45(image):
    #5x5: patch[2, 3]
    return tf.image.crop_to_bounding_box(image, 176, 264, 256, 256)

def crop_test5x5_36(image):
    #5x5: patch[3, 0]
    return tf.image.crop_to_bounding_box(image, 264, 0, 256, 256)

def crop_test5x5_37(image):
    #5x5: patch[3, 1]
    return tf.image.crop_to_bounding_box(image, 264, 88, 256, 256)

def crop_test5x5_47(image):
    #5x5: patch[3, 2]
    return tf.image.crop_to_bounding_box(image, 264, 176, 256, 256)

def crop_test5x5_48(image):
    #5x5: patch[3, 3]
    return tf.image.crop_to_bounding_box(image, 264, 264, 256, 256)

def crop_test5x5_58(image):
    #5x5: patch[3, 4]
    return tf.image.crop_to_bounding_box(image, 264, 352, 256, 256)

def crop_test5x5_67(image):
    #5x5: patch[4, 1]
    return tf.image.crop_to_bounding_box(image, 352, 88, 256, 256)

def crop_test5x5_78(image):
    #5x5: patch[4, 2]
    return tf.image.crop_to_bounding_box(image, 352, 264, 256, 256)

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


def apply_gaussian_block(image, size_bounds = [60, 100]):
    w, h, _ = image.shape
    size = random.randint(size_bounds[0], size_bounds[1])

    gauss_image = get_gaussian_noise_block_image(size = size)
    x_offset = random.randint(0, w - size)
    y_offset =  random.randint(0, h - size)
    newIm = image.numpy()
    newIm[y_offset:y_offset+gauss_image.shape[0], x_offset:x_offset+gauss_image.shape[1]] = gauss_image
    newIm = tf.convert_to_tensor(newIm, dtype=tf.float32)

    return newIm


def get_gaussian_noise_block_image(size):
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(size,size,3))
    gauss = gauss.reshape(size,size,3)
    gauss = cv2.normalize(gauss, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    return gauss

def try_out_gaussian():
    satellite_path = '../testing/images/'
    satellite_images_names = sorted(os.listdir(satellite_path))
    satellite_images_list = [cv2.imread(file) for file in sorted(
    glob.glob(satellite_path + '*.png'))]
    for i, im in enumerate(satellite_images_list):
        fig, axes = plt.subplots(ncols=2, figsize=(6, 4), sharex=True,
                                            sharey=True)
        gauss = apply_gaussian_block(im, size_bounds= [60, 100])
  
        im_list = [im, gauss]
        name_list = ['original', 'with block']
        for i, el in enumerate(axes):
                el.imshow(im_list[i])
                el.set_title(name_list[i])
                el.axis('off')
        plt.show()

# try_out_gaussian()
