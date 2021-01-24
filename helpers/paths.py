import tensorflow as tf


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
  ## Use `convert_image_dtype` to convert to floats in the [0,1] range.
  #mask = tf.image.convert_image_dtype(mask, tf.float32)
  
  # given masks images do have values > 0 and < 255
  mask = tf.cast(mask > 127, tf.float32)

  #return tf.image.resize(mask, [IMG_WIDTH, IMG_HEIGHT])
  return mask

def process_path(file_path, old, new):
  file_path_mask = tf.strings.regex_replace(file_path, old, new)
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
