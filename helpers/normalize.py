
import tensorflow as tf

def normalize(input_image, real_image=None):
  input_image = (input_image / 0.5) - 1
  if real_image is not None:
    real_image = (real_image / 0.5) - 1
    return input_image, real_image
  else:
    return input_image