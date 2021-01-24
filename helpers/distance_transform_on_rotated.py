import os
from skimage import io, util
from skimage.morphology import medial_axis
import numpy as np

#load groundtruth images
input_folder = './training/groundtruth_rotated'
input_files = [os.path.join(root, filename)
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
images = []
for input_file in input_files:
    #images.append(io.imread(input_file).flatten())
    images.append(io.imread(input_file))

#apply distance transform
output_folder = './training/groundtruth_rotated_distance_transform'
output_files = [os.path.join(output_folder, filename)
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    #_, distance_transform = medial_axis(image, mask=None, return_distance=True)
    #io.imsave(output_files[idx], distance_transform)
    
    _, distance_transform_foreground = medial_axis(image, mask=None, return_distance=True)
    #_, distance_transform_background = medial_axis(util.invert(image), mask=None, return_distance=True)
    _, distance_transform_background = medial_axis(237 - image, mask=None, return_distance=True)
    #output_image = ((distance_transform_foreground - distance_transform_background) / 1140 * 255 + 127).astype(np.uint8)
    output_image = (np.clip((distance_transform_foreground - distance_transform_background) + 127, 0, 255)).astype(np.uint8)
    print('min = ', np.amin(output_image), ', max = ', np.amax(output_image))
    io.imsave(output_files[idx], output_image)
