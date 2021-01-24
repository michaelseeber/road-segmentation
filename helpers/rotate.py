import os
from skimage import io, util
from skimage.transform import rotate
import numpy as np

#load training images
input_folder = '../training/images'
input_files = [os.path.join(root, filename)
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
images = []
for input_file in input_files:
    #images.append(io.imread(input_file).flatten())
    images.append(io.imread(input_file))

#apply rotations
output_folder = '../training/images_rotated'
#output_files = [os.path.join(output_folder, 'rot0_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot00.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    io.imsave(output_files[idx], image)

#output_files = [os.path.join(output_folder, 'rot15_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot15.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 15, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)

#output_files = [os.path.join(output_folder, 'rot30_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot30.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 30, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)

#output_files = [os.path.join(output_folder, 'rot45_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot45.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 45, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)

#output_files = [os.path.join(output_folder, 'rot60_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot60.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 60, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)

#output_files = [os.path.join(output_folder, 'rot75_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot75.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 75, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)



#load groundtruth images
input_folder = '../training/groundtruth'
input_files = [os.path.join(root, filename)
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
images = []
for input_file in input_files:
    #images.append(io.imread(input_file).flatten())
    images.append(io.imread(input_file))

#apply rotations
output_folder = '../training/groundtruth_rotated'
#output_files = [os.path.join(output_folder, 'rot0_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot00.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    io.imsave(output_files[idx], image)

#output_files = [os.path.join(output_folder, 'rot15_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot15.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 15, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)

#output_files = [os.path.join(output_folder, 'rot30_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot30.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 30, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)

#output_files = [os.path.join(output_folder, 'rot45_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot45.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 45, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)

#output_files = [os.path.join(output_folder, 'rot60_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot60.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 60, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)

#output_files = [os.path.join(output_folder, 'rot75_' + filename)
output_files = [os.path.join(output_folder, filename[:-4] + '_rot75.png')
          for root, dirs, files in os.walk(input_folder)
          for filename in files
          if filename.lower().endswith('.png')]
for idx, image in enumerate(images):
    rot = rotate(image, 75, mode='reflect')
    rot = (np.clip(rot*255, 0, 255)).astype(np.uint8)
    io.imsave(output_files[idx], rot)
