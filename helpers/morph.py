import cv2
import numpy as np
from skimage.util import img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, area_opening
from skimage.morphology import square, star, diamond


FILTER_SIZE = 50
CLOSING_FILTER_SIZE = 20
DILATED_FILTER_SIZE = 10
EROSION_FILTER_SIZE = 20
OPENING_FILTER_SIZE = 15
AREA_SIZE = 900

# Morphological closing is dilation (=Makes white boundary bigger) followed by erosion(=erodes pixels at boundary away -> Removes white noise, but makes white area smaller).

# This closes small holes
def morph_closing(image, plot_comparison=False, filter_size = CLOSING_FILTER_SIZE):
    data = img_as_ubyte(image)
    new = closing(data, square(filter_size))
    if plot_comparison:
        plot_comparison(image, new, 'closing')
    return new

# Removes noise
def morph_opening(image, plot_comparison=False, filter_size = OPENING_FILTER_SIZE):
    data = img_as_ubyte(image)
    new = opening(data, square(filter_size))
    if plot_comparison:
        plot_comparison(image, new, 'opening')
    return new

def morph_erosion(image, plot_comparison=False, filter_size = EROSION_FILTER_SIZE):
    data = img_as_ubyte(image)
    new = erosion(data, square(filter_size))
    if plot_comparison:
        plot_comparison(image, new, 'erosion')
    return new
     
def morph_dilation(image, plot_comparison=False, filter_size = DILATED_FILTER_SIZE):
    data = img_as_ubyte(image)
    new = dilation(data, square(filter_size))
    if plot_comparison:
        plot_comparison(image, new, 'dilation')
    return new

def morph_area_opening(image, plot_comparison=False, area_size = AREA_SIZE):
    data = img_as_ubyte(image)
    new = area_opening(data, area_size)
    if plot_comparison:
        plot_comparison(image, new, 'area opening')
    return new


def morph_tests(redo=True):
    satellite_path = '../testing/images/'
    old_images_path = 'predictions_thresholded/'
    new_images_path = '../morphed/'

    satellite_images_names = sorted(os.listdir(satellite_path))
    filenames = sorted(os.listdir(old_images_path))

    satellite_images_list = [cv2.imread(file) for file in sorted(
    glob.glob(satellite_path + '*.png'))]

    images_list = [cv2.imread(file) for file in sorted(
    glob.glob(old_images_path + '*.png'))]
    
    Path(new_images_path).mkdir(parents=True, exist_ok=True)

    if redo:
        for i, im in enumerate(images_list):
                closed = morph_closing(im)
                closed_dilated = morph_dilation(closed)

                opened = morph_opening(im)
                opened_closed = morph_area_opening(opened)
                opened_closed_dilated = morph_dilation(opened_closed)

                area = morph_area_closing(im)
                area_closed = morph_area_opening(area)
                area_closed_dilated = morph_dilation(area_closed)

                new_im = area
        
                original_image = satellite_images_list[i]
                
                # encoded = tf.image.encode_png(new_im)
                name = new_images_path + "test_morphed_" + str(filenames[i][5:8]) + ".png"
                io.imsave(name, new_im)

                # cv2.imwrite(new_images_path + "test_morphed" + str(filenames[i][5:8]) + ".png", new_im)
                print("Written to " + new_images_path + "test_morphed_" + str(filenames[i][5:8]) + ".png")

                plot_comparison([original_image, im, closed, closed_dilated,  opened_closed_dilated, area_closed_dilated],
                ['satellite', 'thresholded', 'closed', 'closed+dilated', 'opened+closed+dilated',  'area+closed+dilated'],
                [None] + [plt.cm.gray] * 20,
                str(filenames[i][5:8]),
                show = False)