# Image Processing using numpy, skimage, and matplotlib

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, exposure, transform, feature, measure, morphology, util

# Function plot color image
def plot_color_image(image, title):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis('on')
    plt.show()

def rotate_image(image, angle):
    rotated_image = transform.rotate(image, angle)
    return rotated_image

# Function flip image
def flip_image(image, flip_type):
    flipped_image = np.flip(image, flip_type)
    return flipped_image

# Function crop image
def crop_image(image, crop_factor):
    cropped_image = image[crop_factor[0]:crop_factor[1], crop_factor[2]:crop_factor[3]]
    return cropped_image

def main():
    # Load image
    IMAGE = io.imread('Datasets/img.png')
    plot_color_image(IMAGE, 'Original Image')

    # Rotate image
    ROTATED_IMAGE = rotate_image(IMAGE, 90)
    plot_color_image(ROTATED_IMAGE, 'Rotated Image')

    # Crop the image to 4 quadrants and plot them in a subplot
    CROPPED_IMAGE_1 = crop_image(IMAGE, [0, 200, 0, 200])
    CROPPED_IMAGE_2 = crop_image(IMAGE, [0, 200, 200, 400])
    CROPPED_IMAGE_3 = crop_image(IMAGE, [200, 400, 0, 200])
    CROPPED_IMAGE_4 = crop_image(IMAGE, [200, 400, 200, 400])
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(CROPPED_IMAGE_1)
    ax[0, 0].set_title('Cropped Image 1')
    ax[0, 1].imshow(CROPPED_IMAGE_2)
    ax[0, 1].set_title('Cropped Image 2')
    ax[1, 0].imshow(CROPPED_IMAGE_3)
    ax[1, 0].set_title('Cropped Image 3')
    ax[1, 1].imshow(CROPPED_IMAGE_4)
    ax[1, 1].set_title('Cropped Image 4')
    plt.show()

    # Plot the histogram of the image
    HISTOGRAM = exposure.histogram(IMAGE)
    plt.figure()
    plt.plot(HISTOGRAM[1], HISTOGRAM[0])
    plt.title('Histogram of the Image')
    plt.show()

    # Reduce the noise in the image using Gaussian filter
    GAUSSIAN_FILTERED_IMAGE = filters.gaussian(IMAGE, sigma=1)
    plot_color_image(GAUSSIAN_FILTERED_IMAGE, 'Gaussian Filtered Image')

    # Contrast adjustment using histogram equalization
    HISTOGRAM_EQUALIZED_IMAGE = exposure.equalize_hist(IMAGE)
    plot_color_image(HISTOGRAM_EQUALIZED_IMAGE, 'Histogram Equalized Image')

if __name__ == '__main__':
    main()