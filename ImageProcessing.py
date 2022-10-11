# Image Processing using numpy, skimage, and matplotlib

import numpy as np
from skimage import data, color, io, img_as_float, img_as_ubyte
from skimage import exposure
from skimage import filters
from skimage import feature

if __name__ == '__main__':

    # Download a bird image from the internet
    img = io.imread('https://avatars.githubusercontent.com/u/60761870?v=4')


