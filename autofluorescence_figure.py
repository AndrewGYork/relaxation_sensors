#!/usr/bin/python
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy.ndimage import gaussian_filter
from tifffile import imread, imwrite

def main():
    # Load our worm biosensor images and parse the timestamps:
    data = imread('0_data.tif') # Acquired by Maria on 2020_12_01
    print(data.shape, data.dtype)
    timestamps = decode_timestamps(data)['microseconds']
    
    # Crop off the timestamp pixels, convert to float:
    data = data[:, 7:, :].astype('float32')
    print(data.shape, data.dtype)

    # Extract the relevant images: minimum and maximum photoactivation
    deactivated = data[(5, 21, 37), :, :] # The dimmest image(s)
    imwrite('1_deactivated.tif', deactivated, imagej=True)
    activated = data[(14, 30, 46), :, :] # The brightest image(s)
    imwrite('2_activated.tif', activated, photometric='minisblack')
    photoswitching = activated - deactivated
    imwrite('3_photoswitching.tif', photoswitching, imagej=True)

    # Estimate the contribution of the photoswitching fluorophores to
    # the deactivated image by hand-tuning a subtraction. This is
    # basically low-effort spectral unmixing:
    background = deactivated - 0.53 * photoswitching
    imwrite('4_background.tif', background, imagej=True)

    # ~3.4 seconds elapse between deactivated and activated images:
    intervals = timestamps[(14, 30, 46),] - timestamps[(5, 21, 37),]
    print(intervals / 1e6)

    # Smooth and color-merge photoswitching/background
    photoswitching = adjust_contrast(photoswitching, 0, 500)
    photoswitching = gaussian_filter(photoswitching, sigma=(0, 2, 2))
    background = adjust_contrast(background, 100, 700)
    background = gaussian_filter(background, sigma=(0, 2, 2))
    img = np.zeros((data.shape[1], data.shape[2], 3)) # RGB image
    img[:, :, 0] = background[2, :, :]
    img[:, :, 1] = photoswitching[2, :, :]
    img[:, :, 2] = background[2, :, :]
    just_magenta, just_green = np.array([1, 0, 1]), np.array([0, 1, 0])
    imwrite('5_background_c.tif', (img * just_magenta * 255).astype(np.uint8))
    imwrite('6_fluorophores_c.tif', (img * just_green * 255).astype(np.uint8))
    imwrite('7_overlay.tif', (img * 255).astype(np.uint8))


def decode_timestamps(image_stack):
    """Decode PCO image timestamps from binary-coded decimal.

    See github.com/AndrewGYork/tools/blob/master/pco.py for the full version
    """
    timestamps = image_stack[:, 0, :14]
    timestamps = (timestamps & 0x0F) + (timestamps >> 4) * 10
    ts = {}
    ts['microseconds'] = np.sum(
        timestamps[:, 8:14] * np.array((3600e6, 60e6, 1e6, 1e4, 1e2, 1)),
        axis=1, dtype='uint64')
    return ts

def adjust_contrast(img, min_, max_):
    """Like setting "minimum" and "maximum" image contrast in ImageJ

    Output image intensity will range from zero to one.
    Non-quantitative, just useful for display.
    """
    img = np.clip(img, min_, max_)
    img -= min_
    img /= max_ - min_
    return img

main()
