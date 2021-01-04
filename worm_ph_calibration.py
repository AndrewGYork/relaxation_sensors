#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy.ndimage import gaussian_filter
from tifffile import imread, imwrite # v2020.6.3 or newer

input_dir = Path.cwd() / '0_calibration_beads'
temp_dir = Path.cwd() / 'intermediate_calibration_output'
output_dir = Path.cwd()
# Sanity checks:
assert input_dir.is_dir()
temp_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

def main():
    # Load our bead biosensor calibration images and parse the timestamps:
    # Acquired by Maria on 2020_12_01
    for basename, slice_y, slice_x in (
        ('ph_6p0', slice(198, 540), slice(972, 1455)),
        ('ph_6p5', slice(1074, 1128), slice(678, 1017)),
        ('ph_7p0', slice(1053, 1242), slice(1320, 1749)),
        ('ph_7p5', slice(843, 1071), slice(1158, 1617)),
        ('ph_8p0', slice(840, 936), slice(1170, 1647)),
        ):
        data = imread(input_dir / (basename + '.tif'))
        
        timestamps = (
            1e-6*decode_timestamps(data)['microseconds'].astype('float64'))
        # Crop, convert to float:
        data = data[:, 7:, :].astype('float32')

        relaxation_ratios = []
        nonlinearity_ratios = []
        for which_cycle in range(3):
            # Extract the relevant images
            activation = data[5 + which_cycle*16:16+which_cycle*16, :, :].copy()
            # How much brighter did the image get during 488 nm illumination?
            photoswitching = activation[-2, :, :] - activation[0, :, :]
            # How much dimmer did the sample get after a light-free interval?
            relaxation = activation[-2, :, :] - activation[-1, :, :]
            # How "curved" was the activation curve? If it's a straight line,
            # the average of the intermediate points should equal the value of
            # the exterior points. How much do we deviate?
            nonlinearity = (activation[1:-2, :, :].mean(axis=0) -
                            activation[(0, -2), :, :].mean(axis=0))
            # Inspection of intermediate state is the soul of debugging:
            imwrite(temp_dir /
                    ('1_' + basename + '_activation_%i.tif'%which_cycle),
                    activation, imagej=True)
            imwrite(temp_dir /
                    ('2_' + basename + '_photoswitching_%i.tif'%which_cycle),
                    photoswitching, imagej=True)
            imwrite(temp_dir /
                    ('3_' + basename + '_relaxation_%i.tif'%which_cycle),
                    relaxation, imagej=True)
            imwrite(temp_dir /
                    ('4_' + basename + '_nonlinearity_%i.tif'%which_cycle),
                    nonlinearity, imagej=True)

            # Calculate quantities that should be useful for inferring pH
            norm = np.clip(gaussian_filter(photoswitching, sigma=2), 1, None)
            relaxation_ratio = gaussian_filter(relaxation, sigma=2) / norm
            nonlinearity_ratio = gaussian_filter(nonlinearity, sigma=2) / norm
            imwrite(temp_dir /
                    ('5_' + basename +'_relaxation_ratio_%i.tif'%which_cycle),
                    relaxation_ratio, imagej=True)
            imwrite(temp_dir /
                    ('6_' + basename +'_nonlinearity_ratio_%i.tif'%which_cycle),
                    nonlinearity_ratio, imagej=True)
            relaxation_ratios.append(
                relaxation_ratio[slice_y, slice_x].mean())
            nonlinearity_ratios.append(
                nonlinearity_ratio[slice_y, slice_x].mean())
        print("Relaxation ratios for", basename + ':')
        for x in relaxation_ratios: print('', x)
        print("Nonlinearity ratio for", basename + ':')
        for x in nonlinearity_ratios: print('', x)
        print()

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

main()
