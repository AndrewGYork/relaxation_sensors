#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tifffile import imread, imwrite # v2020.6.3 or newer

input_dir = Path.cwd() / '0_calibration_data'
temp_dir = Path.cwd() / 'intermediate_calibration_output'
output_dir = Path.cwd()
# Sanity checks:
assert input_dir.is_dir()
temp_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

def main():
    # Load our bead biosensor calibration images and parse the timestamps:
    # Acquired by Maria on 2020_12_01
    relaxation_ratios = []
    relaxation_times = []
    nonlinearity_ratios = []
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

        relaxation_ratios.append([])
        relaxation_times.append([])
        nonlinearity_ratios.append([])
        for which_cycle in range(3):
            # Extract the relevant images
            s = slice(5 + which_cycle*16, 16+which_cycle*16)
            activation = data[s, :, :].copy()
            activation_ts = timestamps[s].copy()
            # How much brighter did the image get during 488 nm illumination?
            photoswitching = activation[-2, :, :] - activation[0, :, :]
            # How much dimmer did the sample get after a light-free interval?
            relaxation = activation[-2, :, :] - activation[-1, :, :]
            relaxation_time = activation_ts[-1] - activation_ts[-2]
            # How "curved" was the activation curve? If it's a straight line,
            # the average of the intermediate points should equal the value of
            # the exterior points. How much do we deviate?
            nonlinearity = (activation[1:-2, :, :].mean(axis=0) -
                            activation[(0, -2), :, :].mean(axis=0))
            # Inspection of intermediate state is the soul of debugging:
##            imwrite(temp_dir /
##                    ('1_' + basename + '_activation_%i.tif'%which_cycle),
##                    activation, imagej=True)
##            imwrite(temp_dir /
##                    ('2_' + basename + '_photoswitching_%i.tif'%which_cycle),
##                    photoswitching, imagej=True)
##            imwrite(temp_dir /
##                    ('3_' + basename + '_relaxation_%i.tif'%which_cycle),
##                    relaxation, imagej=True)
##            imwrite(temp_dir /
##                    ('4_' + basename + '_nonlinearity_%i.tif'%which_cycle),
##                    nonlinearity, imagej=True)

            # Calculate quantities that should be useful for inferring pH
            norm = np.clip(gaussian_filter(photoswitching, sigma=2), 1, None)
            relaxation_ratio = gaussian_filter(relaxation, sigma=2) / norm
            nonlinearity_ratio = gaussian_filter(nonlinearity, sigma=2) / norm
##            imwrite(temp_dir /
##                    ('5_' + basename +'_relaxation_ratio_%i.tif'%which_cycle),
##                    relaxation_ratio, imagej=True)
##            imwrite(temp_dir /
##                    ('6_' + basename +'_nonlinearity_ratio_%i.tif'%which_cycle),
##                    nonlinearity_ratio, imagej=True)
            relaxation_ratios[-1].append(
                relaxation_ratio[slice_y, slice_x].mean())
            relaxation_times[-1].append(relaxation_time)
            nonlinearity_ratios[-1].append(
                nonlinearity_ratio[slice_y, slice_x].mean())
        print("Relaxation ratios for", basename + ':')
        for x in relaxation_ratios[-1]: print('', x)
        print("Relaxation times for", basename + ':')
        for x in relaxation_times[-1]: print('', x)
        print("Nonlinearity ratio for", basename + ':')
        for x in nonlinearity_ratios[-1]: print('', x)
        print()

##    print(relaxation_ratios)
##    relaxation_ratios = [[0.00094347354, 0.009571701, 0.047029868], [0.03750617, 0.08096879, 0.18749335], [0.15792592, 0.28675514, 0.56622875], [0.39226782, 0.6284052, 0.98587596], [0.8210087, 1.0815612, 1.2842437]]
##    nonlinearity_ratios = [[0.2088546, 0.21404004, 0.21547164], [0.23996998, 0.24407461, 0.24433099], [0.29617807, 0.29908523, 0.29870716], [0.32550022, 0.33213314, 0.33231893], [0.37825266, 0.38069993, 0.37796608]]

    relaxation_colorbar = (# pH 8, 7.5, 7, 6.5, 6, <6 w/2x interpolation
        (0.774, 0.464, 0.351, 0.264, 0.191, 0.127, 0.068, 0.0147, -0.035, -0.082),
        (1.08, 0.748, 0.569, 0.432, 0.318, 0.218, 0.127, 0.0436, -0.035, -0.1065),
        (1.33, 1.11, 0.919, 0.740, 0.572, 0.415, 0.267, 0.125, -0.011, -0.14),
        )
    nonlinearity_colorbar = (# pH 8, 7.5, 7, 6.5, 6, <6 w/2x interpolation
        (0.3763, 0.355, 0.333, 0.312, 0.29, 0.2685, 0.247, 0.225, 0.204, 0.183),
        (0.3763, 0.355, 0.333, 0.312, 0.29, 0.2685, 0.247, 0.225, 0.204, 0.183),
        (0.3763, 0.355, 0.333, 0.312, 0.29, 0.2685, 0.247, 0.225, 0.204, 0.183),
        )

    def relaxation_ratio_to_ph(rr, which_cycle):
        if which_cycle == 0:
            fit = np.polynomial.Polynomial(
                [ 7.64391097,  0.89941276, -0.56161918],
                domain=[0.00094347, 0.8210087 ])
        elif which_cycle == 1:
            fit = np.polynomial.Polynomial(
                [ 7.46136788,  0.91604103, -0.40289993],
                domain=[0.0095717, 1.0815612])
        elif which_cycle == 2:
            fit = np.polynomial.Polynomial(
                [ 7.14145164,  0.9226259 , -0.11188533],
                domain=[0.04702987, 1.284237 ])
        return fit(rr)

    def nonlinearity_ratio_to_ph(nr):
        fit = np.polynomial.Polynomial(
            [7.04, 0.98], domain=[0.209 , 0.378])
        return fit(nr)

    relaxation_ratios = np.asarray(relaxation_ratios)
    nonlinearity_ratios = np.asarray(nonlinearity_ratios)
    ph_range = np.array((6, 6.5, 7, 7.5, 8))
    plt.figure()
    plt.suptitle("Good-enough fits")
    ax1 = plt.subplot(1, 2, 1)
    for which_cycle in range(3):
        r = relaxation_ratios[:, which_cycle]
        r_fit = np.polynomial.Polynomial.fit(r, ph_range, deg=2)
        r_fine = np.linspace(r[0], r[-1], 1000)
        print("Relaxation ratio fit for cycle %i:\n"%which_cycle, repr(r_fit))
        ax1.plot(r, ph_range, 'x', color='C%i'%which_cycle,)
        ax1.plot(relaxation_colorbar[which_cycle],
                 r_fit(np.array(relaxation_colorbar[which_cycle])),
                 'o', color='C%i'%which_cycle, markersize=3)
        ax1.plot(r_fine, r_fit(r_fine), '--',
                 color='C%i'%which_cycle, label="Cycle %i"%which_cycle)
        ax1.plot(r_fine, relaxation_ratio_to_ph(r_fine, which_cycle), '--',
                 color='C%i'%which_cycle)
    ax1.grid('on')
    ax1.set_xlabel("Relaxation ratio")
    ax1.set_ylabel("pH")
    ax1.legend()
    print()
    ax2 = plt.subplot(1, 2, 2)
    for which_cycle in range(3):
        n = nonlinearity_ratios[:, which_cycle]
        n_fit = np.polynomial.Polynomial.fit(n, ph_range, deg=1)
        n_fine = np.linspace(n[0], n[-1], 1000)
        print("Nonlinearity ratio fit for cycle %i:\n"%which_cycle, repr(n_fit))
        ax2.plot(n, ph_range, 'x', color='C%i'%which_cycle)
        ax2.plot(nonlinearity_colorbar[which_cycle],
                 nonlinearity_ratio_to_ph(
                     np.array(nonlinearity_colorbar[which_cycle])),
                 'o', color='C%i'%which_cycle, markersize=3)
        ax2.plot(n_fine, n_fit(n_fine), '--',
                 color='C%i'%which_cycle, label="Cycle %i"%which_cycle)
        ax2.plot(n_fine, nonlinearity_ratio_to_ph(n_fine), '--',
                 color='C%i'%which_cycle)
    ax2.grid('on')
    ax2.set_xlabel("Nonlinearity ratio")
    ax2.legend()
    print()
    plt.show()

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
