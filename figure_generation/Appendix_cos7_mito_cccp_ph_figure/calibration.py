#!/usr/bin/python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf

# Bottom line up front: a rough mapping from pH to relaxation ratio
average_ratios = np.array(((6.5, 0.09),
                           (7.0, 0.12),
                           (7.5, 0.21),
                           (8.0, 0.49)))
# How we got there
input_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'mito_ph_data')

filenames = ('6p5', '7p0', '7p5', '8p0')
data_list = []
relaxation_ratio_list = []
for fn in filenames:
    data = tf.imread(input_dir / ('calibration_ph_' + fn + '.tif'))
    data = data.astype('float64')
    data = data.reshape(30, 14, 100, 100)
    data = data.sum(axis=(-1, -2))
    # First relaxation didn't pre-use 405, so we skip it
    data = data[1:, :]
    # Normalized the activation curve from 0 to 1:
    data = data - data[:, 0:1]
    data = data / data[:, 9:10]
    data_list.append(data)
    # Calculate relaxation ratios compared to the normalized activation curve:
    relaxation_ratio = data[:, 9] - data[:, 10]
    relaxation_ratio_list.append(relaxation_ratio)

# Hopefully the relaxation curves for a given pH resemble each other
# more than they resemble other pHs:
plt.figure()
for i, (fn, data, relaxation_ratio) in enumerate(zip(filenames,
                                                     data_list,
                                                     relaxation_ratio_list)):
    for t in range(data.shape[0]):
        plt.plot(data[t, :], '.-', c='C%i'%i)
    plt.grid('on')

# How did relaxation ratio drift over time?
# Looks like some thermal drift in the pH 7.5 and pH 8 data, but not
# enough that they become confusable with other pHs:
plt.figure()
for i, (fn, data, relaxation_ratio) in enumerate(zip(filenames,
                                                     data_list,
                                                     relaxation_ratio_list)):
    plt.plot(relaxation_ratio, '.-', c='C%i'%i)
    plt.axhline(average_ratios[i, 1], c='C%i'%i)
    plt.grid('on')

plt.show()
