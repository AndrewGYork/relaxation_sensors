#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

def main():
    data_1 = imread('../../../relaxation_sensors_temp_files/modularity_figure_data/before.tif')
    data_2 = imread('../../../relaxation_sensors_temp_files/modularity_figure_data/after1mMNAD.tif')

    t1 = 1e-6 * decode_timestamps(data_1)['microseconds']
    t2 = 1e-6 * decode_timestamps(data_2)['microseconds']
    
    print(data_1.shape, data_2.shape)
    slices = (slice(77, 104), slice(512, -512), slice(512, -512))
    data_1 = data_1[slices].astype('float32')
    data_2 = data_2[slices].astype('float32')
    t1, t2 = t1[slices[0]], t2[slices[0]]
    t1, t2 = t1 - t1[0], t2 - t2[0]

    plt.figure()
    plt.plot(t1, normalize(data_1.mean(axis=(1, 2))),
             '.-', label="Before NAD")
    plt.plot(t2, normalize(data_2.mean(axis=(1, 2))),
             '.-', label="After 1mM NAD")
    plt.grid('on')
    plt.legend()
    plt.title("Our first relaxation sensor for NAD")
    plt.show()

def normalize(x):
    min_ = x[1:6].mean()
    max_ = x[23:27].mean()
    return (x - min_) / (max_ - min_)

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

if __name__ == '__main__':
    main()