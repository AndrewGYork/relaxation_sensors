#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
import urllib.request  # For downloading raw data
import zipfile         # For unzipping downloads
# You can use 'pip' to install these dependencies:
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
# Sanity check:
input_dir.mkdir(exist_ok=True, parents=True)

def main():
    filenames = ('6p5', '7p0', '7p5', '8p0')
    data_list = []
    relaxation_ratio_list = []
    for fn in filenames:
        data = load_data(fn)
        data = data.astype('float64')
        data = data.reshape(30, 14, 100, 100)
        data = data.sum(axis=(-1, -2))
        # First relaxation didn't pre-use 405, so we skip it
        data = data[1:, :]
        # Normalized the activation curve from 0 to 1:
        data = data - data[:, 0:1]
        data = data / data[:, 9:10]
        data_list.append(data)
        # Calculate relaxation ratios compared to the norm. activation curve:
        relaxation_ratio = data[:, 9] - data[:, 10]
        relaxation_ratio_list.append(relaxation_ratio)

    # Hopefully the relaxation curves for a given pH resemble each other
    # more than they resemble other pHs:
    plt.figure()
    for i, (fn, data, relaxation_ratio) in enumerate(zip(
        filenames, data_list, relaxation_ratio_list)):
        for t in range(data.shape[0]):
            plt.plot(data[t, :], '.-', c='C%i'%i)
        plt.grid('on')

    # How did relaxation ratio drift over time?
    # Looks like some thermal drift in the pH 7.5 and pH 8 data, but not
    # enough that they become confusable with other pHs:
    plt.figure()
    for i, (fn, data, relaxation_ratio) in enumerate(zip(
        filenames, data_list, relaxation_ratio_list)):
        plt.plot(relaxation_ratio, '.-', c='C%i'%i)
        plt.axhline(average_ratios[i, 1], c='C%i'%i)
        plt.grid('on')

    plt.show()

def load_data(fn):
    image_data_filename = input_dir / ('calibration_ph_' + fn + '.tif')
    if not image_data_filename.is_file():
        print("The expected data file:")
        print(image_data_filename)
        print("...isn't where we expect it.\n")
        print(" * Let's try to unzip it...")
        zipped_data_filename = input_dir / "mito_ph_data.zip"
        if not zipfile.is_zipfile(zipped_data_filename):
            print("\n  The expected zipped data file:")
            print(zipped_data_filename)
            print("  ...isn't where we expect it.\n")
            print(" * * Let's try to download it from Zenodo.")
            download_data(zipped_data_filename)
        assert zipped_data_filename.is_file()
        assert zipfile.is_zipfile(zipped_data_filename)
        print(" Unzipping...")
        with zipfile.ZipFile(zipped_data_filename) as zf:
            zf.extract(image_data_filename.name,
                       image_data_filename.parent)
        print(" Successfully unzipped data.\n")
    assert image_data_filename.is_file()
    print("Loading data...")        
    data = tf.imread(image_data_filename)
    print("Data shape:", data.shape)
    print("Data dtype:", data.dtype)
    print()
    return data

def download_data(filename):
    url="https://zenodo.org/record/5816195/files/mito_ph_data.zip"
    u = urllib.request.urlopen(url)
    file_size = int(u.getheader("Content-Length"))
    block_size = 8192
    while block_size * 80 < file_size:
        block_size *= 2
    bar_size = max(1, int(0.5 * (file_size / block_size - 12)))

    print("    Downloading from:")
    print(url)
    print("    Downloading to:")
    print(filename)
    print("    File size: %0.2f MB"%(file_size/2**20))
    print("\nDownloading might take a while, so here's a progress bar:")
    print('0%', "-"*bar_size, '50%', "-"*bar_size, '100%')
    with open(filename, 'wb') as f:
        while True:
            buffer = u.read(block_size)
            if not buffer:
                break
            f.write(buffer)
            print('|', sep='', end='')
    print("\nDone downloading.\n")
    return None


main()
