#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
import urllib.request  # For downloading raw data
import zipfile         # For unzipping downloads
# You can use 'pip' to install these dependencies:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tifffile import imread

input_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'modularity_figure_data')
output_dir = ( # The 'images' directory, two folders up:
    Path(__file__).parents[2] / 'images' / 'modularity_figures')
# Sanity checks:
input_dir.mkdir(exist_ok=True, parents=True)
output_dir.mkdir(exist_ok=True)

def main():
    data_1 = load_tif(input_dir / 'LigA-cpVenus_before_NAD.tif')
    data_2 = load_tif(input_dir / 'LigA-cpVenus_after_1mM_NAD.tif')

    t1 = 1e-6 * decode_timestamps(data_1)['microseconds']
    t2 = 1e-6 * decode_timestamps(data_2)['microseconds']
    
    print(data_1.shape, data_2.shape)
    slices = (slice(77, 104), slice(512, -512), slice(512, -512))
    data_1 = data_1[slices].astype('float32')
    data_2 = data_2[slices].astype('float32')
    t1, t2 = t1[slices[0]], t2[slices[0]]
    t1, t2 = t1 - t1[5], t2 - t2[5]

    print("Saving figure...", end='')
    fig = plt.figure(figsize=(6, 3))
    ax = plt.axes([0.1, 0.15, 0.87, 0.8])
    ax.plot([0, 0], color='gray', alpha=0, label='LigA-cpVenus')
    ax.plot(t1, normalize(data_1.mean(axis=(1, 2))),
             '.-', label="   before NAD$^+$")
    ax.plot(t2, normalize(data_2.mean(axis=(1, 2))),
             '.-', label="   after 1mM NAD$^+$")
    ax.add_patch(Rectangle( # Show when illumination is on:
        (0, 0), t1[0], 1.02,
        fill=True, linewidth=0, color=(0.3, 0, 1, 0.1)
        ))
    ax.grid('on')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized fluorescent signal')
    plt.savefig(output_dir / "LigA_cpVenus_relaxes.png", dpi=400)
##    plt.show()
    print(" done.")
    
def load_tif(image_data_filename):
    if not image_data_filename.is_file():
        print("The expected data file:")
        print(image_data_filename)
        print("...isn't where we expect it.\n")
        print(" * Let's try to unzip it...")
        zipped_data_filename = input_dir / "LigA-cpVenus_data.zip"
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
    data = imread(image_data_filename)
    print("Successfully loaded data.")
    print("Data shape:", data.shape)
    print("Data dtype:", data.dtype)
    print()
    return data

def download_data(filename):
    url = "https://zenodo.org/record/5819006/files/LigA-cpVenus_data.zip"
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
