#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
import shutil          # For copying files
import subprocess      # For calling ffmpeg to make animations
import urllib.request  # For downloading raw data
import zipfile         # For unzipping downloads
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tifffile import imread, imwrite # v2020.6.3 or newer

# You'll need ffmpeg installed to make animations. If you can't type
# "ffmpeg --version" at the command prompt without an error, my code
# will probably fail too.

input_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'worm_ph_data')
temp_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'autofluorescence_figure')
output_dir = ( # The 'images' directory, two folders up:
    Path(__file__).parents[2] / 'images' / 'autofluorescence_figure')
# Sanity checks:
input_dir.mkdir(exist_ok=True, parents=True)
temp_dir.mkdir(exist_ok=True, parents=True)
output_dir.mkdir(exist_ok=True)

def main():
    # Load our worm biosensor images and parse the timestamps:
    data = load_data()
    
    timestamps = 1e-6*decode_timestamps(data)['microseconds'].astype('float64')
    
    # Crop, flip, convert to float:
    data = data[:, 1776:162:-1, 1329:48:-1
                ].transpose((0, 2, 1)).astype('float32')
    print("Cropped data:", data.shape, data.dtype)

    # Extract the relevant images to separate signal from background:
    # minimum and maximum photoactivation
    deactivated = data[(5, 21, 37), :, :] # The dimmest image(s)
    imwrite(temp_dir / '1_deactivated.tif', deactivated, imagej=True)
    activated = data[(14, 30, 46), :, :] # The brightest image(s)
    imwrite(temp_dir / '2_activated.tif', activated, imagej=True)
    photoswitching = activated - deactivated
    imwrite(temp_dir / '3_photoswitching.tif', photoswitching, imagej=True)
    
    # Estimate the contribution of the photoswitching fluorophores to
    # the deactivated image by hand-tuning a subtraction. This is
    # basically low-effort spectral unmixing:
    background = deactivated - 0.53 * photoswitching
    imwrite(temp_dir / '4_background.tif', background, imagej=True)

    # ~3.4 seconds elapse between deactivated and activated images:
    intervals = timestamps[(14, 30, 46),] - timestamps[(5, 21, 37),]
    print("Time intervals:", intervals)

    # Smooth and color-merge photoswitching/background
    photoswitching = adjust_contrast(photoswitching, 0, 450)
    photoswitching = gaussian_filter(photoswitching, sigma=(0, 2, 2))
    background = adjust_contrast(background, 100, 700)
    background = gaussian_filter(background, sigma=(0, 2, 2))
    img = np.zeros((data.shape[1], data.shape[2], 3)) # RGB image
    img[:, :, 0] = background[0, :, :]
    img[:, :, 1] = photoswitching[0, :, :]
    img[:, :, 2] = background[0, :, :]
    just_magenta, just_green = np.array([1, 0, 1]), np.array([0, 1, 0])
    imwrite(temp_dir / '5_background_c.tif',
            (img * just_magenta * 255).astype(np.uint8))
    imwrite(temp_dir / '6_fluorophores_c.tif',
            (img * just_green * 255).astype(np.uint8))
    imwrite(temp_dir / '7_overlay.tif',
            (img * 255).astype(np.uint8))

    # Output annotated png for animation.
    for i in range(4):
        fig = plt.figure(figsize=(1, 1*(img.shape[0]/img.shape[1])))
        ax = plt.axes([0, 0, 1, 1])
        if i == 0:
            x = img # Overlay
        elif i == 1:
            x = img * just_green
        elif i == 2:
            x = img # Overlay
        elif i == 3:
            x = img * just_magenta
        ax.imshow(x, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if i != 3:
            ax.text(900, 1000, "Sensor",
                fontdict={'color': (0, 1, 0),
                          'weight': 'bold',
                          'size': 4})
        if i != 1:
            ax.text(900, 1100, "Background",
                fontdict={'color': (1, 0, 1),
                          'weight': 'bold',
                          'size': 4})
        plt.savefig(temp_dir / ("overlay_frame_%03i.png"%i), dpi=800)

    # Animate the frames into a gif:
    palette = temp_dir / "palette.png"
    filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
    print("Converting pngs to gif...", end=' ')
    convert_command_1 = [
        'ffmpeg',
        '-f', 'image2',
        '-i', str(temp_dir / 'overlay_frame_%3d.png'),
        '-vf', filters + ",palettegen",
        '-y', palette]
    convert_command_2 = [
        'ffmpeg',
        '-framerate', '0.7',
        '-f', 'image2',
        '-i', str(temp_dir / 'overlay_frame_%3d.png'),
        '-i', palette,
        '-lavfi', filters + " [x]; [x][1:v] paletteuse",
        '-y', str(output_dir / "2_overlay_animation.gif")]
    for convert_command in convert_command_1, convert_command_2:
        try:
            with open(temp_dir / 'conversion_messages.txt', 'wt') as f:
                f.write("So far, everthing's fine...\n")
                f.flush()
                subprocess.check_call(convert_command, stderr=f, stdout=f)
                f.flush()
            (temp_dir / 'conversion_messages.txt').unlink()
        except: # This is unlikely to be platform independent :D
            print("GIF conversion failed. Is ffmpeg installed?")
            raise
    print('done.')

    # Copy some of the frames into the output directory:
    for which_frame in (1, 2, 3):
        src =   temp_dir / ('overlay_frame_%03i.png'%(which_frame))
        dst = output_dir / ('overlay_frame_%03i.png'%(which_frame))
        shutil.copyfile(src, dst)

    # Extract images for one photoswitching cycle. Smooth them a little,
    # since we'll be resampling the image anyway for the figure.
    cycle = gaussian_filter(data[4:15, :, :], sigma=(0, 1, 1))
    cycle_timestamps = timestamps[4:15] - timestamps[5]
    imwrite(temp_dir / '8_switching_cycle.tif',
            np.expand_dims(cycle, 1), imagej=True)

    # Output annotated png for animation.
    for i in range(cycle.shape[0]):
        fig = plt.figure(figsize=(8, 8*(cycle.shape[1]/cycle.shape[2])))
        ax1 = plt.axes([0, 0, 1, 1])
        ax1.imshow(cycle[i], interpolation='nearest', cmap=plt.cm.gray,
                  vmin=100, vmax=625)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.text(100, 100, "Raw data",
            fontdict={'color': (1, 1, 1),
                      'weight': 'bold',
                      'size': 32})
        # Show the current time textually
        ax1.text(
            1200, 650,
            "t=%5ss"%('%0.2f'%(cycle_timestamps[i])),
            fontdict={'color': (1, 1, 1),
                      'weight': 'bold',
                      'size': 20,
                      'family': 'monospace',})
        # Make an inset plot to show intensity vs. time for two regions:
        # one with a lot of photoswitching (and not much background),
        # and one with a lot of background (and not much photoswitching)
        ax2 = plt.axes([0.5, 0.095, 0.47, 0.35])
        # Highlight the two "source" regions in the main plot
        ax1.add_patch(Rectangle(
            (279, 789), 21, 21,
            fill=False, linewidth=3, linestyle=(0, (0.5, 0.5)),
            color=(1, 0, 1)))
        ax1.add_patch(Rectangle(
            (722, 376), 21, 21,
            fill=False, linewidth=2, color=(0, 0.9, 0)))
        # Extract and plot intensity vs. time for these two regions
        box_1_photons = to_photoelectrons(
            cycle[:, 789:789+21, 279:279+21].mean(axis=(1, 2)))
        box_2_photons = to_photoelectrons(
            cycle[:, 376:376+21, 722:722+21].mean(axis=(1, 2)))
        ax2.plot(cycle_timestamps, box_1_photons,
                 marker='.', markersize=7,
                 linewidth=2.5, linestyle=(0, (1, 1)),
                 color=(1, 0, 1),
                 label='High background')
        ax2.plot(cycle_timestamps, box_2_photons,
                 marker='.', markersize=7,
                 linewidth=2.5,
                 color=(0, 0.9, 0),
                 label='Low background')
        ax2.legend(loc=(0.1, 0.02), fontsize=8, ncol=2)
        # Highlight how much (or how little) photoswitching occurs in
        # each trace:
        ax2.plot([cycle_timestamps[-1]*1.08]*2,
                 [box_1_photons.min(), box_1_photons.max()],
                 marker=0, markersize=10, linewidth=2.5,
                 color=(1, 0, 1))
        ax2.plot([cycle_timestamps[-1]*1.12]*2,
                 [box_2_photons.min(), box_2_photons.max()],
                 marker=0, markersize=10, linewidth=2.5,
                 color=(0, 0.9, 0))
        ax2.text(
            3.79, 93,
            "%0.0f%%"%(100 * (box_1_photons.max() - box_1_photons.min()) /
                       box_1_photons.min()),
            fontdict={'color': (1, 0, 1),
                      'weight': 'bold',
                      'size': 10,})
        ax2.text(
            3.98, 65,
            "%0.0f%%"%(100 * (box_2_photons.max() - box_2_photons.min()) /
                       box_2_photons.min()),
            fontdict={'color': (0, 0.5, 0),
                      'weight': 'bold',
                      'size': 10,})
        # Misc plot tweaks:
        ax2.set_xlim(cycle_timestamps.min() - 0.2, 1.35*cycle_timestamps.max())
        ax2.set_ylim(0, 1.2*max(box_1_photons.max(), box_2_photons.max()))
        ax2.grid('on', alpha=0.17)
        ax2.tick_params(labelcolor='white')
        ax2.set_xlabel("Time (s)", color='white', weight='bold')
        ax2.set_ylabel("Photoelectrons per pixel", color='white', weight='bold')
        # Emphasize the current time graphically:
        ax2.axvline(cycle_timestamps[i], alpha=0.2)
        # Show when the 395 nm and 470 nm illumination is on:
        for pt in np.linspace(cycle_timestamps[0] - 0.15,
                              cycle_timestamps[1] - 0.15, 10):
            ax2.add_patch(Rectangle(
                (pt, 40),
                (cycle_timestamps[1] - cycle_timestamps[0]) / 30, 60,
                fill=True, linewidth=0, color=(0.3, 0, 1, 0.22)))
        ax2.text(
            0.7 + cycle_timestamps[0], 22,
            "395 nm\nillumination",
            fontdict={'color': (0.5, 0, 1),
                      'weight': 'bold',
                      'size': 8.5,
                      'horizontalalignment': 'center',})
        for ct in cycle_timestamps:
            ax2.add_patch(Rectangle(
                (ct - 0.025, 30), 0.05, 80,
                fill=True, linewidth=0, color=(0, 0.9, 0.9, 0.6)))
        ax2.text(
            0.6 + cycle_timestamps[1], 22,
            "470 nm illumination",
            fontdict={'color': (0, .7, .7),
                      'weight': 'bold',
                      'size': 8.5,})
        plt.savefig(temp_dir / ("data_frame_%03i.png"%i), dpi=100)

    # Animate the frames into a gif:
    palette = temp_dir / "palette2.png"
    filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
    print("Converting pngs to gif...", end=' ')
    convert_command_1 = [
        'ffmpeg',
        '-f', 'image2',
        '-i', str(temp_dir / 'data_frame_%3d.png'),
        '-vf', filters + ",palettegen",
        '-y', palette]
    convert_command_2 = [
        'ffmpeg',
        '-framerate', '12',
        '-f', 'image2',
        '-i', str(temp_dir / 'data_frame_%3d.png'),
        '-i', palette,
        '-lavfi', filters + " [x]; [x][1:v] paletteuse",
        '-y', str(output_dir / "1_data_animation.gif")]
    for convert_command in convert_command_1, convert_command_2:
        try:
            with open(temp_dir / 'conversion_messages.txt', 'wt') as f:
                f.write("So far, everthing's fine...\n")
                f.flush()
                subprocess.check_call(convert_command, stderr=f, stdout=f)
                f.flush()
            (temp_dir / 'conversion_messages.txt').unlink()
        except: # This is unlikely to be platform independent :D
            print("GIF conversion failed. Is ffmpeg installed?")
            raise
    print('done.')

def load_data():
    image_data_filename = input_dir / "worm_with_ph_sensor.tif"
    if not image_data_filename.is_file():
        print("The expected data file:")
        print(image_data_filename)
        print("...isn't where we expect it.\n")
        print(" * Let's try to unzip it...")
        zipped_data_filename = input_dir / "worm_with_ph_sensor.zip"
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
    url="https://zenodo.org/record/4515109/files/worm_with_ph_sensor.zip"
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

def to_photoelectrons(camera_counts):
    """A good approximation for the pco edge 4.2
    """
    return (camera_counts - 100) * 0.46

main()
