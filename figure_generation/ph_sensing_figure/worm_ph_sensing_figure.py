#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
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

input_dir = Path.cwd() / '1_input'
temp_dir = Path.cwd() / 'intermediate_output'
output_dir = Path.cwd()
# Sanity checks:
input_dir.mkdir(exist_ok=True)
temp_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

def main():
    # Load our worm biosensor images and parse the timestamps:
    data = load_data()
    print("Input data:  ", data.shape, data.dtype)
    
    timestamps = 1e-6*decode_timestamps(data)['microseconds'].astype('float64')
    
    # Crop, flip, convert to float:
    data = data[:, 1776:162:-1, 1329:48:-1
                ].transpose((0, 2, 1)).astype('float32')
    print("Cropped data:", data.shape, data.dtype)

    for which_cycle in range(3):
        # Extract the relevant images
        activation = data[5 + which_cycle*16:16+which_cycle*16, :, :].copy()
        # Add artificial patches with carefully chosen dynamics to give an
        # effective "colorbar"
        # These numbers are calculated in 'worm_ph_calibration.py'
        relaxation_colorbar = (# pH 8, 7.5, 7, 6.5, 6, <6 w/2x interpolation
            (0.774, 0.464, 0.351, 0.264, 0.191, 0.127, 0.068, 0.0147, -0.035, -0.082),
            (1.08, 0.748, 0.569, 0.432, 0.318, 0.218, 0.127, 0.0436, -0.035, -0.1065),
            (1.33, 1.11, 0.919, 0.740, 0.572, 0.415, 0.267, 0.125, -0.011, -0.14),
            )[which_cycle]
        nonlinearity_colorbar = (# pH 8, 7.5, 7, 6.5, 6, <6 w/2x interpolation
            (0.3763, 0.355, 0.333, 0.312, 0.29, 0.2685, 0.247, 0.225, 0.204, 0.183),
            (0.3763, 0.355, 0.333, 0.312, 0.29, 0.2685, 0.247, 0.225, 0.204, 0.183),
            (0.3763, 0.355, 0.333, 0.312, 0.29, 0.2685, 0.247, 0.225, 0.204, 0.183),
            )[which_cycle]
        a_max = activation.max()
        for patch in range(10):
            sl_y, sl_x = slice(-100, -20), slice(-(patch+2)*60, -(patch+1)*60)
            activation[ 0, sl_y, sl_x] = 0
            activation[-2, sl_y, sl_x] = a_max
            activation[-1, sl_y, sl_x
                       ] = a_max * (1 - relaxation_colorbar[patch])
            activation[1:-2, sl_y, sl_x
                       ] = a_max*(0.5+nonlinearity_colorbar[patch])
        # How much brighter did the image get during 488 nm illumination?
        photoswitching = activation[-2, :, :] - activation[0, :, :]
        # How much dimmer did the sample get after a light-free interval?
        relaxation = activation[-2, :, :] - activation[-1, :, :]
        # How "curved" was the activation curve? If it's a straight line,
        # the average of the intermediate points should equal the value of
        # the exterior points. How much do we deviate?
        nonlinearity = (activation[1:-2, :, :].mean(axis=0) -
                        activation[(0, -2), :, :].mean(axis=0))
        imwrite(temp_dir / ('1_activation_%i.tif'%which_cycle),
                activation, imagej=True)
        imwrite(temp_dir / ('2_photoswitching_%i.tif'%which_cycle),
                photoswitching, imagej=True)
        imwrite(temp_dir / ('3_relaxation_%i.tif'%which_cycle),
                relaxation, imagej=True)
        imwrite(temp_dir / ('4_nonlinearity_%i.tif'%which_cycle),
                nonlinearity, imagej=True)

        # Calculate quantities that should be useful for inferring pH
        norm = np.clip(gaussian_filter(photoswitching, sigma=1.5), 1, None)
        relaxation_ratio = gaussian_filter(relaxation, sigma=1.5) / norm
        nonlinearity_ratio = gaussian_filter(nonlinearity, sigma=1.5) / norm
        imwrite(temp_dir / ('5_relaxation_ratio_%i.tif'%which_cycle),
                relaxation_ratio, imagej=True)
        imwrite(temp_dir / ('6_nonlinearity_ratio_%i.tif'%which_cycle),
                nonlinearity_ratio, imagej=True)

        # Convert from ratios to pH via calibration data:
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
                    domain=[0.04702987, 1.2842437 ])
            return fit(rr).astype(rr.dtype)

        def nonlinearity_ratio_to_ph(nr):
            fit = np.polynomial.Polynomial(
                [7.04, 0.98], domain=[0.209 , 0.378])
            return fit(nr).astype(nr.dtype)

        relaxation_ph   =   relaxation_ratio_to_ph(
              relaxation_ratio, which_cycle)
        nonlinearity_ph = nonlinearity_ratio_to_ph(
            nonlinearity_ratio)
        imwrite(temp_dir / ('7_relaxation_ph_%i.tif'%which_cycle),
                relaxation_ph, imagej=True)
        imwrite(temp_dir / ('8_nonlinearity_ph_%i.tif'%which_cycle),
                nonlinearity_ph, imagej=True)
        
        # Smooth and color-merge photoswitching/relaxation or
        # photoswitching/nonlinearity
        luminance = adjust_contrast(
            np.log(np.clip(gaussian_filter(photoswitching, sigma=5), 1, None)),
            2, 7)[..., np.newaxis]
        hue_1 = seminormalized_turbo(
            adjust_contrast(relaxation_ph, 5.75, 8.3))
        imwrite(temp_dir / ('7_relaxation_ratio_overlay_%i.tif'%which_cycle),
                (hue_1 * luminance * 255).astype(np.uint8))

        hue_2 = seminormalized_turbo(
            adjust_contrast(nonlinearity_ph, 5.75, 8.3))
        imwrite(temp_dir / ('8_nonlinearity_ratio_overlay_%i.tif'%which_cycle),
                (hue_2 * luminance * 255).astype(np.uint8))
    
        # Output annotated png
        for hue, name in (hue_1, '2_relaxation'), (hue_2, '3_activation'):
            fig = plt.figure(figsize=(1, 1*(hue.shape[0]/hue.shape[1])))
            ax = plt.axes([0, 0, 1, 1])
            ax.imshow(hue * luminance, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.add_patch(Rectangle(
                (945, 1175), 615, 95,
                fill=False, linewidth=0.9,
                color=(0, 0, 0)))
            ax.text(950, 1170, "pH measured via %s rate"%name[2:],
                fontdict={'color': (0.5, 0.5, 0.5),
                          'weight': 'bold',
                          'size': 1.5})
            fontdict = {'color': (0, 0, 0),
                        'weight': 'bold',
                        'size': 1,
                        'horizontalalignment': 'center'}
            ax.text( 984, 1205, "<6",   fontdict=fontdict)
            ax.text(1044, 1205, "6",   fontdict=fontdict)
            ax.text(1164, 1205, "6.5", fontdict=fontdict)
            ax.text(1284, 1205, "7",   fontdict=fontdict)
            ax.text(1404, 1205, "7.5", fontdict=fontdict)
            ax.text(1524, 1205, "8",   fontdict=fontdict)
            plt.savefig(output_dir / ("%s_overlay_%i.png"%(name, which_cycle)),
                        dpi=800)
            plt.close(fig)

        # Extract the relevant images, smooth them a little so it won't
        # look weird when downsampled to png/gif:
        time_slice = slice(4 + which_cycle*16, 16+which_cycle*16)
        cycle = gaussian_filter(data[time_slice, :, :], sigma=(0, 2, 2))
        cycle -= cycle[1, :, :]
        cycle_timestamps = timestamps[time_slice]
        cycle_timestamps -= cycle_timestamps[-2]
        imwrite(temp_dir / ('9_switching_cycle_%i.tif'%which_cycle),
                np.expand_dims(cycle, 1), imagej=True)

        im = np.zeros((data.shape[1], data.shape[2], 3)) # RGB image, 0<=im<=1
        # Output annotated png for animation.
        for which_frame in range(cycle.shape[0]):
            fig = plt.figure(figsize=(8, 8*(cycle.shape[1]/cycle.shape[2])))
            ax1 = plt.axes([0, 0, 1, 1])
            im[:, :, 1] = adjust_contrast(cycle[which_frame], 0, 450)
            ax1.imshow(im, interpolation='nearest')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.text(100, 100, "Sensor",
                fontdict={'color': (0, 1, 0),
                          'weight': 'bold',
                          'size': 32})
            ax1.text(
                1200, 650,
                "t=%5ss"%('%0.2f'%(cycle_timestamps[which_frame])),
                fontdict={'color': (0, 1, 0),
                          'weight': 'bold',
                          'size': 20,
                          'family': 'monospace',})
            ax2 = plt.axes([0.3, 0.095, 0.62, 0.35], facecolor=(0.8, 0.8, 0.8))
            ax3 = ax2.twinx()
            ax1.add_patch(Rectangle(
                (611, 215), 29, 29, # Oversized, so you can see it
                fill=False, linewidth=4, linestyle=(0, (0.3, 0.3)),
                color=(0, 0, 1)))
            ax1.add_patch(Rectangle(
                (1182, 232), 21, 21, # Oversized, so you can see it
                fill=False, linewidth=2,
                color=(1, 1, 0)))
            box_1_photons = to_photoelectrons(
                cycle[:, 225:225+9, 621:621+9].mean(axis=(1, 2)), offset=0)
            box_2_photons = to_photoelectrons(
                cycle[:, 238:238+9, 1188:1188+9].mean(axis=(1, 2)), offset=0)
            ax2.plot(cycle_timestamps[:-1], box_1_photons[:-1],
                     marker='.', markersize=7,
                     linewidth=1, linestyle=(0, (1, 1)),
                     color=(0, 0, 1))
            t = np.linspace(cycle_timestamps[-2], cycle_timestamps[-1], 100)
            A = box_1_photons[-2]
            T = -t[-1]/np.log(box_1_photons[-1]/box_1_photons[-2])
            ax2.plot(t, A * np.exp(-t/T),
                     marker=None,
                     linewidth=1, linestyle=(0, (1, 1)),
                     color=(0, 0, 1),
                     label="Slow switching (low pH)")
            ax2.plot(cycle_timestamps[-1], box_1_photons[-1],
                     marker='.', markersize=7,
                     linewidth=1, linestyle=(0, (1, 1)),
                     color=(0, 0, 1))
            ax3.plot(cycle_timestamps[:-1], box_2_photons[:-1],
                     marker='.', markersize=7,
                     linewidth=1,
                     color=(1, 1, 0))
            A = box_2_photons[-2]
            T = -t[-1]/np.log(box_2_photons[-1]/box_2_photons[-2])
            ax3.plot(t, A * np.exp(-t/T),
                     marker=None,
                     linewidth=1,
                     color=(1, 1, 0),
                     label="Fast switching (high pH)")
            ax3.plot(cycle_timestamps[-1], box_2_photons[-1],
                     marker='.', markersize=7,
                     linewidth=1,
                     color=(1, 1, 0))
            # Misc plot tweaks:
            ax2.set_xlim(cycle_timestamps.min() - 0.1, 18)
            ax2.set_ylim(-26.5, 1650)
            ax3.set_ylim(-6.7, 415)
            ax2.grid('on', alpha=0.2)
            ax2.tick_params(labelcolor='white')
            ax2.tick_params('y', labelcolor='blue')
            ax3.tick_params('y', labelcolor='yellow')
            ax2.set_xlabel("Time (s)", color='white', weight='bold')
            ax2.set_ylabel("Photons per pixel", color='blue', weight='bold')
            ax3.set_ylabel("Photons per pixel", color='yellow', weight='bold')
            h2, l2 = ax2.get_legend_handles_labels()
            h3, l3 = ax3.get_legend_handles_labels()
            ax2.legend(h2+h3, l2+l3, facecolor=(0.8, 0.8, 0.8),
                       fontsize=8, loc=(0.53, 0.4))
            # Emphasize the current time graphically:
            ax2.axvline(cycle_timestamps[which_frame])
            # Show when the 405 nm and 488 nm illumination is on:
            for pt in np.linspace(cycle_timestamps[0] - 0.15,
                                  cycle_timestamps[1] - 0.15, 10):
                ax2.add_patch(Rectangle(
                    (pt, 250),
                    (cycle_timestamps[1] - cycle_timestamps[0]) / 30, 900,
                    fill=True, linewidth=0, color=(0.3, 0, 1, 0.22)))
            ax2.text(
                0.1 + cycle_timestamps[0], 1350,
                "405 nm\nillumination",
                fontdict={'color': (0.5, 0, 1),
                          'weight': 'bold',
                          'size': 8.5,})
            ax2.plot((cycle_timestamps[0] - 0.15, cycle_timestamps[1] - 0.15),
                     (1275, 1275), color=(0.5, 0, 1), marker='|')
            for ct in cycle_timestamps:
                ax2.add_patch(Rectangle(
                    (ct - 0.025, 200), 0.05, 1000,
                    fill=True, linewidth=0, color=(0, 1, 1, 0.8)))
            ax2.text(
                0.5 + cycle_timestamps[1], 20,
                "488 nm illumination",
                fontdict={'color': (0, .7, .7),
                          'weight': 'bold',
                          'size': 8.5,})
            txt = "488 nm illumination"
            if which_cycle == 2:
                txt = "488 nm\nillumination"
            ax2.text(
                -2 + cycle_timestamps[-1], 200,
                txt,
                fontdict={'color': (0, .7, .7),
                          'weight': 'bold',
                          'size': 8.5,
                          'verticalalignment': 'top'})
            ax2.plot((cycle_timestamps[1], cycle_timestamps[-2]), (150, 150),
                     color=(0, 0.7, 0.7), marker='|')
            ax2.text(
                (cycle_timestamps[-2] + cycle_timestamps[-1]) / 2, 1450,
                "No illumination",
                fontdict={'color': (0, 0, 0),
                          'weight': 'bold',
                          'size': 8.5,
                          'horizontalalignment': 'center',})
            ax2.plot((cycle_timestamps[-2] + 0.1, cycle_timestamps[-1] - 0.1),
                     (1400, 1400), color='black', marker='|')
            if which_frame < cycle.shape[0] - 2:
                plt.savefig(
                    temp_dir /
                    ("data_frame_%i_%03i.png"%(which_cycle, which_frame)),
                    dpi=100)
            elif which_frame == cycle.shape[0] - 2: # Repeat the penultimate
                for x in range(which_frame, which_frame+5):
                    plt.savefig(
                        temp_dir /
                        ("data_frame_%i_%03i.png"%(which_cycle, x)),
                        dpi=100)
            elif which_frame == cycle.shape[0] - 1: # Repeat the final
                for x in range(which_frame+4, which_frame+7):
                    plt.savefig(
                        temp_dir /
                        ("data_frame_%i_%03i.png"%(which_cycle, x)),
                        dpi=100)
            plt.close(fig)

        # Animate the frames into a gif:
        palette = temp_dir / "palette3.png"
        filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
        print("Converting pngs to gif...", end=' ')
        convert_command_1 = [
            'ffmpeg',
            '-f', 'image2',
            '-i', str(temp_dir / ('data_frame_%i_%%3d.png'%(which_cycle))),
            '-vf', filters + ",palettegen",
            '-y', palette]
        convert_command_2 = [
            'ffmpeg',
            '-framerate', '8',
            '-f', 'image2',
            '-i', str(temp_dir / ('data_frame_%i_%%3d.png'%(which_cycle))),
            '-i', palette,
            '-lavfi', filters + " [x]; [x][1:v] paletteuse",
            '-y', str(output_dir / ("1_data_animation_%i.gif"%(which_cycle)))]
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

def seminormalized_turbo(x):
    """Flat-luminance version of
    ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
    """
    turbo = plt.get_cmap('turbo')(x)[..., :-1]
    luminance = (0.2126 * turbo[..., 0] +
                 0.7152 * turbo[..., 1] +
                 0.0722 * turbo[..., 2])
    norm_turbo = turbo / np.expand_dims(luminance, -1)
    return np.clip(0.5 * norm_turbo, 0, 1) # Not-quite-flat luminance

def adjust_contrast(img, min_, max_, gamma=1):
    """Like setting "minimum" and "maximum" image contrast in ImageJ

    Output image intensity will range from zero to one.
    Non-quantitative, just useful for display. Be careful and honest if
    you use gamma != 1 (presumably to show higher dynamic range)
    """
    img = np.clip(img, min_, max_)
    img -= min_
    img /= max_ - min_
    img = img**gamma
    img = np.clip(img, 0, 1) # Fix small numerical errors from the previous line
    return img

def to_photoelectrons(camera_counts, offset=100):
    """A good approximation for the pco edge 4.2
    """
    return (camera_counts - offset) * 0.46

main()
