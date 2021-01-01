#!/usr/bin/python
# Dependencies from the python standard library:
import subprocess
from pathlib import Path
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tifffile import imread, imwrite # v2020.6.3 or newer

input_dir = Path.cwd()
temp_dir = Path.cwd() / 'intermediate_output'
output_dir = Path.cwd()
# Sanity checks:
assert input_dir.is_dir()
temp_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

def main():
    # Load our worm biosensor images and parse the timestamps:
    data = imread(input_dir / '0_data.tif') # Acquired by Maria on 2020_12_01
    print(data.shape, data.dtype)
    
    timestamps = decode_timestamps(data)['microseconds'].astype('float64')
    
    # Crop, flip, convert to float:
    data = data[:, 1776:162:-1, 1329:48:-1
                ].transpose((0, 2, 1)).astype('float32')
    print(data.shape, data.dtype)

    # Extract the relevant images
    which_cycle = 0
    activation = data[5 + which_cycle*16:16+which_cycle*16, :, :]
    # How much brighter did the image get during 488 nm illumination?
    photoswitching = activation[-2, :, :] - activation[0, :, :]
    # How much dimmer did the sample get after a light-free interval?
    relaxation = activation[-2, :, :] - activation[-1, :, :]
    # How "curved" was the activation curve? If it's a straight line,
    # the average of the intermediate points should equal the value of
    # the exterior points. How much do we deviate?
    nonlinearity = (activation[1:-2, :, :].mean(axis=0) -
                    activation[(0, -2), :, :].mean(axis=0))
    imwrite(temp_dir / '1_activation.tif', activation, imagej=True)
    imwrite(temp_dir / '2_photoswitching.tif', photoswitching, imagej=True)
    imwrite(temp_dir / '3_relaxation.tif', relaxation, imagej=True)
    imwrite(temp_dir / '4_nonlinearity.tif', nonlinearity, imagej=True)

    # Calculate quantities that should be useful for inferring pH
    norm = np.clip(gaussian_filter(photoswitching, sigma=1), 1, None)
    relaxation_ratio = gaussian_filter(relaxation, sigma=1) / norm
    nonlinearity_ratio = gaussian_filter(nonlinearity, sigma=1) / norm
    imwrite(temp_dir / '5_relaxation_ratio.tif', relaxation_ratio, imagej=True)
    imwrite(temp_dir / '6_nonlinearity_ratio.tif', nonlinearity_ratio,
            imagej=True)
    
    

##    # Smooth and color-merge photoswitching/background
##    photoswitching = adjust_contrast(photoswitching, 0, 450)
##    photoswitching = gaussian_filter(photoswitching, sigma=(0, 2, 2))
##    background = adjust_contrast(background, 100, 700)
##    background = gaussian_filter(background, sigma=(0, 2, 2))
##    img = np.zeros((data.shape[1], data.shape[2], 3)) # RGB image
##    img[:, :, 0] = background[0, :, :]
##    img[:, :, 1] = photoswitching[0, :, :]
##    img[:, :, 2] = background[0, :, :]
##    just_magenta, just_green = np.array([1, 0, 1]), np.array([0, 1, 0])
##    imwrite(temp_dir / '5_background_c.tif',
##            (img * just_magenta * 255).astype(np.uint8))
##    imwrite(temp_dir / '6_fluorophores_c.tif',
##            (img * just_green * 255).astype(np.uint8))
##    imwrite(temp_dir / '7_overlay.tif',
##            (img * 255).astype(np.uint8))
##
##    # Output annotated png for animation.
##    for i in range(4):
##        fig = plt.figure(figsize=(1, 1*(img.shape[0]/img.shape[1])))
##        ax = plt.axes([0, 0, 1, 1])
##        if i == 0:
##            x = img # Overlay
##        elif i == 1:
##            x = img * just_green
##        elif i == 2:
##            x = img # Overlay
##        elif i == 3:
##            x = img * just_magenta
##        ax.imshow(x, interpolation='nearest')
##        ax.set_xticks([])
##        ax.set_yticks([])
##        if i != 3:
##            ax.text(900, 1000, "Sensor",
##                fontdict={'color': (0, 1, 0),
##                          'weight': 'bold',
##                          'size': 4})
##        if i != 1:
##            ax.text(900, 1100, "Background",
##                fontdict={'color': (1, 0, 1),
##                          'weight': 'bold',
##                          'size': 4})
##        plt.savefig(temp_dir / ("overlay_frame_%03i.png"%i), dpi=800)
##
##    # Animate the frames into a gif:
##    palette = temp_dir / "palette.png"
##    filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
##    print("Converting pngs to gif...", end=' ')
##    convert_command_1 = [
##        'ffmpeg',
##        '-f', 'image2',
##        '-i', str(temp_dir / 'overlay_frame_%3d.png'),
##        '-vf', filters + ",palettegen",
##        '-y', palette]
##    convert_command_2 = [
##        'ffmpeg',
##        '-framerate', '0.7',
##        '-f', 'image2',
##        '-i', str(temp_dir / 'overlay_frame_%3d.png'),
##        '-i', palette,
##        '-lavfi', filters + " [x]; [x][1:v] paletteuse",
##        '-y', str(output_dir / "1_overlay_animation.gif")]
##    for convert_command in convert_command_1, convert_command_2:
##        try:
##            with open(temp_dir / 'conversion_messages.txt', 'wt') as f:
##                f.write("So far, everthing's fine...\n")
##                f.flush()
##                subprocess.check_call(convert_command, stderr=f, stdout=f)
##                f.flush()
##            (temp_dir / 'conversion_messages.txt').unlink()
##        except: # This is unlikely to be platform independent :D
##            print("GIF conversion failed. Is ffmpeg installed?")
##            raise
##    print('done.')
##
##    # Extract images for one photoswitching cycle. Smooth them a little,
##    # since we'll be resampling the image anyway for the figure.
##    cycle = gaussian_filter(data[4:15, :, :], sigma=(0, 1, 1))
##    cycle_timestamps = (timestamps[4:15] - timestamps[5]) * 1e-6
##    imwrite(temp_dir / '8_switching_cycle.tif',
##            np.expand_dims(cycle, 1), imagej=True)
##
##    # Output annotated png for animation.
##    for i in range(cycle.shape[0]):
##        fig = plt.figure(figsize=(8, 8*(cycle.shape[1]/cycle.shape[2])))
##        ax1 = plt.axes([0, 0, 1, 1])
##        ax1.imshow(cycle[i], interpolation='nearest', cmap=plt.cm.gray,
##                  vmin=100, vmax=625)
##        ax1.set_xticks([])
##        ax1.set_yticks([])
##        ax1.text(100, 100, "Raw data",
##            fontdict={'color': (1, 1, 1),
##                      'weight': 'bold',
##                      'size': 32})
##        ax1.text(
##            1200, 650,
##            "t=%5ss"%('%0.2f'%(cycle_timestamps[i])),
##            fontdict={'color': (1, 1, 1),
##                      'weight': 'bold',
##                      'size': 20,
##                      'family': 'monospace',})
##        ax2 = plt.axes([0.5, 0.095, 0.47, 0.35])
##        ax1.add_patch(Rectangle(
##            (279, 789), 21, 21,
##            fill=False, linewidth=3, linestyle=(0, (0.5, 0.5)),
##            color=(1, 0, 1)))
##        ax1.add_patch(Rectangle(
##            (722, 376), 21, 21,
##            fill=False, linewidth=2, color=(0, 1, 0)))
##        box_1_photons = to_photoelectrons(
##            cycle[:, 789:789+21, 279:279+21].mean(axis=(1, 2)))
##        box_2_photons = to_photoelectrons(
##            cycle[:, 376:376+21, 722:722+21].mean(axis=(1, 2)))
##        ax2.plot(cycle_timestamps, box_1_photons,
##                 marker='.', markersize=7,
##                 linewidth=2.5, linestyle=(0, (1, 1)),
##                 color=(1, 0, 1))
##        ax2.plot(cycle_timestamps, box_2_photons,
##                 marker='.', markersize=7,
##                 linewidth=2.5,
##                 color=(0, 1, 0))
##        ax2.plot([cycle_timestamps[-1]*1.08]*2,
##                 [box_1_photons.min(), box_1_photons.max()],
##                 marker=0, markersize=10, linewidth=2.5,
##                 color=(1, 0, 1))
##        ax2.plot([cycle_timestamps[-1]*1.12]*2,
##                 [box_2_photons.min(), box_2_photons.max()],
##                 marker=0, markersize=10, linewidth=2.5,
##                 color=(0, 1, 0))
##        ax2.text(
##            3.79, 93,
##            "%0.0f%%"%(100 * (box_1_photons.max() - box_1_photons.min()) /
##                       box_1_photons.min()),
##            fontdict={'color': (1, 0, 1),
##                      'weight': 'bold',
##                      'size': 10,})
##        ax2.text(
##            3.98, 65,
##            "%0.0f%%"%(100 * (box_2_photons.max() - box_2_photons.min()) /
##                       box_2_photons.min()),
##            fontdict={'color': (0, 0.5, 0),
##                      'weight': 'bold',
##                      'size': 10,})
##        ax2.set_xlim(cycle_timestamps.min() - 0.1, 1.35*cycle_timestamps.max())
##        ax2.set_ylim(0, 1.2*max(box_1_photons.max(), box_2_photons.max()))
##        ax2.grid('on')
##        ax2.tick_params(labelcolor='white')
##        ax2.set_xlabel("Time (s)", color='white', weight='bold')
##        ax2.set_ylabel("Photons per pixel", color='white', weight='bold')
##        ax2.axvline(cycle_timestamps[i])
##        ax2.add_patch(Rectangle(
##            (cycle_timestamps[0] - 0.1, 0),
##            (cycle_timestamps[1] - cycle_timestamps[0]) - 0.1, 130,
##            fill=True, linewidth=0, color=(0, 0, 1, 0.17)))
##        ax2.text(
##            0.7 + cycle_timestamps[0], 22,
##            "405 nm\nillumination",
##            fontdict={'color': (0, 0, 1),
##                      'weight': 'bold',
##                      'size': 8.5,
##                      'horizontalalignment': 'center',})
##        plt.savefig(temp_dir / ("data_frame_%03i.png"%i), dpi=100)
##
##    # Animate the frames into a gif:
##    palette = temp_dir / "palette2.png"
##    filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
##    print("Converting pngs to gif...", end=' ')
##    convert_command_1 = [
##        'ffmpeg',
##        '-f', 'image2',
##        '-i', str(temp_dir / 'data_frame_%3d.png'),
##        '-vf', filters + ",palettegen",
##        '-y', palette]
##    convert_command_2 = [
##        'ffmpeg',
##        '-framerate', '12',
##        '-f', 'image2',
##        '-i', str(temp_dir / 'data_frame_%3d.png'),
##        '-i', palette,
##        '-lavfi', filters + " [x]; [x][1:v] paletteuse",
##        '-y', str(output_dir / "2_data_animation.gif")]
##    for convert_command in convert_command_1, convert_command_2:
##        try:
##            with open(temp_dir / 'conversion_messages.txt', 'wt') as f:
##                f.write("So far, everthing's fine...\n")
##                f.flush()
##                subprocess.check_call(convert_command, stderr=f, stdout=f)
##                f.flush()
##            (temp_dir / 'conversion_messages.txt').unlink()
##        except: # This is unlikely to be platform independent :D
##            print("GIF conversion failed. Is ffmpeg installed?")
##            raise
##    print('done.')


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

##def adjust_contrast(img, min_, max_):
##    """Like setting "minimum" and "maximum" image contrast in ImageJ
##
##    Output image intensity will range from zero to one.
##    Non-quantitative, just useful for display.
##    """
##    img = np.clip(img, min_, max_)
##    img -= min_
##    img /= max_ - min_
##    return img
##
##def to_photoelectrons(camera_counts):
##    """A good approximation for the pco edge 4.2
##    """
##    return (camera_counts - 100) * 0.46

main()
