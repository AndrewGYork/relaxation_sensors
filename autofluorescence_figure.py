#!/usr/bin/python
# Dependencies from the python standard library:
import subprocess
from pathlib import Path
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
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
    
    timestamps = decode_timestamps(data)['microseconds']
    
    # Crop, flip, convert to float:
    data = data[:, 1776:162:-1, 1329:48:-1].transpose((0, 2, 1)).astype('float32')
    print(data.shape, data.dtype)

    # Extract the relevant images: minimum and maximum photoactivation
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
    print("Time intervals:", intervals / 1e6)

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
        plt.savefig(temp_dir / ("animation_frame_%03i.png"%i), dpi=800)

    # Animate the frames into a gif:
    palette = temp_dir / "palette.png"
    filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
    print("Converting pngs to gif...", end=' ')
    convert_command_1 = [
        'ffmpeg',
        '-f', 'image2',
        '-i', str(temp_dir / 'animation_frame_%3d.png'),
        '-vf', filters + ",palettegen",
        '-y', palette]
    convert_command_2 = [
        'ffmpeg',
        '-framerate', '0.7',
        '-f', 'image2',
        '-i', str(temp_dir / 'animation_frame_%3d.png'),
        '-i', palette,
        '-lavfi', filters + " [x]; [x][1:v] paletteuse",
        '-y', str(output_dir / "animation.gif")]
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
