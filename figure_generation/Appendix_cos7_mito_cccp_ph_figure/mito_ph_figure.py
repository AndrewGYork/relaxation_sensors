#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
import shutil          # For copying files
import subprocess      # For calling ffmpeg to make animations
import urllib.request  # For downloading raw data
import zipfile         # For unzipping downloads
# You can use 'pip' to install these dependencies:
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite # v2020.6.3 or newer

# You'll need ffmpeg installed to make animations. If you can't type
# "ffmpeg --version" at the command prompt without an error, my code
# will probably fail too.

input_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'mito_ph_data')
temp_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'mito_figure')
output_dir = ( # The 'images' directory, two folders up:
    Path(__file__).parents[2] / 'images' /
    'Appendix_cos7_mitochondria_ph_figure')
# Sanity checks:
input_dir.mkdir(exist_ok=True, parents=True)
temp_dir.mkdir(exist_ok=True, parents=True)
output_dir.mkdir(exist_ok=True)

# Estimated by running 'calibration.py':
relaxation_ratios_to_ph = np.array((
    (0.09, 6.5),
    (0.12, 7.0),
    (0.21, 7.5),
    (0.49, 8.0)))

data = imread(input_dir / 'data.tif').astype('float64')
# 30 measurements, with 14 frames per measurement:
data = data.reshape(30, 14, 602, 671)
# First measurement didn't prepare with 405 nm light, so skip it:
data = data[1:, :, :, :]

# We're only calculating average pH for the whole frame, so sum up all
# the spatial pixels to estimate pH:
brightness = data.sum(axis=(-1, -2))
normalized_brightness = brightness - brightness[:, 0:1]
normalized_brightness = normalized_brightness / normalized_brightness[:, 9:10]
relaxation_ratios = normalized_brightness[:, 9] - normalized_brightness[:, 10]
time = 7 * np.arange(len(relaxation_ratios))

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

print("Generating animation frames...", end='')
for which_timepoint in range(data.shape[0]):
    print('.', end='', sep='')
    luminance = adjust_contrast(data[which_timepoint, 9],
                                data[which_timepoint, 9].min(),
                                data[which_timepoint, 9].max()
                                )[..., np.newaxis]
    hue = seminormalized_turbo(
        adjust_contrast(np.atleast_2d(relaxation_ratios[which_timepoint]),
                        0.09, 0.49))
    show_me = hue * luminance

    for patch, ratio in enumerate((0.09, 0.12, 0.21, 0.49)):
        ph_calibration_hue = seminormalized_turbo(
            adjust_contrast(np.atleast_2d(ratio), .09, .49))
        sl_y, sl_x = slice(-100, -20), slice(-60*(patch+2), -60*(patch+1))
        show_me[sl_y, sl_x, :] = 1
        show_me[sl_y, sl_x, :] *= ph_calibration_hue

    fig = plt.figure(figsize=(10, 4))
    ax1 = plt.axes([0.0, 0.15, 0.38, 0.8])
    ax1.imshow(show_me, interpolation='nearest')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(280, 630, "pH measured via relaxation rate",
        fontdict={'color': (0, 0, 0),
                  'weight': 'bold',
                  'size': 9})
    for xp, txt in ((380, "8.0"),
                    (440, "7.5"),
                    (500, "7.0"),
                    (560, "6.5"),):
        ax1.text(xp, 550, txt,
            fontdict={'color': (0, 0, 0),
                      'weight': 'bold',
                      'size': 9})

    ax2 = plt.axes([0.45, 0.15, 0.5, 0.8])
    ax2.plot(time, relaxation_ratios, '.-', c='black')
    ax2.grid('on')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Relaxation ratio")
    ax2.axvline(time[which_timepoint], alpha=0.3, c='black')
    ax2.annotate("CCCP", xy=(time[5], 0.4), xytext=(time[5]-25, 0.35),
                 arrowprops=dict(facecolor='black'))
    for ratio in (0.09, 0.12, 0.21, 0.49):
        line_color = seminormalized_turbo(
            adjust_contrast(np.atleast_2d(ratio), 0.09, 0.49))[0, 0, :]
        ax2.axhline(ratio, alpha=0.3, c=line_color, linewidth=5)
    plt.savefig(temp_dir / ("animation_frame_%06i.png"%which_timepoint))
    plt.close(fig)
print("done.")

# Animate the frames into a gif:
palette = temp_dir / "palette.png"
filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
print("Converting pngs to gif...", end=' ')
convert_command_1 = [
    'ffmpeg',
    '-f', 'image2',
    '-i', str(temp_dir / ("animation_frame_%06d.png")),
    '-vf', filters + ",palettegen",
    '-y', palette]
convert_command_2 = [
    'ffmpeg',
    '-framerate', '8',
    '-f', 'image2',
    '-i', str(temp_dir / ('animation_frame_%6d.png')),
    '-i', palette,
    '-lavfi', filters + " [x]; [x][1:v] paletteuse",
    '-y', str(output_dir / ("1_data_animation.gif"))]
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

