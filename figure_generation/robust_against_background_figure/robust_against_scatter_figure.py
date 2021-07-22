#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
import shutil     # For copying files
import subprocess # For calling ffmpeg to make animations
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tifffile import imread # v2020.6.3 or newer

input_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'chicken_ph_data')
temp_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'robust_against_background_figure')
output_dir = ( # The 'images' directory, two folders up:
    Path(__file__).parents[2] / 'images' / 'robust_against_background_figure')
# Sanity checks:
temp_dir.mkdir(exist_ok=True, parents=True)
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading...", end='')
fluorescence = imread(
    input_dir / '2021_13_10c_capillaries_under_chicken_skin.tif')
reflected = imread(
    input_dir / '2021_13_10c_capillaries_under_chicken_skin_reflected.tif')
print(" done.")

def main():
    print("Saving frames...", sep='', end='')
##    for i in range(51, 52):
##    for i in (28, 35, 38, 41, 43, 56, 58, 61, 79):
    out_frame = 0
    for i in range(15):
        save_animation_frame(in_frame=28, out_frame=out_frame, bb=0.1)
        out_frame += 1
    for i in (28, 35, 38, 41, 43) + tuple(range(51, 80)):
        save_animation_frame(in_frame=i, out_frame=out_frame, bb=0.1)
        out_frame += 1
    for i in range(5):
        save_animation_frame(in_frame=79, out_frame=out_frame,
                             bb=0.1 + 0.9*(i+1)/5)
        out_frame += 1
    for i in range(36):
        save_animation_frame(in_frame=79, out_frame=out_frame, bb=1)
        out_frame += 1
    for i in range(5):
        save_animation_frame(in_frame=79, out_frame=out_frame,
                             bb=0.1 + 0.9*(5-i)/5)
        out_frame += 1
    for i in range(5):
        save_animation_frame(in_frame=79, out_frame=out_frame, bb=0.1)
        out_frame += 1
    print("done.")
    make_gif()

def save_animation_frame(in_frame, out_frame, bb=1):
    # Decode timestamps
    t = decode_timestamps(fluorescence)['microseconds'] * 1e-6
    t -= t[51]
    # Crop and convert to float
    slice_a = (..., slice(   8,  722), slice( None, 1900))
    slice_b = (..., slice(1244, 1958), slice( None, 1900))
    fluorescence_a = fluorescence[slice_a].astype('float32')
    fluorescence_b = fluorescence[slice_b].astype('float32')
    reflected_a = reflected[slice_a].astype('float32')
    reflected_b = reflected[slice_b].astype('float32')
    # Construct RBG images overlaying reflected and fluorescence images
    img_a = np.zeros((fluorescence_a.shape[-2], fluorescence_a.shape[-1], 3))
    img_b = np.zeros((fluorescence_b.shape[-2], fluorescence_b.shape[-1], 3))
    img_a[:, :, 1] = adjust_contrast(fluorescence_a[in_frame, :, :],
                                      120, 1000, gamma=0.5)
    img_b[:, :, 1] = adjust_contrast(fluorescence_b[in_frame, :, :],
                                      100, 500, gamma=0.5)
    for i in range(3):
        img_a[:, :, i] += adjust_contrast(reflected_a, 100, 60000)
        img_b[:, :, i] += adjust_contrast(reflected_b, 100, 60000)
    img_a = np.clip(img_a, 0, 1)
    img_b = np.clip(img_b, 0, 1)
    # Extract time-traces for each pH:
    # pH 8
    x0, y0, x1, y1 = (104, 100, 141, 581)
    ph_8p0_a = fluorescence_a[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_8p0_a_box = ((x0, y0), x1-x0, y1-y0)
    x0, y0, x1, y1 = (140, 82, 192, 664)
    ph_8p0_b = fluorescence_b[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_8p0_b_box = ((x0, y0), x1-x0, y1-y0)
    # pH 7.5
    x0, y0, x1, y1 = (499, 45, 542, 690)
    ph_7p5_a = fluorescence_a[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_7p5_a_box = ((x0, y0), x1-x0, y1-y0)
    x0, y0, x1, y1 = (531, 155, 604, 605)
    ph_7p5_b = fluorescence_b[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_7p5_b_box = ((x0, y0), x1-x0, y1-y0)
    # pH 7
    x0, y0, x1, y1 = (903, 32, 946, 580)
    ph_7p0_a = fluorescence_a[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_7p0_a_box = ((x0, y0), x1-x0, y1-y0)
    x0, y0, x1, y1 = (950, 71, 1032, 615)
    ph_7p0_b = fluorescence_b[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_7p0_b_box = ((x0, y0), x1-x0, y1-y0)
    # pH 6.5
    x0, y0, x1, y1 = (1309, 22, 1355, 591)
    ph_6p5_a = fluorescence_a[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_6p5_a_box = ((x0, y0), x1-x0, y1-y0)
    x0, y0, x1, y1 = (1371, 71, 1448, 603)
    ph_6p5_b = fluorescence_b[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_6p5_b_box = ((x0, y0), x1-x0, y1-y0)
    # pH 6
    x0, y0, x1, y1 = (1688, 43, 1738, 463)
    ph_6p0_a = fluorescence_a[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_6p0_a_box = ((x0, y0), x1-x0, y1-y0)
    x0, y0, x1, y1 = (1749, 22, 1837, 627)
    ph_6p0_b = fluorescence_b[:, y0:y1, x0:x1].mean(axis=(-1, -2))
    ph_6p0_b_box = ((x0, y0), x1-x0, y1-y0)


    fig = plt.figure(figsize=(6.4, 4.8))
    # Images
    ax0 = plt.axes([0.0, 0.61, 0.5, 0.5])
    ax1 = plt.axes([0.5, 0.61, 0.5, 0.5])
    ax0.imshow(img_a)
    ax1.imshow(img_b)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    # pH 8.0
    ax0.text(ph_8p0_a_box[0][0]-40, 890, 'pH\n8.0',
             weight='bold', color=(0.8, 0.8, 0, 1))
    ax1.text(ph_8p0_b_box[0][0]-40, 890, 'pH\n8.0',
             weight='bold',color=(0.8, 0.8, 0, 1*bb))
    ax0.add_patch(Rectangle(*ph_8p0_a_box, fill=False,
                            linewidth=1, color=(0.8, 0.8, 0, 0.5)))
    ax1.add_patch(Rectangle(*ph_8p0_b_box, fill=False,
                            linewidth=1, color=(0.7, 0.7, 0, 1*bb)))
    # pH 7.5
    ax0.text(ph_7p5_a_box[0][0]-40, 890, 'pH\n7.5',
             weight='bold', color=(1, 0.5, 0.05, 0.8))
    ax1.text(ph_7p5_b_box[0][0]-40, 890, 'pH\n7.5',
             weight='bold', color=(1, 0.5, 0.05, 0.8*bb))
    ax0.add_patch(Rectangle(*ph_7p5_a_box, fill=False,
                            linewidth=1, color=(1, 0.5, 0.05, 0.4)))
    ax1.add_patch(Rectangle(*ph_7p5_b_box, fill=False,
                            linewidth=1, color=(1, 0.5, 0.05, 1*bb)))
    # pH 7.0
    ax0.text(ph_7p0_a_box[0][0]-40, 890, 'pH\n7.0',
             weight='bold', color=(0.17, 0.63, 0.17, 1))
    ax1.text(ph_7p0_b_box[0][0]-40, 890, 'pH\n7.0',
             weight='bold', color=(0.17, 0.63, 0.17, 1*bb))
    ax0.add_patch(Rectangle(*ph_7p0_a_box, fill=False,
                            linewidth=1, color=(0.17, 0.63, 0.17, 0.5)))
    ax1.add_patch(Rectangle(*ph_7p0_b_box, fill=False,
                            linewidth=1, color=(0.17, 0.63, 0.17, 1*bb)))
    # pH 6.5
    ax0.text(ph_6p5_a_box[0][0]-40, 890, 'pH\n6.5',
             weight='bold', color=(0.84, 0.15, 0.16, 0.7))
    ax1.text(ph_6p5_b_box[0][0]-40, 890, 'pH\n6.5',
             weight='bold', color=(0.84, 0.15, 0.16, 0.7*bb))
    ax0.add_patch(Rectangle(*ph_6p5_a_box, fill=False,
                            linewidth=1, color=(0.84, 0.15, 0.16, 0.5)))
    ax1.add_patch(Rectangle(*ph_6p5_b_box, fill=False,
                            linewidth=1, color=(0.84, 0.15, 0.16, 0.7*bb)))
    # pH 6.0
    ax0.text(ph_6p0_a_box[0][0]-40, 890, 'pH\n6.0',
             weight='bold', color=(0.12, 0.47, 0.7, 1))
    ax1.text(ph_6p0_b_box[0][0]-40, 890, 'pH\n6.0',
             weight='bold', color=(0.12, 0.47, 0.7, 1*bb))
    ax0.add_patch(Rectangle(*ph_6p0_a_box, fill=False,
                            linewidth=1, color=(0.12, 0.47, 0.7, 0.5)))
    ax1.add_patch(Rectangle(*ph_6p0_b_box, fill=False,
                            linewidth=1, color=(0.12, 0.47, 0.7, 1*bb)))
    # Line plots
    ax2 = plt.axes([0.1, 0.1, 0.85, 0.55], facecolor=(0.92, 0.92, 0.92))
    ax2.axvline(t[in_frame], alpha=0.2)
    ax2.add_patch(Rectangle( # Show when the illumination is on:
        (t[29], 0), t[51]-t[29], 1,
        fill=True, linewidth=0, color=(0.05, 0.95, 0.95, 0.3)))

    # pH 6.0
    ax2.plot(t, normalize_relaxation(t, ph_6p0_a), '.--',
             c=(0.12, 0.47, 0.7, 0.5),
             label='pH 6.0')
    ax2.plot(t, normalize_relaxation(t, ph_6p0_b), 'o',
             mfc=(0, 0, 0, 0), mec=(0.12, 0.47, 0.7, 1*bb),
             label='  under skin')
    ax2.plot(0, 0, c=(0, 0, 0, 0), label=' ') # Dummy gap for legend
    # pH 6.5
    ax2.plot(t, normalize_relaxation(t, ph_6p5_a), '.-',
             c=(0.84, 0.15, 0.16, 0.4),
             label='pH 6.5')
    ax2.plot(t, normalize_relaxation(t, ph_6p5_b), 'o',
             mfc=(0, 0, 0, 0), mec=(0.84, 0.15, 0.16, 1*bb),
             label='  under skin')
    ax2.plot(0, 0, c=(0, 0, 0, 0), label=' ') # Dummy gap for legend
    # pH 7.0
    ax2.plot(t, normalize_relaxation(t, ph_7p0_a), '.--',
             c=(0.17, 0.63, 0.17, 0.4),
             label='pH 7.0')
    ax2.plot(t, normalize_relaxation(t, ph_7p0_b), 'o',
             mfc=(0, 0, 0, 0), mec=(0.17, 0.63, 0.17, 1*bb),
             label='  under skin')
    ax2.plot(0, 0, c=(0, 0, 0, 0), label=' ') # Dummy gap for legend
    # pH 7.5
    ax2.plot(t, normalize_relaxation(t, ph_7p5_a), '.-',
             c=(1, 0.5, 0.05, 0.4),
             label='pH 7.5')
    ax2.plot(t, normalize_relaxation(t, ph_7p5_b), 'o',
             mfc=(0, 0, 0, 0), mec=(1, 0.5, 0.05, 1*bb),
             label='  under skin')
    ax2.plot(0, 0, c=(0, 0, 0, 0), label=' ') # Dummy gap for legend
    # pH 8.0
    ax2.plot(t, normalize_relaxation(t, ph_8p0_a), '.--',
             c=(1, 1, 0), mec=(0.8, 0.8, 0),
             label='pH 8.0')
    ax2.plot(t, normalize_relaxation(t, ph_8p0_b), 'o',
             mfc=(0, 0, 0, 0), mec=(0.7, 0.7, 0, 1*bb),
             label='  under skin')

    ax2.set_xlim(-85, 400)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Time (s)", weight='bold')
    ax2.set_ylabel("Normalized brightness", weight='bold')
    ax2.grid('on')
    ax2.legend(loc=(0.86, 0.05), fontsize=8,
               facecolor=(0.8, 0.8, 0.8), framealpha=0.95)
    plt.savefig(temp_dir / ('animation_frame_%06i.png'%out_frame),
                dpi=300)
    print('.', sep='', end='')
##    plt.show()
    plt.close(fig)

def make_gif():
    # Animate the frames into a gif:
    palette = temp_dir / "palette.png"
    filters = "scale=trunc(iw/5)*2:trunc(ih/5)*2:flags=lanczos"
    print("Converting pngs to gif...", end=' ')
    convert_command_1 = [
        'ffmpeg',
        '-f', 'image2',
        '-i', temp_dir / 'animation_frame_%06d.png',
        '-vf', filters + ",palettegen",
        '-y', palette]
    convert_command_2 = [
        'ffmpeg',
        '-framerate', '15',
        '-f', 'image2',
        '-i', temp_dir / 'animation_frame_%06d.png',
        '-i', palette,
        '-lavfi', filters + " [x]; [x][1:v] paletteuse",
        '-y', output_dir / 'figure.gif']
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

def normalize_relaxation(t, x):
    s = slice(29, None)
    return (x - x[s].min()) / (x[s].max() - x[s].min())

if __name__ == '__main__':
    main()
