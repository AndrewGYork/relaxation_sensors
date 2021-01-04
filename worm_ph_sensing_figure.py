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
    
    timestamps = 1e-6*decode_timestamps(data)['microseconds'].astype('float64')
    
    # Crop, flip, convert to float:
    data = data[:, 1776:162:-1, 1329:48:-1
                ].transpose((0, 2, 1)).astype('float32')
    print(data.shape, data.dtype)

    for which_cycle in range(3):
        # Extract the relevant images
        activation = data[5 + which_cycle*16:16+which_cycle*16, :, :].copy()
        # Add artificial patches with carefully chosen dynamics to give an
        # effective "colorbar"
        # These numbers are calculated by 'worm_ph_calibration.py'
        relaxation_colorbar = ( # pH 8, 7.5, 7, 6.5, 6 with 2x interpolation
            (0.82, 0.605, 0.39, 0.275, 0.16, 0.099, 0.038, 0.019, 0),
            (1.08, 0.855, 0.63, 0.46, 0.29, 0.185, 0.08, 0.045, 0.01),
            (1.28, 1.135, 0.99, 0.78, 0.57, 0.38, 0.19, 0.1185, 0.047),
            )[which_cycle]
        nonlinearity_colorbar = ( # pH 8, 7.5, 7, 6.5, 6 with 2x interpolation
            (0.38, 0.355, 0.33, 0.315, 0.3, 0.27, 0.24, 0.225, 0.21),
            (0.38, 0.355, 0.33, 0.315, 0.3, 0.27, 0.24, 0.225, 0.21),
            (0.38, 0.355, 0.33, 0.315, 0.3, 0.27, 0.24, 0.225, 0.21),
            )[which_cycle]
        a_max = activation.max()
        for patch in range(9):
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
        norm = np.clip(gaussian_filter(photoswitching, sigma=2), 1, None)
        relaxation_ratio = gaussian_filter(relaxation, sigma=2) / norm
        nonlinearity_ratio = gaussian_filter(nonlinearity, sigma=2) / norm
        imwrite(temp_dir / ('5_relaxation_ratio_%i.tif'%which_cycle),
                relaxation_ratio, imagej=True)
        imwrite(temp_dir / ('6_nonlinearity_ratio_%i.tif'%which_cycle),
                nonlinearity_ratio, imagej=True)
        
        # Smooth and color-merge photoswitching/relaxation or
        # photoswitching/nonlinearity
        normed_photoswitching = adjust_contrast(
            gaussian_filter(photoswitching, sigma=0), 50, 1000, 0.3)
        scale = (0.37, 0.55, 0.85)[which_cycle] # Amount of relaxation varies
        normed_relaxation_ratio = adjust_contrast(
            relaxation_ratio, 0, scale, 3)
        im1 = np.zeros((data.shape[1], data.shape[2], 3)) # RGB image, 0<=im<=1
        im1[:, :, 0] = normed_photoswitching * normed_relaxation_ratio
        im1[:, :, 1] = normed_photoswitching * normed_relaxation_ratio
        im1[:, :, 2] = normed_photoswitching * (1 - normed_relaxation_ratio)
        im1 /= im1.max()
        imwrite(temp_dir / ('7_relaxation_ratio_overlay_%i.tif'%which_cycle),
                (im1 * 255).astype(np.uint8))
        normed_nonlinearity_ratio = adjust_contrast(
            nonlinearity_ratio, 0.04, 0.36, 3)
        im2 = np.zeros((data.shape[1], data.shape[2], 3)) # RGB image, 0<=im<=1
        im2[:, :, 0] = normed_photoswitching * normed_nonlinearity_ratio
        im2[:, :, 1] = normed_photoswitching * normed_nonlinearity_ratio
        im2[:, :, 2] = normed_photoswitching * (1 - normed_nonlinearity_ratio)
        im2 /= im2.max()
        imwrite(temp_dir / ('8_nonlinearity_ratio_overlay_%i.tif'%which_cycle),
                (im2 * 255).astype(np.uint8))
    
        # Output annotated png
        fig = plt.figure(figsize=(1, 1*(im1.shape[0]/im1.shape[1])))
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(im1, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
##        ax.text(900, 1000, "Sensor",
##            fontdict={'color': (0, 1, 0),
##                      'weight': 'bold',
##                      'size': 4})
        plt.savefig(output_dir / ("relaxation_overlay_%i.png"%which_cycle),
                    dpi=800)
        plt.close(fig)
        fig = plt.figure(figsize=(1, 1*(im2.shape[0]/im2.shape[1])))
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(im2, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
##        ax.text(900, 1000, "Sensor",
##            fontdict={'color': (0, 1, 0),
##                      'weight': 'bold',
##                      'size': 4})
        plt.savefig(output_dir / ("nonlinearity_overlay_%i.png"%which_cycle),
                    dpi=800)



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
                     color=(0, 0, 1))
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
                     linewidth=1, linestyle=(0, (1, 1)),
                     color=(1, 1, 0))
            ax3.plot(cycle_timestamps[-1], box_2_photons[-1],
                     marker='.', markersize=7,
                     linewidth=1,
                     color=(1, 1, 0))
            # Misc plot tweaks:
            ax2.set_xlim(cycle_timestamps.min() - 0.1, 18)
            ax2.set_ylim(-box_1_photons.max()*0.02, box_1_photons.max() * 1.2)
            ax3.set_ylim(-box_2_photons.max()*0.02, box_2_photons.max() * 1.2)
            ax2.grid('on', alpha=0.2)
            ax2.tick_params(labelcolor='white')
            ax2.tick_params('y', labelcolor='blue')
            ax3.tick_params('y', labelcolor='yellow')
            ax2.set_xlabel("Time (s)", color='white', weight='bold')
            ax2.set_ylabel("Photons per pixel", color='blue', weight='bold')
            ax3.set_ylabel("Photons per pixel", color='yellow', weight='bold')
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
            '-y', str(output_dir / ("data_animation_%i.gif"%(which_cycle)))]
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
