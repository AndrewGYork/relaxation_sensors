#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
import subprocess # For calling ffmpeg to make animations
import urllib.request  # For downloading raw data
import zipfile         # For unzipping downloads
# You can use 'pip' to install these dependencies:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tifffile as tf
# Our stuff from https://github.com/AndrewGYork/tools
import stack_registration as sr

input_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'mouse_mito_bam_data')
temp_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'mouse_mito_bam_figure')
output_dir = ( # The 'images' directory, two folders up:
    Path(__file__).parents[2] / 'images' / 'mouse_mito_bam_figure')

# Sanity checks:
input_dir.mkdir(exist_ok=True, parents=True)
temp_dir.mkdir(exist_ok=True, parents=True)
output_dir.mkdir(exist_ok=True)

def main():
    semiorganized_data = load_and_organize_data()
    for which_comparison in ('10_minutes', '40_minutes'):
        print("Saving animation for: ", which_comparison, " after BAM...",
              end='', sep='')
        which_output_frame = -1
        for before_or_after in ('before', 'after'):
            for which_input_frame in range(41):
                which_output_frame += 1
                make_figure(semiorganized_data,
                            which_input_frame,
                            which_output_frame,
                            before_or_after,
                            which_comparison)
                print('.', sep='', end='')
        print(" done.")
        make_gif(which_comparison)

def load_and_organize_data():
    # Load
    data_1 = load_tif(input_dir / '2021_10_21_1_mito_mouse_before_bam.tif')
    data_2 = load_tif(input_dir / '2021_10_21_2_mito_mouse_10m_after_bam.tif')
    data_3 = load_tif(input_dir / '2021_10_21_3_mito_mouse_40m_after_bam.tif')
    calibr = load_tif(input_dir / '2021_10_22_calibration.tif')
    calibr_ts = load_tif(input_dir / '2021_10_22_calibration_timestamps.tif')
    # Extract timestamps
    time_1_s = 1e-6*decode_timestamps(data_1)['microseconds']
    time_2_s = 1e-6*decode_timestamps(data_2)['microseconds']
    time_3_s = 1e-6*decode_timestamps(data_3)['microseconds']
    time_c_s = 1e-6*decode_timestamps(calibr_ts)['microseconds']
    # Register based on a sub-landmark
    landmark_1 = data_1[:, 1511:1552, 988:1030].astype('float64')
    shifts_1 = sr.stack_registration(
        landmark_1, refinement='integer', fourier_cutoff_radius=0.15)
    sr.apply_registration_shifts(data_1, shifts_1, 'nearest_integer')

    landmark_2 = data_2[:, 1491:1532, 983:1025].astype('float64')
    shifts_2 = sr.stack_registration(
        landmark_2, refinement='integer', fourier_cutoff_radius=0.15)
    sr.apply_registration_shifts(data_2, shifts_2, 'nearest_integer')

    landmark_3 = data_3[:, 1486:1527, 983:1025].astype('float64')
    shifts_3 = sr.stack_registration(
        landmark_3, refinement='integer', fourier_cutoff_radius=0.15)
    sr.apply_registration_shifts(data_3, shifts_3, 'nearest_integer')

    # Crop to a common window; landmark should be at 966, 756
    data_1 = data_1[:, 770:2009, 45:1971]
    data_2 = data_2[:, 753:1992, 40:1966]
    data_3 = data_3[:, 745:1984, 34:1960]

    # Extract signals for plotting
    box_xi, box_xf, box_yi, box_yf = (955, 999, 683, 727)
    signal_1 = data_1[:, box_yi:box_yf, box_xi:box_xf]
    signal_2 = data_2[:, box_yi:box_yf, box_xi:box_xf]
    signal_3 = data_3[:, box_yi:box_yf, box_xi:box_xf]
    signal_ph_7p5 = calibr[:,  906:958,  114:161]
    signal_ph_8p0 = calibr[:, 1108:1160, 118:165]
    # Reorganize
    mean_1a = signal_1.mean(axis=(-1, -2))[:41]
    mean_1b = signal_1.mean(axis=(-1, -2))[69:]
    time_1a_s = time_1_s[:41] - time_1_s[26]
    time_1b_s = time_1_s[69:] - time_1_s[95]
    
    mean_2a = signal_2.mean(axis=(-1, -2))[:41]
    mean_2b = signal_2.mean(axis=(-1, -2))[69:]
    time_2a_s = time_2_s[:41] - time_2_s[26]
    time_2b_s = time_2_s[69:] - time_2_s[95]
    
    mean_3a = signal_3.mean(axis=(-1, -2))[:41]
    mean_3b = signal_3.mean(axis=(-1, -2))[69:]
    time_3a_s = time_3_s[:41] - time_3_s[26]
    time_3b_s = time_3_s[69:] - time_3_s[95]

    mean_ph_7p5 = (signal_ph_7p5.mean(axis=(-1, -2))[:41] +
                   signal_ph_7p5.mean(axis=(-1, -2))[69:110]) / 2
    mean_ph_8p0 = (signal_ph_8p0.mean(axis=(-1, -2))[:41] +
                   signal_ph_8p0.mean(axis=(-1, -2))[69:110]) / 2
    time_c_s = time_c_s[:41] - time_c_s[26]

    return (data_1, data_2, data_3,
            box_xi, box_xf, box_yi, box_yf,
            mean_1a, mean_1b, time_1a_s, time_1b_s,
            mean_2a, mean_2b, time_2a_s, time_2b_s,
            mean_3a, mean_3b, time_3a_s, time_3b_s,
            mean_ph_7p5, mean_ph_8p0, time_c_s)

def load_tif(image_data_filename):
    if not image_data_filename.is_file():
        print("The expected data file:")
        print(image_data_filename)
        print("...isn't where we expect it.\n")
        print(" * Let's try to unzip it...")
        zipped_data_filename = input_dir / "mito_ph_mouse_with_bam_data.zip"
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
    print("Successfully loaded data.")
    print("Data shape:", data.shape)
    print("Data dtype:", data.dtype)
    print()
    return data

def download_data(filename):
    url = ("https://zenodo.org/record/5818986/files/" +
           "mito_ph_mouse_with_bam_data.zip")
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

def make_figure(semiorganized_data,
                which_input_frame,
                which_output_frame,
                before_or_after,
                which_comparison):
    (data_1, data_2, data_3,
     box_xi, box_xf, box_yi, box_yf,
     mean_1a, mean_1b, time_1a_s, time_1b_s,
     mean_2a, mean_2b, time_2a_s, time_2b_s,
     mean_3a, mean_3b, time_3a_s, time_3b_s,
     mean_ph_7p5, mean_ph_8p0, time_c_s
    ) = semiorganized_data

    fig = plt.figure(figsize=(8, 4))

    # Animation
    if before_or_after == 'before':
        before_alpha = 1
        after_alpha = 0.35
        data = data_1
        box_color, box_alpha = 'C1', 0.7
        plt.figtext(0.05, 0.85, "Before BAM15",
                    fontsize=14, weight='bold', c='C1')
    elif before_or_after == 'after':
        before_alpha = 0.4
        after_alpha = 1
        if which_comparison == '10_minutes':
            data = data_2
            box_color, box_alpha = 'C0', 0.4
        elif which_comparison == '40_minutes':
            data = data_3
            box_color, box_alpha = 'C2', 0.5
        plt.figtext(0.05, 0.85, "After BAM15",
                    fontsize=14, weight='bold', c=box_color)
    def norm(x):
        x = x.astype('float64')
        xmin=150
        xmax=1000
        return np.log(1e-3 + np.clip((x - xmin) / (xmax - xmin), 0, 1))
    ax1 = plt.axes((0.0, 0.05, 0.55, 1))    
    ax1.imshow(norm(data[which_input_frame, :, :]), cmap=plt.cm.gray)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.add_patch(Rectangle(
        (box_xi, box_yi), box_xf-box_xi, box_yf-box_yi,
        fill=False, linewidth=1.5, color=box_color, alpha=box_alpha))

    # Plot    
    ax2 = plt.axes((0.6, 0.135, 0.39, 0.83))
    def norm(x):
        xmin = x[40]
        xmax = x[26]
        return (x - xmin) / (xmax - xmin)
    # Before
    ax2.plot(time_1a_s, norm(mean_1a), marker='.', c='C1',
             linestyle='dotted', linewidth=2, alpha=before_alpha)
    ax2.plot(time_1b_s, norm(mean_1b), marker='.', c='C1',
             linestyle='solid', linewidth=2, alpha=before_alpha,
             label='Before BAM15')
    if which_comparison == '10_minutes':
        # 10 min after
        ax2.plot(time_2a_s, norm(mean_2a), c='C0',
                 marker='^', alpha=after_alpha,
                 linestyle='dotted', linewidth=2)
        ax2.plot(time_2b_s, norm(mean_2b), c='C0',
                 marker='^', alpha=after_alpha,
                 linestyle='solid', linewidth=2,
                 label='10m after BAM15')
    if which_comparison == '40_minutes':
        # 40 min after
        ax2.plot(time_3a_s, norm(mean_3a), c='C2',
                 marker='v', alpha=after_alpha,
                 linestyle='dotted', linewidth=2)
        ax2.plot(time_3b_s, norm(mean_3b), c='C2',
                 marker='v', alpha=after_alpha,
                 linestyle='solid', linewidth=2,
                 label='40m after BAM15')
    ax2.axvline(time_1a_s[which_input_frame], c=box_color, alpha=0.2)
    # Calibration
    ax2.plot(time_c_s[26:],  norm(mean_ph_7p5)[26:], linestyle='solid',
             c='gray', alpha=0.2, label='pH 7.5 calibration')
    ax2.plot(time_c_s[26:],  norm(mean_ph_8p0)[26:], linestyle='dashdot',
             c='gray', alpha=0.2, label='pH 8.0 calibration')
    ax2.set_xlim(-22, 35)
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Fluorescent signal (normalized)", fontsize=8)
    ax2.set_yticks(np.linspace(0, 1, 5))
    ax2.set_yticklabels(['' for x in ax2.get_yticklabels()])
    ax2.grid('on')
    ax2.legend(loc=(0.41, 0.76), fontsize=6, framealpha=1,
               prop={'weight': 'bold'})
    ax2.add_patch(Rectangle( # Show when illumination is on:
        (time_1a_s[26], -0.08), time_1a_s[6], 1.23,
        fill=True, linewidth=0, color=(0.05, 0.95, 0.95, 0.3)
        ))
##    plt.show()
    plt.savefig(temp_dir / ("frame_%06i.png"%(which_output_frame)),
                dpi=85)
    plt.close(fig)

def make_gif(which_comparison):
    # Animate the frames into a gif:
    palette = temp_dir / "palette.png"
    filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
    print("Converting pngs to gif...", end=' ')
    convert_command_1 = [
        'ffmpeg',
        '-f', 'image2',
        '-i', temp_dir / 'frame_%06d.png',
        '-vf', filters + ",palettegen",
        '-y', palette]
    convert_command_2 = [
        'ffmpeg',
        '-framerate', '15',
        '-f', 'image2',
        '-i', temp_dir / 'frame_%06d.png',
        '-i', palette,
        '-lavfi', filters + " [x]; [x][1:v] paletteuse",
        '-y', output_dir / ('%s.gif'%(which_comparison))]
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

if __name__ == '__main__':
    main()
