#!/usr/bin/python
# Dependencies from the python standard library:
from pathlib import Path
import subprocess # For calling ffmpeg to make animations
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

temp_dir = ( # A temp directory, three folders up:
    Path(__file__).parents[3] /
    'relaxation_sensors_temp_files' / 'mechanism_comparison_figure')
output_dir = ( # The 'images' directory, two folders up:
    Path(__file__).parents[2] / 'images' / 'sensor_mechanism_comparison_figure')
# Sanity checks:
temp_dir.mkdir(exist_ok=True, parents=True)
output_dir.mkdir(exist_ok=True)


def main():
    print("This might take a while. Be patient.")
    cycle_1 = np.concatenate((np.ones(15),
                              2**(-np.sin(np.linspace(0, 5*np.pi, 50))),
                              np.ones(10)))
    cycle_2 = np.concatenate((2**(-np.sin(np.linspace(0, 3*np.pi, 30))),
                              np.ones(5)))
    which_frame = -1
    for norm in (True, ):
        for key, cycle in (
            ('analyte_amount',      cycle_1),
            ('background_amount',   cycle_2),
            ('sensor_amount',       cycle_2),
            ('illumination_amount', cycle_2),
            ):
            for amount in cycle:
                which_frame += 1
                kwargs = {'which_frame': which_frame,
                          key: amount,
                          'normalize': norm}
                make_frame(**kwargs)
    print()
    for start_frame, end_frame, filename in (
        (  0, 180,         "full_animation.gif"),
        ( 35,  70,      "analyte_animation.gif"),
        ( 75, 110,   "background_animation.gif"),
        (110, 145,       "sensor_animation.gif"),
        (145, 180, "illumination_animation.gif"),
        ):
        make_gif(start_frame, end_frame, filename)

def make_frame(
    which_frame,
    analyte_amount=1,
    background_amount=1,
    sensor_amount=1,
    illumination_amount=1,
    normalize=False,
    ):
    print('.', sep='', end='')
    fig = plt.figure(figsize=(8, 4.8))
    ax0 = plt.axes((0, 0, 1, 1))

    ax0.text(0.65, 0.9, "Quantity of interest:",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 12})
    # Amount of analyte
    ax0.add_patch(Rectangle(
        (0.62, 0.8), 0.35*(0.5*analyte_amount), 0.07,
        fill=True, linewidth=0,
        color=(0.12, 0.47, 0.71, 0.8)))
    ax0.add_patch(Rectangle(
        (0.62, 0.8), 0.35, 0.07,
        fill=False, linewidth=1))
    ax0.text(0.68, 0.825, "Amount of analyte",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 11})

    ax0.text(0.65, 0.6, "Confounding factors:",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 12})
    # Amount of background
    ax0.add_patch(Rectangle(
        (0.62, 0.5), 0.35*(0.5*background_amount), 0.07,
        fill=True, linewidth=0,
        color=(1, 0, 1, 0.5)))
    ax0.add_patch(Rectangle(
        (0.62, 0.5), 0.35, 0.07,
        fill=False, linewidth=1))
    ax0.text(0.68, 0.525, "Amount of background",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 11})

    # Amount of sensor
    ax0.add_patch(Rectangle(
        (0.62, 0.4), 0.35*(0.5*sensor_amount), 0.07,
        fill=True, linewidth=0,
        color=(0, 1, 0, 0.5)))
    ax0.add_patch(Rectangle(
        (0.62, 0.4), 0.35, 0.07,
        fill=False, linewidth=1))
    ax0.text(0.68, 0.425, "Amount of sensor",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 11})

    # Amount of illumination
    ax0.add_patch(Rectangle(
        (0.62, 0.3), 0.35*(0.5*illumination_amount), 0.07,
        fill=True, linewidth=0,
        color=(0.12, 0.47, 0.71, 0.15)))
    ax0.add_patch(Rectangle(
        (0.62, 0.3), 0.35, 0.07,
        fill=False, linewidth=1))
    ax0.text(0.68, 0.325, "Amount of illumination",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 11})


    # Intensity sensors
    ax1 = plt.axes((0.07, 0.76, 0.5, 0.23))
    ax1.text(0.24, 0.8, "Fluorescence intensity sensors",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 12},
             transform=ax1.transAxes)
    t = np.linspace(-5, 100, 10000)
    illumination = illumination_amount * (t > 0) * (t < 10)
    background_signal = illumination * 0.2 * background_amount
    x = illumination * (
        background_signal +
        0.8 * sensor_amount * (0.5 + 0.5*analyte_amount))
    ax1.fill_between(t, 1000*illumination-300, -500,
                     alpha=0.1, linewidth=0, color=(0.12, 0.47, 0.71))
    ax1.fill_between(t, background_signal,
                     alpha=0.03, linewidth=0, color='magenta')
    ax1.plot(t, x, linewidth=2)
    ax1.plot(5, x.max(), marker='.', color='blue')

    ax1.set_xlabel("Time (milliseconds)", weight='bold', labelpad=-1)
    ax1.set_xlim(-10, 100)

    ax1.text(-0.1, 0.3, "Signal", rotation=90,
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 10},
             transform=ax1.transAxes)
    ax1.set_yticks(np.linspace(0, 1.75, 8))
    ax1.set_yticklabels(['0', '', '', '', '1', '', '', ''])
    if normalize:
        ax1.set_ylim(-0.05*x.max(), 1.75*x.max())
    else:
        ax1.set_ylim(-0.05, 1.75)
    ax1.grid('on', alpha=0.15)

    # Lifetime sensors
    ax2 = plt.axes((0.07, 0.43, 0.5, 0.23))
    ax2.text(0.24, 0.8, "Fluorescence lifetime sensors",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 12},
             transform=ax2.transAxes)
    t = np.linspace(0, 10, 10000) # Nanoseconds
    illumination = illumination_amount * (t < 0.2)
    background_lifetime = 1
    sensor_lifetime = 3 + 1.5*(1 - analyte_amount)
    background_signal = illumination_amount * (
        0.2 * background_amount * np.exp(-t / background_lifetime))
    x = background_signal + illumination_amount * (
        0.8 * sensor_amount     * np.exp(-t / sensor_lifetime))
    t_samples, x_samples = t[::500], x[::500]
    ax2.fill_between(t-0.1, 1000*illumination-300, -500,
                     alpha=0.1, linewidth=0, color=(0.12, 0.47, 0.71))
    ax2.fill_between(t, background_signal,
                     alpha=0.03, linewidth=0, color='magenta')
    ax2.plot(t, x, linewidth=2)
    ax2.plot(t_samples, x_samples, marker='.', color='blue')
    ref = np.linspace(x_samples[0], x_samples[-1], len(t_samples))
    ax2.plot(t_samples, ref, ':', color='black', linewidth=1, alpha=0.5)
    ax2.plot(t_samples, ref + (x_samples - ref).min(),
             ':', color='black', linewidth=1, alpha=0.5)

    ax2.set_xlabel("Time (nanoseconds)", weight='bold', labelpad=-1)
    ax2.set_xlim(-0.5, 10)

    ax2.text(-0.1, 0.3, "Signal", rotation=90,
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 10},
             transform=ax2.transAxes)
    ax2.set_yticks(np.linspace(0, 1.75, 8))
    ax2.set_yticklabels(['0', '', '', '', '1', '', '', ''])
    if normalize:
        ax2.set_ylim(-0.05*x.max(), 1.1*x.max())
    else:
        ax2.set_ylim(-0.05, 1.1)

    ax2.grid('on', alpha=0.15)

    # Relaxation sensors
    ax3 = plt.axes((0.07, 0.1, 0.5, 0.23))
    ax3.text(0.24, 0.8, "Isomer relaxation sensors",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 12},
             transform=ax3.transAxes)

    t = np.linspace(-3, 16, 10000) # Seconds
    illumination = np.zeros_like(t)
    pulse_duration = 0.2
    pulse_times = np.concatenate((
        np.linspace(t[0], 0, 10), np.array([10])))
    for pt in pulse_times:
        illumination[(t > pt) & (t < pt + pulse_duration)
                     ] = illumination_amount
    def activation_rate(current_time, activated_sensor):
        current_illumination = illumination[np.searchsorted(t, current_time)]
        k_off = 0.2 + 0.35 * (analyte_amount - 1)
        k_on = (5*current_illumination *
                (0.2 + 0.35 * (analyte_amount - 1)))
        return (k_on * (sensor_amount - activated_sensor) - 
                k_off * activated_sensor)
    sol = solve_ivp(
        activation_rate, [t[0], t[-1]], [0], t_eval=t, max_step=0.01)
    activated_sensor = sol.y[0]
    background_signal = illumination_amount * 0.2 * background_amount
    x = background_signal + illumination_amount * 0.8 * activated_sensor
    t_samples, x_samples = [], []
    for pt in pulse_times:
        ti = np.searchsorted(t, pt)
        tf = np.searchsorted(t, pt+pulse_duration)
        t_samples.append(t[ti:tf].mean())
        x_samples.append(x[ti:tf].mean())


    ax3.fill_between(t, 1000*illumination-300, -500,
                     alpha=0.1, linewidth=0, color=(0.12, 0.47, 0.71))
    ax3.fill_between(t, background_signal,
                     alpha=0.03, linewidth=0, color='magenta')
    ax3.plot(t, x, linewidth=1, color='blue')
    ax3.plot(t_samples, x_samples, linewidth=0, marker='.', color='blue')
    ax3.plot(t_samples[-2:], [x_samples[-2]]*2,
             ':', color='black', linewidth=1, alpha=0.5)
    ax3.plot(t_samples[-2:], [x_samples[-1]]*2,
             ':', color='black', linewidth=1, alpha=0.5)
    ref = np.linspace(x_samples[0], x_samples[-2], len(t_samples)-1)
    ax3.plot(t_samples[0:-1], ref, ':', color='black', linewidth=1, alpha=0.5)
    ax3.plot(t_samples[0:-1], ref + (x_samples[:-1] - ref).max(),
             ':', color='black', linewidth=1, alpha=0.5)
    ax3.set_xlabel("Time (seconds)", weight='bold')
    ax3.set_xlim(-3.5, 17)
    ax3.set_xticks(np.arange(-3, 16, 3))

    ax3.text(-0.1, 0.3, "Signal", rotation=90,
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 10},
             transform=ax3.transAxes)
    ax3.set_ylim(0, 1)
    ax3.set_yticks(np.linspace(0, 1.75, 8))
    ax3.set_yticklabels(['0', '', '', '', '1', '', '', ''])
    if normalize: # TODO: Actually get this right ugh
        ax3.set_ylim(x.min() - 0.4412 * (x.max() - x.min()),
                     x.max() + 0.588 * (x.max() - x.min()))
    else:
        ax3.set_ylim(-0.05, 1.1)
    ax3.grid('on', alpha=0.15)

    plt.savefig(temp_dir / ("frame_%03i.png"%which_frame), dpi=200)
    plt.close(fig)

def make_gif(start_frame, end_frame, filename):
    # Animate the frames into a gif:
    palette = temp_dir / "palette.png"
    filters = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
    print("Converting pngs to %s..."%filename, end=' ')
    convert_command_1 = [
        'ffmpeg',
        '-f', 'image2',
        '-i', temp_dir / 'frame_%03d.png',
        '-vf', filters + ",palettegen",
        '-y', palette]
    convert_command_2 = [
        'ffmpeg',
        '-start_number', str(start_frame),
        '-framerate', '25',
        '-f', 'image2',
        '-i', temp_dir / 'frame_%03d.png',
        '-i', palette,
        '-vframes', str(end_frame-start_frame),
        '-lavfi', filters + " [x]; [x][1:v] paletteuse",
        '-y', output_dir / filename]
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

main()
