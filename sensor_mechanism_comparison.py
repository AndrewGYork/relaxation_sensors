#!/usr/bin/python
# Dependencies from the python standard library:
import subprocess
from pathlib import Path
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

temp_dir = Path.cwd() / 'intermediate_output'
output_dir = Path.cwd()
# Sanity checks:
temp_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

def main():
    which_frame = -1
    for amount in np.concatenate((np.linspace(1, 2, 5),
                                  np.linspace(2, 0, 10),
                                  np.linspace(0, 1, 5))
                                 ):
        which_frame += 1
        make_frame(
            which_frame=which_frame,
            analyte_amount=1,
            background_amount=1,
            sensor_amount=amount,
            illumination_amount=1,
            normalize=True
            )

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

    # Amount of background
    ax0.add_patch(Rectangle(
        (0.62, 0.6), 0.35*(0.5*background_amount), 0.07,
        fill=True, linewidth=0,
        color=(1, 0, 1, 0.5)))
    ax0.add_patch(Rectangle(
        (0.62, 0.6), 0.35, 0.07,
        fill=False, linewidth=1))
    ax0.text(0.68, 0.625, "Amount of background",
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
        (0.62, 0.2), 0.35*(0.5*illumination_amount), 0.07,
        fill=True, linewidth=0,
        color=(0.12, 0.47, 0.71, 0.15)))
    ax0.add_patch(Rectangle(
        (0.62, 0.2), 0.35, 0.07,
        fill=False, linewidth=1))
    ax0.text(0.68, 0.225, "Amount of illumination",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 11})


    # Intensity sensors
    ax1 = plt.axes((0.07, 0.76, 0.5, 0.23))
    ax1.text(0.24, 0.8, "Fluorescence intensity",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 12},
             transform=ax1.transAxes)
    t = np.linspace(-5, 100, 10000)
    illumination = illumination_amount * (t > 0) * (t < 10)
    x = illumination * (
        0.2 * background_amount +
        0.8 * sensor_amount * (0.5 + 0.5*analyte_amount))
    ax1.fill_between(t, illumination, alpha=0.1)
    ax1.plot(t, x, linewidth=2)
    ax1.plot(5, x.max(), marker='.', color='blue')

    ax1.set_xlabel("Time (milliseconds)", weight='bold', labelpad=-1)
    ax1.set_xlim(-10, 100)

    ax1.set_ylabel("Signal", weight='bold')
    ax1.set_yticks(np.linspace(0, 1.75, 8))
    ax1.set_yticklabels(['0', '', '', '', '1', '', ''])
    if normalize:
        ax1.set_ylim(-0.05*x.max(), 1.75*x.max())
    else:
        ax1.set_ylim(-0.05, 1.75)
    ax1.grid('on', alpha=0.15)

    # Lifetime sensors
    ax2 = plt.axes((0.07, 0.43, 0.5, 0.23))
    ax2.text(0.24, 0.8, "Fluorescence lifetime",
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 12},
             transform=ax2.transAxes)
    t = np.linspace(0, 10, 10000) # Nanoseconds
    illumination = illumination_amount * (t < 0.2)
    background_lifetime = 1
    sensor_lifetime = 3 + 1.5*(1 - analyte_amount)
    x = illumination_amount * (
        0.2 * background_amount * np.exp(-t / background_lifetime) +
        0.8 * sensor_amount     * np.exp(-t / sensor_lifetime))
    ax2.fill_between(t-0.1, illumination, alpha=0.1)
    ax2.plot(t, x, linewidth=2)
    ax2.plot(t[::500], x[::500], marker='.', color='blue')

    ax2.set_xlabel("Time (nanoseconds)", weight='bold', labelpad=-1)
    ax2.set_xlim(-0.5, 10)

    ax2.set_ylabel("Signal", weight='bold')
    ax2.set_yticks(np.linspace(0, 1.75, 8))
    ax2.set_yticklabels(['0', '', '', '', '1', '', ''])
    if normalize:
        ax2.set_ylim(-0.05*x.max(), 1.75*x.max())
    else:
        ax2.set_ylim(-0.05, 1.75)

    ax2.grid('on', alpha=0.15)

    # Relaxation sensors
    ax3 = plt.axes((0.07, 0.1, 0.5, 0.23))
    ax3.text(0.24, 0.8, "Isomer lifetime",
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
        k_off = activated_sensor * (1/5 + (analyte_amount - 1) * 1/5)
        k_on = current_illumination * (sensor_amount - activated_sensor) * 1.5
        return k_on - k_off
    sol = solve_ivp(
        activation_rate, [t[0], t[-1]], [0], t_eval=t, max_step=0.01)
    activated_sensor = sol.y[0]
    x = 1.2 * illumination_amount * (0.2 * background_amount +
                                     0.8 * activated_sensor)
    t_samples, x_samples = [], []
    for pt in pulse_times:
        ti = np.searchsorted(t, pt)
        tf = np.searchsorted(t, pt+pulse_duration)
        t_samples.append(t[ti:tf].mean())
        x_samples.append(x[ti:tf].mean())


    ax3.fill(t, illumination, alpha=0.1)
    ax3.plot(t, x, linewidth=1, color='blue')
    ax3.plot(t_samples, x_samples, linewidth=0, marker='.', color='blue')
    ax3.set_xlabel("Time (seconds)", weight='bold')
    ax3.set_xlim(-3.5, 17)
    ax3.set_xticks(np.arange(-3, 16, 3))

    ax3.set_ylabel("Signal", weight='bold')
    ax3.set_ylim(0, 1)
    ax3.set_yticks(np.linspace(0, 1.75, 8))
    ax3.set_yticklabels(['0', '', '', '', '1', '', ''])
    if normalize: # TODO: Actually get this right ugh
        ax3.set_ylim(-0.05 * 0.96 * (x.max() - x.min()) + (x.min() - 0.24),
                     1.75 * 0.96 * (x.max() - x.min()) + (x.min() - 0.24))
    else:
        ax3.set_ylim(-0.05, 1.75)

    ax3.grid('on', alpha=0.15)


    plt.savefig(temp_dir / ("frame_%03i.png"%which_frame), dpi=200)
    plt.close(fig)

main()
