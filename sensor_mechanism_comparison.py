#!/usr/bin/python
# Dependencies from the python standard library:
import subprocess
from pathlib import Path
# You can use 'pip' to install these dependencies:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

temp_dir = Path.cwd() / 'intermediate_output'
output_dir = Path.cwd()
# Sanity checks:
temp_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Script
# Start with intensity sensors
# Then show lifetime sensors
# Then show countdown sensors

def main():
    which_frame = 0
    make_frame(which_frame=which_frame)
    for _ in range(5):
            which_frame += 1
            make_frame(which_frame=which_frame)
    for _ in range(2):
        for i in range(15):
            which_frame += 1
            sensor_amount=0.46 + 0.54*(0.5 - 0.5*np.cos(i*2*np.pi/15))
            make_frame(
                which_frame=which_frame,
                sensor_amount=sensor_amount,
                amplitude=sensor_amount*0.46 + 0.46
                )
    for _ in range(5):
            which_frame += 1
            make_frame(which_frame=which_frame)
    for _ in range(2):
        for i in range(15):
            which_frame += 1
            analyte_amount=0.46 + 0.54*(0.5 - 0.5*np.cos(i*2*np.pi/15))
            make_frame(
                which_frame=which_frame,
                analyte_amount=analyte_amount,
                amplitude=0.46*analyte_amount + 0.46
                )
    for _ in range(5):
            which_frame += 1
            make_frame(which_frame=which_frame)
    for _ in range(2):
        for i in range(15):
            which_frame += 1
            background_amount=0.46 + 0.54*(0.5 - 0.5*np.cos(i*2*np.pi/15))
            make_frame(
                which_frame=which_frame,
                background_amount=background_amount,
                amplitude=0.46*0.46 + background_amount
                )
    for _ in range(5):
            which_frame += 1
            make_frame(which_frame=which_frame)
    tf_frames = 10e-9 * (1.3)**(np.arange(15, -1, -1))
    t0_frames = 10e-9 - tf_frames
    title_alpha = np.linspace(1, 0, 15)
    for t0_frame, tf_frame, alpha in zip(t0_frames, tf_frames, title_alpha):
        which_frame += 1
        make_frame(which_frame=which_frame,
                   title_1_alpha=alpha,
                   title_2_alpha=1-alpha,
                   t0_frame=t0_frame,
                   tf_frame=tf_frame,
                   tick_delta=1e-9)
    for _ in range(15):
        which_frame += 1
        make_frame(which_frame=which_frame,
                   title_1='fluorescence lifetime',
                   t0_frame=0,
                   tf_frame=10e-9,
                   tick_delta=1e-9)
    for _ in range(2):
        for i in range(15):
            which_frame += 1
            sensor_amount=0.46 + 0.54*(0.5 - 0.5*np.cos(i*2*np.pi/15))
            make_frame(which_frame=which_frame,
                       title_1='fluorescence lifetime',
                       sensor_amount=sensor_amount,
                       amplitude=sensor_amount*0.46 + 0.46,
                       t0_frame=0,
                       tf_frame=10e-9,
                       tick_delta=1e-9)
    for _ in range(5):
        which_frame += 1
        make_frame(which_frame=which_frame,
                   title_1='fluorescence lifetime',
                   t0_frame=0,
                   tf_frame=10e-9,
                   tick_delta=1e-9)
    for _ in range(2):
        for i in range(15):
            which_frame += 1
            analyte_amount=0.46 + 0.54*(0.5 - 0.5*np.cos(i*2*np.pi/15))
            make_frame(which_frame=which_frame,
                       title_1='fluorescence lifetime',
                       analyte_amount=analyte_amount,
                       T=3e-9 - 3e-9*(analyte_amount-0.46),
                       t0_frame=0,
                       tf_frame=10e-9,
                       tick_delta=1e-9)


def make_frame(
    which_frame,
    title_1="fluorescence intensity",
    title_2="fluorescence lifetime",
    title_1_alpha=1,
    title_2_alpha=0,
    sensor_amount=0.46,
    analyte_amount=0.46,
    background_amount=0.46,
    amplitude=0.6716,
    T=3e-9,
    t0=0,
    tf=10e-9,
    t0_frame=-1,
    tf_frame=1,
    tick_delta=10e-9,
    ):
    print('.', sep='', end='')
    fig = plt.figure(figsize=(6, 3.6))
    ax1 = plt.axes((0, 0, 1, 1))
    for title, title_alpha in ((title_1, title_1_alpha),
                               (title_2, title_2_alpha)):
        ax1.text(0.04, 0.94, "Biosensor readout: " + title,
                 fontdict={'color': (0, 0, 0, title_alpha),
                           'weight': 'bold',
                           'size': 11})

    ax1.add_patch(Rectangle(
        (0.6, 0.3), 0.07, 0.6*sensor_amount,
        fill=True, linewidth=0,
        color=(0, 0.9, 0, 0.5)))
    ax1.add_patch(Rectangle(
        (0.6, 0.3), 0.07, 0.6,
        fill=False, linewidth=1))
    ax1.text(0.624, 0.33, "Amount of sensor", rotation=90,
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 11})

    ax1.add_patch(Rectangle(
        (0.72, 0.3), 0.07, 0.6*analyte_amount,
        fill=True, linewidth=0,
        color=(0, 0, 0.9, 0.5)))
    ax1.add_patch(Rectangle(
        (0.72, 0.3), 0.07, 0.6,
        fill=False, linewidth=1))
    ax1.text(0.744, 0.33, "Amount of analyte", rotation=90,
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 11})

    ax1.add_patch(Rectangle(
        (0.84, 0.3), 0.07, 0.6*background_amount,
        fill=True, linewidth=0,
        color=(0.9, 0, 0.9, 0.5)))
    ax1.add_patch(Rectangle(
        (0.84, 0.3), 0.07, 0.6,
        fill=False, linewidth=1))
    ax1.text(0.864, 0.33, "Amount of background", rotation=90,
             fontdict={'color': (0, 0, 0),
                       'weight': 'bold',
                       'size': 11})
    
    ax2 = plt.axes((0.08, 0.15, 0.7, 0.7))
    t = np.linspace(t0, tf, 1000)
    x = amplitude * np.exp(-t/T)
    ax2.plot(t, x, linewidth=4)

    ax2.set_xlabel("Time (s)", weight='bold')
    ax2.set_xlim(t0_frame, tf_frame)
    ax2.set_xticks(np.arange(t[0], t[-1], tick_delta))

    ax2.set_ylabel("Emitted light intensity", weight='bold')
    ax2.set_ylim(-0.05, 1.5)
    ax2.set_yticks(np.linspace(0, 1.5, 5))
    ax2.set_yticklabels(['' for t in ax2.get_yticks()])

    ax2.set_frame_on(False)
    ax2.grid('on', alpha=0.15)
    plt.savefig(temp_dir / ("frame_%03i.png"%which_frame), dpi=200)
    plt.close(fig)

main()
