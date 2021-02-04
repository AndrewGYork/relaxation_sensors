#!/usr/bin/python
# Dependencies from the python standard library:
import subprocess
from pathlib import Path
import shutil
# You can use 'pip' to install these dependencies:
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
from tifffile import imread, imwrite # v2020.6.3 or newer

input_dir = Path.cwd() / 'icons'
temp_dir = Path.cwd() / 'intermediate_output'
output_dir = Path.cwd()
# Sanity checks:
assert input_dir.is_dir()
temp_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

def main():
    sensor_parameters = dict(
        num_molecules=1000,
        initially_active=False,
        # Odds of activation/deactivation:
        p_activation_bound=    0.005,
        p_activation_unbound=  0.125,
        p_deactivation_bound=  0.001,
        p_deactivation_unbound=0.045)
    bound_sensors   = SwitchingSensors(
        initially_bound=True,  **sensor_parameters)
    unbound_sensors = SwitchingSensors(
        initially_bound=False, **sensor_parameters)

    # Keep track of a little bit of the simulation history:
    bound_emissions   = [  bound_sensors.expected_emissions('average')]
    unbound_emissions = [unbound_sensors.expected_emissions('average')]
    # Time-evolve the simulation and save animation frames:
    num_frames = 150
    initial_pause_frames = 30
    light_off_frame = 55
    for which_frame in range(initial_pause_frames, num_frames):
        # Animation
        save_animation_frame(
            which_frame=which_frame,
            initial_pause_frames=initial_pause_frames,
            light_off_frame=light_off_frame,
            num_frames=num_frames,
            bound_sensors=bound_sensors,
            unbound_sensors=unbound_sensors,
            bound_emissions=bound_emissions,
            unbound_emissions=unbound_emissions)
        # Simulation
        if which_frame == light_off_frame:
            for s in bound_sensors, unbound_sensors:
                s.p_activation_bound   = 0
                s.p_activation_unbound = 0
        bound_sensors.step()
        unbound_sensors.step()
        bound_emissions.append(bound_sensors.expected_emissions('average'))
        unbound_emissions.append(unbound_sensors.expected_emissions('average'))
        print('.', end='')
    # A long pause at the beginning:
    for which_frame in range(initial_pause_frames):
        shutil.copyfile(
            temp_dir / ('animation_frame_%06i.png'%initial_pause_frames),
            temp_dir / ('animation_frame_%06i.png'%which_frame))
    # Lazy way of normalizing the curves:
    print()
    print("bound_norm:",
          np.array(  bound_emissions).max() /   bound_sensors.num_molecules)
    print("unbound_norm:",
          np.array(unbound_emissions).max() / unbound_sensors.num_molecules)
    # Convert from png to gif and mp4:
    make_gif()
    make_mp4()
    return None

class SwitchingSensors:
    """
    Simple simulations of a multiply-stochastic process: stochastic
    photoswitching, stochastic binding, stochasitc photon emission.
    """
    def __init__(
        self,
        # Initial states:
        num_molecules=1,
        initially_active=False,
        initially_bound=False,
        # Odds of activation/deactivation:
        p_activation_bound=0,
        p_activation_unbound=0,
        p_deactivation_bound=0,
        p_deactivation_unbound=0,
        # Odds of binding/unbinding:
        p_binding_active=0,
        p_binding_inactive=0,
        p_unbinding_active=0,
        p_unbinding_inactive=0,
        # Brightness of each state:
        active_bound_brightness=1,
        active_unbound_brightness=1,
        inactive_bound_brightness=0,
        inactive_unbound_brightness=0,
        # Position and orientation of each sensor:
        x_positions=None,
        y_positions=None,
        angles=None,
        ):
        # Initial states:
        self.num_molecules = num_molecules
        self.active = np.full((num_molecules,), initially_active, dtype='bool')
        self.bound  = np.full((num_molecules,), initially_bound,  dtype='bool')
        # Average behavior:
        a, b, n = self.active, self.bound, self.num_molecules
        self.avg_active_bound_fraction     = np.count_nonzero( a &  b) / n
        self.avg_inactive_bound_fraction   = np.count_nonzero(~a &  b) / n
        self.avg_active_unbound_fraction   = np.count_nonzero( a & ~b) / n
        self.avg_inactive_unbound_fraction = np.count_nonzero(~a & ~b) / n
        # Odds of activation/deactivation:
        self.p_activation_bound     = p_activation_bound
        self.p_activation_unbound   = p_activation_unbound
        self.p_deactivation_bound   = p_deactivation_bound
        self.p_deactivation_unbound = p_deactivation_unbound
        # Odds of binding/unbinding:
        self.p_binding_active     = p_binding_active
        self.p_binding_inactive   = p_binding_inactive
        self.p_unbinding_active   = p_unbinding_active
        self.p_unbinding_inactive = p_unbinding_inactive
        # Brightness of each state:
        self.active_bound_brightness     = active_bound_brightness
        self.active_unbound_brightness   = active_unbound_brightness
        self.inactive_bound_brightness   = inactive_bound_brightness
        self.inactive_unbound_brightness = inactive_unbound_brightness
        # Position and orientation of each sensor:
        if x_positions is None:
            x_positions = np.random.random(num_molecules)
        if y_positions is None:
            y_positions = np.random.random(num_molecules)
        if angles is None:
            angles = 360 * np.random.random(num_molecules)
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.angles = angles
        return None

    def step(self):
        """Simulate binding changes and activation changes.
        """
        #####################################
        ###### Update stochastic state ######
        #####################################
        # Shorthand for the four states:
        ab =  self.active &  self.bound
        au =  self.active & ~self.bound
        ib = ~self.active &  self.bound
        iu = ~self.active & ~self.bound
        # Do we bind/unbind?
        x = np.random.random(len(self.bound))
        bound = np.empty_like(self.bound)
        bound[ab] = (x[ab] > self.p_unbinding_active)   # Fail to unbind
        bound[au] = (x[au] < self.p_binding_active)     # Bind
        bound[ib] = (x[ib] > self.p_unbinding_inactive) # Fail to unbind
        bound[iu] = (x[iu] < self.p_binding_inactive)   # Bind
        # Do we activate/deactivate?
        x = np.random.random(len(self.active))
        active = np.empty_like(self.active)
        active[ab] = (x[ab] > self.p_deactivation_bound)   # Fail to deactivate
        active[au] = (x[au] > self.p_deactivation_unbound) # Fail to deactivate
        active[ib] = (x[ib] < self.p_activation_bound)     # Activate
        active[iu] = (x[iu] < self.p_activation_unbound)   # Activate
        # Finishing touches:
        self.active = active
        self.bound = bound
        #####################################
        ###### Update average state    ######
        #####################################
        # Copies of the four state fractions w/short variable names:
        ab = float(self.avg_active_bound_fraction)
        ib = float(self.avg_inactive_bound_fraction)
        au = float(self.avg_active_unbound_fraction)
        iu = float(self.avg_inactive_unbound_fraction)
        # Four updates, each depends on four probabilities and three fractions:
        self.avg_active_bound_fraction += (
            + ib * self.p_activation_bound     # Activating
            - ab * self.p_deactivation_bound   # Deactivating
            + au * self.p_binding_active       # Binding
            - ab * self.p_unbinding_active)    # Unbinding
        self.avg_inactive_bound_fraction += (
            - ib * self.p_activation_bound     # Activating
            + ab * self.p_deactivation_bound   # Deactivating
            + iu * self.p_binding_inactive     # Binding
            - ib * self.p_unbinding_inactive)  # Unbinding
        self.avg_active_unbound_fraction += (
            + iu * self.p_activation_unbound   # Activating
            - au * self.p_deactivation_unbound # Deactivating
            - au * self.p_binding_active       # Binding
            + ab * self.p_unbinding_active)    # Unbinding
        self.avg_inactive_unbound_fraction += (
            - iu * self.p_activation_unbound   # Activating
            + au * self.p_deactivation_unbound # Deactivating
            - iu * self.p_binding_inactive     # Binding
            + ib * self.p_unbinding_inactive)  # Unbinding
        return None

    def num_active(self):
        return np.count_nonzero(self.active)

    def num_bound(self):
        return np.count_nonzero(self.bound)

    def num_emissions(self):
        return np.random.poisson(np.sum(
            self.active_bound_brightness     * ( self.active &  self.bound) +
            self.active_unbound_brightness   * ( self.active & ~self.bound) + 
            self.inactive_bound_brightness   * (~self.active &  self.bound) +
            self.inactive_unbound_brightness * (~self.active & ~self.bound)))

    def expected_emissions(self, state='average'):
        """On average, how many photons do we expect to emit?
        """
        if state == 'average': # Calculate from average behavior
            ab = self.avg_active_bound_fraction
            ib = self.avg_inactive_bound_fraction
            au = self.avg_active_unbound_fraction
            iu = self.avg_inactive_unbound_fraction
        elif state == 'stochastic': # Calculate from stochastic behavior
            a, b, n = self.active, self.bound, self.num_molecules
            ab = np.count_nonzero( a &  b) / n
            ib = np.count_nonzero(~a &  b) / n
            au = np.count_nonzero( a & ~b) / n
            iu = np.count_nonzero(~a & ~b) / n
        return self.num_molecules * (
            ab * self.active_bound_brightness +
            au * self.active_unbound_brightness +
            ib * self.inactive_bound_brightness +
            iu * self.inactive_unbound_brightness)


def save_animation_frame(
    which_frame,
    initial_pause_frames,    
    light_off_frame,
    num_frames,
    bound_sensors,
    unbound_sensors,
    bound_emissions,
    unbound_emissions,
    ):
    # Load sprites
    active_unbound_icon   = imread(input_dir / 'au.tif')
    active_bound_icon     = imread(input_dir / 'ab.tif')
    inactive_unbound_icon = imread(input_dir / 'iu.tif')
    inactive_bound_icon   = imread(input_dir / 'ib.tif')
    def pick_sprite(active, bound):
        if active:
            if bound:
                return active_bound_icon
            else:
                return active_unbound_icon
        else:
            if bound:
                return inactive_bound_icon
            else:
                return inactive_unbound_icon

    # Initialize figure
    fig_aspect = 2
    fig = plt.figure(figsize=(fig_aspect*5, 5))

    # Panel for state transition diagram
    ax0 = plt.axes([0.04 + 0.5/fig_aspect, 0.43, 0.94 - 1/fig_aspect, 0.57],
                   frameon=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.text(0.11, 0.72, "Active,\nbound")
    ax0.text(0.11, 0.22, "Inactive,\nbound")
    ax0.text(0.75, 0.72, "Active,\nunbound")
    ax0.text(0.75, 0.22, "Inactive,\nunbound")
    for im, x, y in ((active_bound_icon,     0.35, 0.75),
                     (active_unbound_icon,   0.65, 0.75),
                     (inactive_bound_icon,   0.35, 0.25),
                     (inactive_unbound_icon, 0.65, 0.25)):
        ax0.add_artist(AnnotationBbox(
            OffsetImage(im, zoom=1.5),
            (x, y), xycoords='data', frameon=False))
    arrow_params = dict(linewidth=1,
                        head_width=0.02,
                        head_length=0.025,
                        length_includes_head=True,
                        edgecolor='black',
                        facecolor='black',
                        shape='right')
    # Arrows for (un)binding rates:
    ax0.arrow(0.45, 0.26,  0.10, 0, **arrow_params)
    ax0.arrow(0.55, 0.24, -0.10, 0, **arrow_params)
    ax0.arrow(0.45, 0.76,  0.10, 0, **arrow_params)
    ax0.arrow(0.55, 0.74, -0.10, 0, **arrow_params)
    # Arrows for (de)activation rates:
    ax0.arrow(0.66, 0.5625,  0, -0.125, **arrow_params)
    ax0.arrow(0.36, 0.5250,  0, -0.075, **arrow_params)
    if initial_pause_frames < which_frame <= light_off_frame:
        arrow_params['facecolor'] = (0, 0.9, 0.9, 0.6)
        arrow_params['edgecolor'] = (0, 0.9, 0.9, 0.6)
        arrow_params['linewidth'] = 2.5
        ax0.arrow(0.64, 0.375,  0, 0.25, **arrow_params)
        ax0.arrow(0.34, 0.425,  0, 0.15, **arrow_params)
        ax0.text(0.17, 0.485, "Photoswitching", weight='bold', ha='center',
                 color=(0, 0.9, 0.9, 1))
    if which_frame > light_off_frame:
        ax0.text(0.17, 0.485, "Relaxation", weight='bold', ha='center')
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)

    # Panels for single-molecule photoswitching animations
    ax1 = plt.axes([0.005, 0.01, 0.5/fig_aspect, 0.93],
                    frameon=True, facecolor=(0.97, 0.97, 0.97))
    ax2 = plt.axes([0.995 - 0.5/fig_aspect, 0.01, 0.5/fig_aspect, 0.93],
                   frameon=True, facecolor=(0.97, 0.97, 0.97))
    ax1.set_title("100% bound")
    ax2.set_title("100% unbound")
    for ax, sensors, color, linestyle in (
        (ax1, bound_sensors, 'C0', (0, (0.02, 2))),
        (ax2, unbound_sensors, (1, 1, 0), 'solid')):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(6)
            sp.set_linestyle(linestyle)
        for m in range(sensors.num_molecules):
            im = pick_sprite(sensors.active[m], sensors.bound[m])
            im = rotate(im, sensors.angles[m], reshape=False)
            im = AnnotationBbox(
                OffsetImage(im, zoom=0.3),
                (sensors.x_positions[m], sensors.y_positions[m]),
                xycoords='data',
                frameon=False)
            ax.add_artist(im)

    # Panel for activation curves
    ax3 = plt.axes([0.045 +0.5/fig_aspect, 0.07, 0.93 -1/fig_aspect, 0.35],
                   frameon=True, facecolor=(0.92, 0.92, 0.92))
    ax3.add_patch(Rectangle( # Show when the illumination is on:
        (1, 0), light_off_frame-initial_pause_frames-1, 1.2,
        fill=True, linewidth=0, color=(0.05, 0.95, 0.95, 0.3)))

    bound_norm   = 0.1164006 *   bound_sensors.num_molecules
    unbound_norm = 0.7283212 * unbound_sensors.num_molecules
    ax3.plot(np.array(  bound_emissions) /   bound_norm,
             linewidth=4, linestyle=(0, (0.5, 0.6)), color='C0',
             label="100% bound")
    ax3.plot(np.array(unbound_emissions) / unbound_norm,
             linewidth=3, color=(1, 1, 0),
             label="100% unbound")
    ax3.set_xticks(np.arange(0, num_frames, 25))
    ax3.set_xticklabels(['' for x in ax3.get_xticklabels()])
    ax3.set_yticks(np.linspace(0, 1, 5))
    ax3.set_yticklabels(['' for x in ax3.get_yticklabels()])
    ax3.set_xlim(0, num_frames - initial_pause_frames - 30)
    ax3.set_ylim(0, 1.2)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Normalized signal")
    ax3.legend(loc=(0.585, 0.41), facecolor=(0.85, 0.85, 0.85))
    ax3.grid('on', alpha=0.1)
   
    plt.savefig(temp_dir / ('animation_frame_%06i.png'%(which_frame)),
                dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return None        

def make_gif():
    # Animate the frames into a gif:
    palette = temp_dir / "palette.png"
    filters = "scale=trunc(iw/3)*2:trunc(ih/3)*2:flags=lanczos"
    print("Converting pngs to gif...", end=' ')
    convert_command_1 = [
        'ffmpeg',
        '-f', 'image2',
        '-i', temp_dir / 'animation_frame_%06d.png',
        '-vf', filters + ",palettegen",
        '-y', palette]
    convert_command_2 = [
        'ffmpeg',
        '-framerate', '25',
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

def make_mp4():
    ## Make video from images
    print("Converting images to mp4...", end='')
    convert_command = [
       'ffmpeg', '-y',            # auto overwrite files
       '-r', '20',                # frame rate
       '-f', 'image2',            # format is image sequence
       '-i', temp_dir / 'animation_frame_%06d.png', # image sequence name
       '-movflags', 'faststart',  # internet optimisation(?)
       '-pix_fmt', 'yuv420p',     # cross browser compatibility
       '-vcodec', 'libx264',      # codec choice
       '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',#even pixel number (important)
       '-preset', 'veryslow',     # take time and compress to max
       '-crf', '25',              # image quality vs file size
       output_dir / 'figure.mp4'] # output file name
    try:
       with open(temp_dir / 'conversion_messages.txt', 'wt') as f:
           f.write("So far, everthing's fine...\n")
           f.flush()
           subprocess.check_call(convert_command, stderr=f, stdout=f)
           f.flush()
       (temp_dir / 'conversion_messages.txt').unlink()
    except: # This is unlikely to be platform independent :D
       print("MP4 conversion failed. Is ffmpeg installed?")
       raise
    print('done.')

if __name__ == '__main__':
    main()
