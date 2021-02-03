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
    

    sensors_1 = SwitchingSensors(
        num_molecules=1000,
        initially_active=False,
        initially_bound=False,
        # Odds of activation/deactivation:
        p_activation_bound=    0.005,
        p_activation_unbound=  0.125,
        p_deactivation_bound=  0.001,
        p_deactivation_unbound=0.030,
        # Odds of binding/unbinding:
        p_binding_active=0,
        p_binding_inactive=0,
        p_unbinding_active=0,
        p_unbinding_inactive=0)
    
    sensors_2 = SwitchingSensors(
        num_molecules=1000,
        initially_active=False,
        initially_bound=True,
        # Odds of activation/deactivation:
        p_activation_bound=    0.005,
        p_activation_unbound=  0.125,
        p_deactivation_bound=  0.001,
        p_deactivation_unbound=0.030,
        # Odds of binding/unbinding:
        p_binding_active=0,
        p_binding_inactive=0,
        p_unbinding_active=0,
        p_unbinding_inactive=0)

    # Remember a little bit of the simulation history:
    emissions_1 = [sensors_1.expected_emissions('average')]
    emissions_2 = [sensors_2.expected_emissions('average')]
    # Start time-evolving the simulation and saving animation frames:
    num_frames = 110
    light_off_frame = 25
    for i in range(num_frames):
        # Figure generation
        fig_aspect = 2
        fig = plt.figure(figsize=(fig_aspect*5, 5))
        ax0 = plt.axes([0.04 + 0.5/fig_aspect, 0.43, 0.94 - 1/fig_aspect, 0.57],
                       frameon=False)
        ax0.set_xticks([])
        ax0.set_yticks([])
        for im, x, y in ((active_bound_icon,     0.65, 0.75),
                         (active_unbound_icon,   0.35, 0.75),
                         (inactive_bound_icon,   0.65, 0.25),
                         (inactive_unbound_icon, 0.35, 0.25)):
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
        ax0.arrow(0.36, 0.5625,  0, -0.125, **arrow_params)
        ax0.arrow(0.66, 0.5250,  0, -0.075, **arrow_params)
        if 0 < i < light_off_frame:
            arrow_params['facecolor'] = (0, 0.9, 0.9, 0.6)
            arrow_params['edgecolor'] = (0, 0.9, 0.9, 0.6)
            arrow_params['linewidth'] = 2.5
            ax0.arrow(0.34, 0.375,  0, 0.25, **arrow_params)
            ax0.arrow(0.64, 0.425,  0, 0.15, **arrow_params)

        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1)
        ax1 = plt.axes([0.005, 0.01, 0.5/fig_aspect, 0.98],
                       frameon=True)
        ax2 = plt.axes([0.995 - 0.5/fig_aspect, 0.01, 0.5/fig_aspect, 0.98],
                        frameon=True)
        for ax, sensors, color, linestyle in (
            (ax1, sensors_1, 'C0', 'solid'),
            (ax2, sensors_2, 'C1', (0, (0.02, 2)))):
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
        ax3 = plt.axes([0.045 +0.5/fig_aspect, 0.07, 0.93 -1/fig_aspect, 0.35],
                       frameon=True)
        ax3.add_patch(Rectangle( # Show when the illumination is on:
            (1, 0), light_off_frame-2, 1.2,
            fill=True, linewidth=0, color=(0, 0.9, 0.9, 0.2)))

        norm_1 = 0.792289 * sensors_1.num_molecules
        norm_2 = 0.112073 * sensors_2.num_molecules
        ax3.plot(np.array(emissions_1) / norm_1,
                 linewidth=2)
        ax3.plot(np.array(emissions_2) / norm_2,
                 linewidth=3, linestyle=(0, (0.5, 0.6)))
        ax3.set_xticks(np.linspace(0, num_frames, 7))
        ax3.set_xticklabels(['' for x in ax3.get_xticklabels()])
        ax3.set_yticks(np.linspace(0, 1, 5))
        ax3.set_yticklabels(['' for x in ax3.get_yticklabels()])
        ax3.set_xlim(0, num_frames - 31)
        ax3.set_ylim(0, 1.2)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Normalized signal")
        ax3.grid('on', alpha=0.1)
       
        plt.savefig(temp_dir / ('animation_frame_%06i.png'%(i+30)))
        plt.close()

        # Simulation
        if i == (light_off_frame-1):
            for s in sensors_1, sensors_2:
                s.p_activation_bound   = 0
                s.p_activation_unbound = 0
        sensors_1.step()
        sensors_2.step()
        emissions_1.append(sensors_1.expected_emissions('average'))
        emissions_2.append(sensors_2.expected_emissions('average'))
        print('.', end='')
    for i in range(30):
        shutil.copyfile(
            temp_dir / ('animation_frame_%06i.png'%30),
            temp_dir / ('animation_frame_%06i.png'%i))
##        shutil.copyfile(
##            temp_dir / ('animation_frame_%06i.png'%(29+num_frames)),
##            temp_dir / ('animation_frame_%06i.png'%(30+num_frames + i)))
    print()
    print("norm_1:", np.array(emissions_1).max() / sensors_1.num_molecules)
    print("norm_2:", np.array(emissions_2).max() / sensors_2.num_molecules)



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
        self.expected_active_fraction = float(initially_active)
        self.expected_bound_fraction  = float(initially_bound)
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
        # Shorthand for the four state fractions:
        a, b = self.expected_active_fraction, self.expected_bound_fraction
        ab =    a  *    b
        au =    a  * (1-b)
        ib = (1-a) *    b
        iu = (1-a) * (1-b)
        # How does binding change, on average?
        self.expected_bound_fraction += (
            - self.p_unbinding_active   * ab  # Unbinding
            + self.p_binding_active     * au  # Binding
            - self.p_unbinding_inactive * ib  # Unbinding
            + self.p_binding_inactive   * iu) # Binding
        # How does activation change, on average?
        self.expected_active_fraction += (
            - self.p_deactivation_bound   * ab  # Deactivation
            - self.p_deactivation_unbound * au  # Deactivation
            + self.p_activation_bound     * ib  # Activation
            + self.p_activation_unbound   * iu) # Activation
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
            a = self.expected_active_fraction
            b = self.expected_bound_fraction
        elif state == 'stochastic': # Calculate from stochastic behavior
            a = self.num_active() / self.num_molecules
            b = self.num_bound() / self.num_molecules
        return self.num_molecules * (
            (  a) * (  b) * self.active_bound_brightness +
            (  a) * (1-b) * self.active_unbound_brightness +
            (1-a) * (  b) * self.inactive_bound_brightness +
            (1-a) * (1-b) * self.inactive_unbound_brightness)
        

if __name__ == '__main__':
    main()
