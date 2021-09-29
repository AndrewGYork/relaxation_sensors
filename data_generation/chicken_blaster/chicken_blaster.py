# Standard library
import threading
# Third-party
import numpy as np
# Our stuff from https://github.com/AndrewGYork/tools
import pco
import ni

# Name-to-channel; what is our analog-out card plugged into?
n2c = {'camera': 0,
       '488': 1,
       '405': 2,}

class ChickenBlaster:
    def __init__(self):
        self.ao = ni.Analog_Out(
            num_channels=3,
            rate=400e3,
            daq_type='6738',
            board_name='Dev1')
        self.camera = pco.Camera()
        self.camera.apply_settings(
            trigger='external_trigger',
            region_of_interest={'left': 1, 'right': 2048,
                                'top': 1, 'bottom': 2048})
        self.camera._set_timestamp_mode("binary+ASCII")
        # Our 405 LED's minimum pulse duration is 500 us. Also note that
        # the LED turns on fairly promptly (~10 us) at max power knob
        # setting, but lower settings can cause a variable delay in
        # on-switching. Probably simplest to always use max knob
        # setting.
        self.led_minimum_duration_s = {'405': 500e-6,
                                       '488':  56e-6}
        self.set_measurement_illumination_time_s(
            max(self.led_minimum_duration_s.values()))

    def close(self):
        self.camera.close()
        self.ao.close()

    def set_measurement_illumination_time_s(self, t, led=None):
        if led is None: # Limited by the slowest LED
            min_t = max(self.led_minimum_duration_s.values())
        else: # Limited by only one LED
            min_t = self.led_minimum_duration_s[led]
        assert t >= min_t
        self.camera.disarm()
        self.camera._set_exposure_time(1e6*t +
                                       self.camera.rolling_time_microseconds)
        self.camera.arm(16)
        self.measurement_illumination_time_s = t

    def acquire(self, sequence_of_parameter_dicts):
        # Calculate voltages
        voltages = []
        image_should_be_saved = []
        timeouts = [10]
        # Precalculations we'll probably use
        camera_jitter_s = 20e-6
        camera_jitter_pix = max(1, self.ao.s2p(camera_jitter_s))
        camera_exposure_pix = self.ao.s2p(
            1e-6*self.camera.exposure_time_microseconds)
        camera_rolling_pix = self.ao.s2p(
            1e-6*self.camera.rolling_time_microseconds)
        # Loop over input parameters:
        for d in sequence_of_parameter_dicts:
            d = dict(d)
            # Check arguments
            preframes = d.pop('preframes', 1)
            images = d.pop('images', 1)
            image_light = d.pop('image_light', '488')
            image_light_voltage = d.pop('image_light_voltage', 5)
            switching_light = d.pop('switching_light', None)
            switching_light_voltage = d.pop('switching_light_voltage', 5)
            switching_time_s = d.pop('switching_time_s', 0)
            assert len(d) == 0
            assert int(preframes) == preframes
            assert preframes >= 0
            assert int(images) == images
            assert images >= 0
            assert image_light in ('488', '405', None)
            assert 0 <= image_light_voltage <= 5
            assert switching_light in ('488', '405', None)
            if switching_light != '488':
                assert switching_light_voltage == 5
            assert 0 <= switching_light_voltage <= 5
            assert 0 <= switching_time_s < 30 # If you really need >30, go 2x
            if switching_light is not None:
                assert (switching_time_s >
                        self.led_minimum_duration_s[switching_light])
            # Construct voltage sequence
            #### Preframes
            v = np.zeros((camera_exposure_pix + camera_jitter_pix,
                          self.ao.num_channels),
                         dtype='float64')
            v[:camera_rolling_pix, n2c['camera']] = 4 # Trigger the camera
            for i in range(preframes):
                voltages.append(v)
                image_should_be_saved.append(False)
                timeouts.append(2)
            #### Images
            v = np.zeros((camera_exposure_pix + camera_jitter_pix,
                          self.ao.num_channels),
                         dtype='float64')
            v[:camera_rolling_pix, n2c['camera']] = 4 # Trigger the camera
            if image_light is not None: # ...while illuminating the sample
                v[camera_rolling_pix:camera_exposure_pix,
                  n2c[image_light]] = image_light_voltage
            for i in range(images):
                voltages.append(v)
                image_should_be_saved.append(True)
                timeouts.append(2)
            #### Wait for the camera to unroll
            v = np.zeros((camera_rolling_pix, self.ao.num_channels),
                         dtype='float64')
            voltages.append(v)
            #### Allow the sample to (photo)switch
            v = np.zeros((self.ao.s2p(switching_time_s), self.ao.num_channels),
                         dtype='float64')
            if switching_light is not None: # Illuminate the sample
                v[:, n2c[switching_light]] = switching_light_voltage
            voltages.append(v)
            timeouts[-1] = switching_time_s + 2
        del timeouts[-1]
        voltages = np.concatenate(voltages, axis=0)
        voltages[-1, :] = 0 # Always end on a safe voltage
        
        # Acquire data
        data = []
        def record():
            for save, t in zip(image_should_be_saved, timeouts):
                try:
                    im = self.camera.record_to_memory(
                        num_images=1, first_trigger_timeout_seconds=t)
                except pco.TimeoutError:
                    print(i)
                    raise
                if save:
                    data.append(im)
        th = threading.Thread(target=record)
        th.start()
        self.ao.play_voltages(voltages)
        th.join()
        data = np.concatenate(data, axis=0)
        data = data.reshape(data.shape[0], 1, 1, data.shape[1], data.shape[2])
        return data

if __name__ == '__main__':
    import tifffile as tf
    
    blaster = ChickenBlaster()
    blaster.set_measurement_illumination_time_s(500e-6)

    data = blaster.acquire(
        15*[{'switching_light': '488', 'switching_time_s': 0.5}] +
        25*[{'switching_light': None, 'switching_time_s': 1}] +
        10 *[{'switching_light': '405', 'switching_time_s': 0.5}] +
        25*[{'switching_light': None, 'switching_time_s': 1}] 
        )
    tf.imwrite('out.tif', data, imagej=True)
    blaster.close()

