from functools import partial
import threading as th
from time import sleep
import multiprocessing as mp

from devices import INIT_CAMERA_PARAMETERS, EnumParameterPosition
from devices.CameraProcess import CameraCtrl
NAME_DEVICES_THREAD = 'th_devices'


class ThreadDevices(th.Thread):
    def __init__(self, path_to_save) -> None:
        super().__init__()
        self.name = NAME_DEVICES_THREAD
        self.daemon = False

        # Init two cameras - Panchromatic and Monochromatic
        params = INIT_CAMERA_PARAMETERS
        params['ffc_mode'] = 'auto'
        params['ffc_period'] = 3600  # ffc every one minute

        # Hardware rate setter
        # If not connected, will be disregarded
        self._th_hardware_trigger = th.Thread(target=self.th_rpi_trigger_for_cam, daemon=True, name='hardware_trigger')

        # Init cameras
        # Cams are synced by the barrier, which releases all cams simultaniously when N_CAMERAS acquire it.
        N_CAMERAS = 2
        self._barrier_camera_sync = mp.Barrier(parties=N_CAMERAS)
        func_cam = partial(CameraCtrl, camera_parameters=params, is_dummy=False,
                           barrier_camera_sync=self._barrier_camera_sync,
                           time_to_save=5e9)  # dump to disk every 5 seconds
        self._camera_pan = func_cam(path_to_save=path_to_save / 'pan', name='pan')
        self._camera_mono = func_cam(path_to_save=path_to_save / 'mono', name='mono')

    def run(self) -> None:
        self._th_hardware_trigger.start()
        self._camera_mono.start()
        while self._camera_mono.camera_parameters_setting_position != EnumParameterPosition.DONE.value:
            sleep(1)
        self._camera_pan.start()

    def __del__(self) -> None:
        try:
            self._camera_mono.terminate()
        except:
            pass
        try:
            self._camera_pan.terminate()
        except:
            pass

    @staticmethod
    def th_rpi_trigger_for_cam():
        # The sampling rate of the cameras
        FREQUENCY_CAMERAS = 40  # Hz
        DUTY_CYCLE = 0.5
        period = 1 / FREQUENCY_CAMERAS
        low_time = period * DUTY_CYCLE  # seconds
        high_time = period - low_time  # seconds

        try:
            import RPi.GPIO as GPIO

            # Initialize the GPIO pin for the trigger signal
            PIN_TRIGGER = 17
            GPIO.setmode(GPIO.BCM)
            GPIO.cleanup()
            GPIO.setup(PIN_TRIGGER, GPIO.OUT)
        except:
            print('Could not initialize GPIO pin for trigger signal', flush=True)
            return
        print('Initialized hardware trigger for cameras', flush=True)

        # Define a function to toggle the trigger signal at the given rate
        while True:
            # Set the trigger pin to high
            GPIO.output(PIN_TRIGGER, GPIO.HIGH)
            # Wait for half the period
            sleep(high_time)
            # Set the trigger pin to low
            GPIO.output(PIN_TRIGGER, GPIO.LOW)
            # Wait for the other half of the period
            sleep(low_time)

    @property
    def rate_pan(self):
        return self._camera_pan.rate_camera

    @property
    def rate_mono(self):
        return self._camera_mono.rate_camera

    @property
    def n_files_pan(self):
        return self._camera_pan.n_files_saved

    @property
    def n_files_mono(self):
        return self._camera_mono.n_files_saved

    def _make_status(self, stat: int):
        stat = EnumParameterPosition(stat)
        if stat == EnumParameterPosition.DONE:
            return 'Ready'
        else:
            return stat.name

    @property
    def status_mono(self) -> str:
        return self._make_status(self._camera_mono._param_setting_pos.value)

    @property
    def status_pan(self) -> str:
        return self._make_status(self._camera_pan._param_setting_pos.value)
