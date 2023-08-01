from ctypes import c_uint
from functools import partial
import threading as th
from time import sleep
import multiprocessing as mp

from devices import INIT_CAMERA_PARAMETERS, EnumParameterPosition
from devices.CameraProcess import CameraCtrl
from utils.tools import DummyGPIO
NAME_DEVICES_THREAD = 'th_devices'


try:
    import RPi.GPIO as GPIO
    print('Loaded GPIO module for RPi', flush=True)
except (ModuleNotFoundError, RuntimeError):
    print('Could not load GPIO module for RPi', flush=True)
    GPIO = DummyGPIO()


class ThreadDevices(th.Thread):
    def __init__(self, path_to_save: mp.Value) -> None:
        super().__init__()
        self.name = NAME_DEVICES_THREAD
        self.daemon = False

        # Init two cameras - Panchromatic and Monochromatic
        params = INIT_CAMERA_PARAMETERS
        params['ffc_mode'] = 'auto'
        params['ffc_period'] = 3600  # ffc every one minute

        # Hardware rate setter, if not connected, will be disregarded
        # The hardware trigger is a square wave with a given frequency and duty cycle
        # The cameras are triggered by the rising edge of the square wave
        # The trigger and cameras are synced by the barrier, which releases all cams simultaneously.
        # Before the barrier is released, the cameras are purged.
        N_CAMERAS = 2
        N_CTRL_THREADS = 1
        self._counter: mp.Value = mp.Value(c_uint)
        self._counter.value = 0
        self._barrier_camera_sync = mp.Barrier(parties=N_CAMERAS + N_CTRL_THREADS)
        self._mp_hardware_trigger = mp.Process(target=self.mp_rpi_trigger_for_cam, daemon=True, name='hardware_trigger')

        # Init cameras
        # Cams are synced by the barrier, which releases all cams simultaneously when N_CAMERAS acquire it.
        func_cam = partial(CameraCtrl, camera_parameters=params, is_dummy=False,
                           barrier_camera_sync=self._barrier_camera_sync, path_to_save=path_to_save,
                           counter_frames=self._counter, time_to_save=2e9)  # dump to disk every 2 seconds
        self._camera_pan = func_cam(name='pan')
        self._camera_mono = func_cam(name='mono')

    def run(self) -> None:
        self._mp_hardware_trigger.start()
        self._camera_mono.start()
        while self._camera_mono.camera_parameters_setting_position != EnumParameterPosition.DONE.value:
            sleep(1)
        self._camera_pan.start()

    def __del__(self) -> None:
        try:
            self._camera_mono.terminate()
        except Exception:
            pass
        try:
            self._camera_pan.terminate()
        except Exception:
            pass
        GPIO.cleanup()

    def mp_rpi_trigger_for_cam(self):
        # The sampling rate of the cameras
        FREQUENCY_CAMERAS = 30  # Hz
        DUTY_CYCLE = 0.01  # 1% duty cycle
        period = 1 / FREQUENCY_CAMERAS
        low_time = period * DUTY_CYCLE  # seconds
        high_time = period - low_time  # seconds

        # Configure the trigger pin on the RPi
        PIN_TRIGGER = 17
        GPIO.setmode(GPIO.BCM)
        GPIO.cleanup()
        GPIO.setup(PIN_TRIGGER, GPIO.OUT)
        GPIO.output(PIN_TRIGGER, GPIO.HIGH)  # Set the trigger pin to high -> no trigger
        self._barrier_camera_sync.wait()  # Wait for all camera processes to init and acquire the barrier

        # Define a function to toggle the trigger signal at the given rate
        while True:
            # Wait for all cameras to reach the barrier so its released.
            self._barrier_camera_sync.wait(timeout=None)

            # Count before triggering, so the cameras are always synced to the correct counter value
            self._counter.value = self._counter.value + 1
            # Set the trigger pin to low -> trigger
            GPIO.output(PIN_TRIGGER, GPIO.LOW)
            # Wait for the low_time of duty cycle
            sleep(low_time)
            # Set the trigger pin to high -> no trigger
            GPIO.output(PIN_TRIGGER, GPIO.HIGH)
            # Wait for high_time of duty cycle
            sleep(high_time)

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

    @property
    def frame(self):
        return self._camera_pan.frame
