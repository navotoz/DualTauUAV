from ctypes import c_float
from datetime import datetime
from functools import partial
from pathlib import Path
import threading as th
import multiprocessing as mp
from time import sleep

from devices import INIT_CAMERA_PARAMETERS, EnumParameterPosition
from devices.CameraProcess import CameraCtrl
from devices.utils import DummyGPIO
NAME_DEVICES_THREAD = 'th_devices'


try:
    import RPi.GPIO as GPIO
    print('Loaded GPIO module for RPi', flush=True)
    from gpiozero import CPUTemperature
    print('Loaded gpiozero module for RPi', flush=True)
except (ModuleNotFoundError, RuntimeError):
    print('Could not load GPIO module for RPi', flush=True)
    GPIO = DummyGPIO()
    CPUTemperature = DummyGPIO


class ThreadDevices(th.Thread):
    def __init__(self, path_to_save: mp.Value) -> None:
        super().__init__()
        self.name = NAME_DEVICES_THREAD
        self.daemon = False

        # Init two cameras - Panchromatic and Monochromatic
        params = INIT_CAMERA_PARAMETERS
        params['ffc_mode'] = 'manual'  # manually control the ffc from the CameraProcess
        params['ffc_period'] = 0

        # Init cameras
        func_cam = partial(CameraCtrl,
                           camera_parameters=params,
                           is_dummy=False,
                           path_to_save=path_to_save,
                           time_to_save=2e9)  # dump to disk every 2 seconds
        self._camera = func_cam(name='pan')
        self._path_to_files = path_to_save

        # Collect RPi temperature
        self._mp_rpi_temp = mp.Process(target=self._rpi_temp, daemon=True, name='rpi_temp')
        self._t_rpi = mp.Value(c_float)
        self._t_rpi.value = 0.0

    def run(self) -> None:
        self._mp_rpi_temp.start()
        self._camera.start()

    def __del__(self) -> None:
        try:
            self._camera.terminate()
        except Exception:
            pass

    @property
    def rate(self):
        return self._camera.rate_camera

    @property
    def n_files(self):
        return self._camera.n_files_saved

    def _make_status(self, stat: int):
        stat = EnumParameterPosition(stat)
        if stat == EnumParameterPosition.DONE:
            return 'Ready'
        else:
            return stat.name

    @property
    def status(self) -> str:
        return self._make_status(self._camera._param_setting_pos.value)

    @property
    def frame(self):
        return self._camera_pan.frame

    def _rpi_temp_func(self):
        cpu = CPUTemperature()
        while True:
            try:
                t = cpu.temperature
            except Exception:
                continue
            try:
                t = float(t)
                t = round(t, 2)
            except Exception:
                continue
            path_to_save = Path(self._path_to_files.value)
            if not path_to_save.is_dir():
                path_to_save.mkdir(parent=True)
            with open(path_to_save / 'rpi_temp.txt', 'a') as f:
                f.write(f'{datetime.utcnow().strftime("%Y%m%d %H%M%S")}\t{t:.2f}C\n')
            self._t_rpi.value = t
            sleep(30)

    @property
    def temperature_rpi(self):
        return self._t_rpi.value
