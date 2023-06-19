from collections import deque
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import threading as th
from ctypes import c_ushort, c_uint
from itertools import cycle
from time import sleep, time_ns
from typing import Union

import numpy as np

from usb.core import USBError

from devices import DeviceAbstract
from devices.Camera import CameraAbstract, INIT_CAMERA_PARAMETERS, T_HOUSING, T_FPA, EnumParameterPosition
from devices.Camera.Tau.Tau2Grabber import Tau2Grabber

from utils.tools import make_logger
TEMPERATURE_ACQUIRE_FREQUENCY_SECONDS = 5


class CameraCtrl(DeviceAbstract):
    _camera: CameraAbstract = None

    def __init__(self, path_to_save: Union[str, Path], name: str = '', time_to_save: int = 12e10,
                 camera_parameters: dict = INIT_CAMERA_PARAMETERS, is_dummy: bool = False):
        super().__init__()
        self._path_to_save = Path(path_to_save)
        if not self._path_to_save.is_dir():
            self._path_to_save.mkdir(parents=True, exist_ok=True)
        self._camera_params = camera_parameters
        self._event_connected = mp.Event()
        self._event_connected.clear() if not is_dummy else self._event_connected.set()
        self._lock_measurements = th.RLock()
        self._frames = {}
        self._fpa, self._housing = 0, 0
        self._time_to_save = time_to_save

        # process-safe param setting position
        self._param_setting_pos: mp.Value = mp.Value(typecode_or_type=c_ushort)  # uint16
        self._param_setting_pos.value = EnumParameterPosition.DISCONNECTED

        # Process-safe rate of camera
        self._rate_camera: mp.Value = mp.Value(c_uint)
        self._event_get_camera_rate = th.Event()
        self._event_get_camera_rate.clear()
        self._event_set_camera_rate = th.Event()
        self._event_set_camera_rate.clear()

        # Thread-safe saving
        self._semaphore_save = th.Semaphore(value=0)
        self._n_files_saved: mp.Value = mp.Value(c_uint)

        self._name = name
        self._logger = make_logger(name=f"{name}Process")
        self._logger.info('Created instance.')

    def _terminate_device_specifics(self) -> None:
        try:
            self._flag_run.set(False)
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass
        try:
            self._event_connected.set()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass
        sleep(0.5)
        try:
            self._event_connected.clear()  # so that the is_connected property will return False
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass
        try:
            self._semaphore_save.release()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass

    def _run(self) -> None:
        self._workers_dict['rate'] = th.Thread(target=self._th_rate_camera_function, name='th_dev_rate', daemon=True)
        self._workers_dict['get_t'] = th.Thread(target=self._th_getter_temperature, name='th_cam_get_t', daemon=True)
        self._workers_dict['getter'] = th.Thread(target=self._th_getter_frame, name='th_cam_getter', daemon=False)
        self._workers_dict['dump'] = th.Thread(target=self._th_dump_data, name='th_dump_data', daemon=False)
        self._workers_dict['timer'] = th.Thread(target=self._th_timer, name='th_timer', daemon=True)
        self._workers_dict['conn'] = th.Thread(target=self._th_connect, name='th_cam_conn', daemon=False)

    @property
    def camera_parameters_setting_position(self) -> int:
        return self._param_setting_pos.value

    def _update_params(self) -> None:
        while self._flag_run and self._camera._param_position != EnumParameterPosition.DONE:
            self._param_setting_pos.value = self._camera._param_position.value
            sleep(1)
        self._param_setting_pos.value = self._camera._param_position.value

    def _th_connect(self) -> None:
        self._logger.info('Connecting...')
        while self._flag_run:
            try:
                self._camera = Tau2Grabber(name=self._name)
                th.Thread(target=self._update_params, daemon=True).start()
                self._camera.set_params_by_dict(self._camera_params) if self._camera_params else None
                self._camera._param_position = EnumParameterPosition.DONE
                self._logger.info('Finished setting parameters.')
                self._getter_temperature(T_FPA)
                self._getter_temperature(T_HOUSING)
                self._logger.info(f'Initial temperatures {self._fpa / 100:.2f}C.')
                self._event_connected.set()
                self._logger.info('Camera connected.')
                return
            except (RuntimeError, BrokenPipeError, USBError):
                pass
            sleep(1)

    def _getter_temperature(self, t_type: str):  # this function exists for the th_connect function, otherwise redundant
            t = self._camera.get_inner_temperature(t_type) if self._camera is not None else None
            if t is not None and t != 0.0 and t != -float('inf'):
                try:
                    t = round(t * 100)
                    with self._lock_measurements:
                        if t_type == T_FPA:
                            self._fpa = round(t, -1)  # precision for the fpa is 0.1C
                        elif t_type == T_HOUSING:
                            self._housing = t  # precision of the housing is 0.01C
                except (BrokenPipeError, RuntimeError):
                    pass

    def _th_getter_temperature(self) -> None:
        self._event_connected.wait()
        for t_type in cycle([T_FPA, T_HOUSING]):
            self._getter_temperature(t_type=t_type)
            sleep(TEMPERATURE_ACQUIRE_FREQUENCY_SECONDS)

    def _th_getter_frame(self) -> None:
        frame = None
        self._event_connected.wait()
        while self._flag_run:
            with self._lock_measurements:
                try:
                    frame = self._camera.grab() if self._camera is not None else None
                    time_frame = time_ns()
                except Exception as e:
                    self._logger.error(f'Exception in _th_getter_frame: {e}')
                    self.terminate()
                if frame is not None:
                    self._frames.setdefault('time_ns', []).append(time_frame)
                    self._frames.setdefault('frame', []).append(frame)
                    self._frames.setdefault('fpa', []).append(self._fpa)
                    self._frames.setdefault('housing', []).append(self._housing)

    @property
    def is_connected(self) -> bool:
        return self._event_connected.is_set()

    def _th_rate_camera_function(self) -> None:
        self._event_connected.wait()
        while True:
            self._event_set_camera_rate.wait()
            with self._lock_measurements:
                times = self._frames.get('time_ns', [0, 0])
            try:
                self._rate_camera.value = int(len(times) * 1e9 / (times[-1] - times[0]))
            except (IndexError, ZeroDivisionError):
                self._rate_camera.value = 0
            self._event_get_camera_rate.set()
            self._event_set_camera_rate.clear()

    @property
    def rate_camera(self) -> int:
        self._event_get_camera_rate.clear()
        self._event_set_camera_rate.set()
        self._event_get_camera_rate.wait()
        return self._rate_camera.value

    def _th_timer(self) -> None:
        self._event_connected.wait()
        timer = time_ns()
        while True:
            if time_ns() - timer >= self._time_to_save:
                self._semaphore_save.release()
                timer = time_ns()
            sleep(2)

    def _th_dump_data(self) -> None:
        self._event_connected.wait()
        while self._flag_run:  # non-daemon thread, so must use flag_run for loop
            self._semaphore_save.acquire()
            now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = (self._path_to_save / f'{now}').with_suffix('.npz')
            with self._lock_measurements:
                if not self._frames:
                    continue
                data = self._frames
                self._frames = {}

            np.savez(str(path),
                     frames=np.stack(data['frame']),
                     fpa=np.stack(data['fpa']),
                     housing=np.stack(data['housing']),
                     time_ns=np.stack(data['time_ns']))
            self._logger.info(f'Dumped image {str(path)}')
            self._n_files_saved.value = self._n_files_saved.value + 1

    @property
    def n_files_saved(self) -> int:
        return self._n_files_saved.value
