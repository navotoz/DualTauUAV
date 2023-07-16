from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import threading as th
from ctypes import c_ushort, c_uint
from time import sleep, time_ns
from typing import Union

import numpy as np

from usb.core import USBError

from devices import INIT_CAMERA_PARAMETERS, T_HOUSING, T_FPA, EnumParameterPosition
from devices.Tau2Grabber import Tau2

from utils.tools import make_logger
TEMPERATURE_ACQUIRE_FREQUENCY_SECONDS = 30


class CameraCtrl(mp.Process):
    _workers_dict = {}
    _camera: Tau2 = None

    def __init__(self, path_to_save: Union[str, Path],
                 barrier_camera_sync: mp.Barrier,
                 name: str = '', time_to_save: int = 10e9,
                 camera_parameters: dict = INIT_CAMERA_PARAMETERS, is_dummy: bool = False):
        super().__init__()
        self.daemon = False
        self._barrier_camera_sync: mp.Barrier = barrier_camera_sync

        self._path_to_save = Path(path_to_save)
        if not self._path_to_save.is_dir():
            self._path_to_save.mkdir(parents=True, exist_ok=True)
        self._lock_camera = th.Lock()
        self._camera_params = camera_parameters
        self._event_connected = th.Event()
        self._event_connected.clear() if not is_dummy else self._event_connected.set()
        self._lock_measurements = th.Lock()
        self._frames = {}
        self._fpa, self._housing = 0, 0
        self._time_to_save = time_to_save

        # process-safe param setting position
        self._param_setting_pos: mp.Value = mp.Value(typecode_or_type=c_ushort)  # uint16
        self._param_setting_pos.value = EnumParameterPosition.DISCONNECTED.value

        # Process-safe rate of camera
        self._rate_camera: mp.Value = mp.Value(c_uint)
        self._semaphore_get_rate = mp.Semaphore(value=0)
        self._semaphore_set_rate = mp.Semaphore(value=0)
        self._n_frames = 0
        self._time_start = time_ns()

        # Thread-safe saving
        self._semaphore_save = th.Semaphore(value=0)
        self._n_files_saved: mp.Value = mp.Value(c_uint)

        self._name = name
        self._logger = make_logger(name=f"{name}Process")
        self._logger.info('Created instance.')

    def terminate(self) -> None:
        self.kill()

    def start(self) -> None:
        self._workers_dict['rate'] = th.Thread(target=self._th_rate_camera_function,
                                               name=f'th_dev_rate_{self._name}', daemon=True)
        self._workers_dict['get_t'] = th.Thread(target=self._th_getter_temperature,
                                                name=f'th_cam_get_t_{self._name}', daemon=True)
        self._workers_dict['getter'] = th.Thread(target=self._th_getter_frame,
                                                 name=f'th_cam_getter_{self._name}', daemon=False)
        self._workers_dict['dump'] = th.Thread(
            target=self._th_dump_data, name=f'th_dump_data_{self._name}', daemon=False)
        self._workers_dict['timer'] = th.Thread(target=self._th_timer, name=f'th_timer_{self._name}', daemon=True)
        self._workers_dict['conn'] = th.Thread(target=self._th_connect, name=f'th_cam_conn_{self._name}', daemon=False)
        [p.start() for p in self._workers_dict.values()]

    @property
    def camera_parameters_setting_position(self) -> int:
        return self._param_setting_pos.value

    def _update_params(self) -> None:
        while self._camera.param_position != EnumParameterPosition.DONE:
            self._param_setting_pos.value = self._camera.param_position.value
            sleep(1)
        self._param_setting_pos.value = self._camera.param_position.value

    def _th_connect(self) -> None:
        self._logger.info('Connecting...')
        while True:
            try:
                self._camera = Tau2(name=self._name)
                th.Thread(target=self._update_params, daemon=True).start()
                self._camera.set_params_by_dict(self._camera_params) if self._camera_params else None
                self._camera.param_position = EnumParameterPosition.DONE
                self._logger.info('Finished setting parameters.')
                self._getter_temperature(T_FPA)
                # self._getter_temperature(T_HOUSING)
                self._logger.info(f'Initial temperatures {self._fpa / 100:.2f}C.')
                self._event_connected.set()
                self._logger.info('Camera connected.')
                return
            except (RuntimeError, BrokenPipeError, USBError):
                pass
            sleep(1)

    def _getter_temperature(self, t_type: str):  # this function exists for the th_connect function, otherwise redundant
        with self._lock_camera:
            t = self._camera.get_inner_temperature(t_type) if isinstance(self._camera, Tau2) else None
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
        while True:
            self._getter_temperature(t_type=T_FPA)
            sleep(TEMPERATURE_ACQUIRE_FREQUENCY_SECONDS)
        # for t_type in cycle([T_FPA, T_HOUSING]):
        #     self._getter_temperature(t_type=t_type)
        #     sleep(TEMPERATURE_ACQUIRE_FREQUENCY_SECONDS)

    def _th_getter_frame(self) -> None:
        frame = None
        self._event_connected.wait()
        self._barrier_camera_sync.wait(timeout=None)  # sync the initialization of both cameras
        self._time_start = time_ns()
        while True:
            try:
                with self._lock_camera:
                    try:
                        # timeout of barrier set to 5Hz.
                        # Allows even one camera to keep working.
                        # This low value (5Hz) prevents timeout on saving files.
                        self._barrier_camera_sync.wait(timeout=0.2)
                    except RuntimeError:
                        pass
                    frame = self._camera.grab() if self._camera is not None else None
                    time_frame = time_ns()
            except Exception as e:
                self._logger.error(f'Exception in _th_getter_frame: {e}')
                self.terminate()
            if frame is not None:
                with self._lock_measurements:
                    self._frames.setdefault('time_ns', []).append(time_frame)
                    self._frames.setdefault('frame', []).append(frame)
                    self._frames.setdefault('fpa', []).append(self._fpa)
                    # self._frames.setdefault('housing', []).append(self._housing)
                    self._n_frames += 1
            else:
                self._logger.warning('Skipped frame')

    def _th_rate_camera_function(self) -> None:
        while True:  # no wait for _event_connected to avoid being blocked by the _th_connect
            self._semaphore_set_rate.acquire()
            with self._lock_measurements:
                n_frames = self._n_frames
            self._rate_camera.value = int(n_frames * 1e9 / (time_ns() - self._time_start))
            self._semaphore_get_rate.release()

    @property
    def rate_camera(self) -> int:
        self._semaphore_set_rate.release()
        self._semaphore_get_rate.acquire()
        return self._rate_camera.value

    def _th_timer(self) -> None:
        self._event_connected.wait()
        timer = time_ns()
        while True:
            if time_ns() - timer >= self._time_to_save:
                self._semaphore_save.release()
                timer = time_ns()
            sleep(1)

    def _th_dump_data(self) -> None:
        self._event_connected.wait()
        while True:  # non-daemon thread, so must use flag_run for loop
            self._semaphore_save.acquire()
            now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = (self._path_to_save / f'{now}').with_suffix('.npz')
            with self._lock_measurements:
                if not isinstance(self._frames, dict) or not self._frames.get('time_ns', []):
                    continue
                data = self._frames
                self._frames = {}

            np.savez(str(path),
                     frames=np.stack(data['frame']),
                     fpa=np.stack(data['fpa']),
                     #  housing=np.stack(data['housing']),
                     time_ns=np.stack(data['time_ns']))
            self._logger.info(f'Dumped image {str(path)}')
            self._n_files_saved.value = self._n_files_saved.value + 1

            # Zeroize the rate
            self._n_frames = 0
            self._time_start = time_ns()

    @property
    def n_files_saved(self) -> int:
        return self._n_files_saved.value
