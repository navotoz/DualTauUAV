import multiprocessing as mp
from pathlib import Path
import threading as th
from ctypes import c_ushort, c_byte
from itertools import cycle
from time import sleep
from typing import Union

from numpy import copyto, frombuffer, uint16
from usb.core import USBError

from devices import DeviceAbstract
from devices.Camera import (CameraAbstract, INIT_CAMERA_PARAMETERS, HEIGHT_IMAGE_TAU2, WIDTH_IMAGE_TAU2, T_HOUSING,
                            T_FPA, EnumParameterPosition)
from devices.Camera.Tau.Tau2Grabber import Tau2Grabber
import logging
logger = logging.getLogger(__name__)
TEMPERATURE_ACQUIRE_FREQUENCY_SECONDS = 10


class CameraCtrl(DeviceAbstract):
    _camera: (CameraAbstract, None) = None

    def __init__(self, path_to_save: Union[str, Path],
                 camera_parameters: dict = INIT_CAMERA_PARAMETERS, is_dummy: bool = False):
        super().__init__()
        self._event_connected = mp.Event()
        self._event_connected.clear() if not is_dummy else self._event_connected.set()
        self._lock_camera = th.RLock()
        self._lock_image = mp.Lock()
        self._event_new_image = mp.Event()
        self._event_new_image.clear()
        self._semaphore_ffc_do = mp.Semaphore(value=0)
        self._semaphore_ffc_finished = mp.Semaphore(value=0)

        # process-safe ffc results
        self._ffc_result: mp.Value = mp.Value(typecode_or_type=c_byte)
        self._ffc_result.value = 0

        # process-safe image
        self._image_array = mp.RawArray(c_ushort, HEIGHT_IMAGE_TAU2 * WIDTH_IMAGE_TAU2)
        self._image_array = frombuffer(self._image_array, dtype=uint16)
        self._image_array = self._image_array.reshape(HEIGHT_IMAGE_TAU2, WIDTH_IMAGE_TAU2)

        # process-safe temperature
        self._fpa: mp.Value = mp.Value(typecode_or_type=c_ushort)  # uint16
        self._housing: mp.Value = mp.Value(typecode_or_type=c_ushort)  # uint16

        # process-safe param setting position
        self._param_setting_pos: mp.Value = mp.Value(typecode_or_type=c_ushort)  # uint16
        self._param_setting_pos.value = EnumParameterPosition.DISCONNECTED

        self._camera_params = camera_parameters

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
            self._semaphore_ffc_do.release()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass
        try:
            self._semaphore_ffc_finished.release()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass
        try:
            self._event_new_image.set()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass

    def _run(self) -> None:
        self._workers_dict['conn'] = th.Thread(target=self._th_connect, name='th_cam_conn', daemon=False)
        self._workers_dict['get_t'] = th.Thread(target=self._th_getter_temperature, name='th_cam_get_t', daemon=True)
        self._workers_dict['getter'] = th.Thread(target=self._th_getter_image, name='th_cam_getter', daemon=False)
        self._workers_dict['ffc'] = th.Thread(target=self._th_ffc_func, name='th_cam_ffc', daemon=True)

    @property
    def camera_parameters_setting_position(self) -> int:
        return self._param_setting_pos.value

    def _update_params(self) -> None:
        while self._flag_run and self._camera._param_position != EnumParameterPosition.DONE:
            self._param_setting_pos.value = self._camera._param_position.value
            sleep(2)
        self._param_setting_pos.value = self._camera._param_position.value

    def _th_connect(self) -> None:
        while self._flag_run:
            with self._lock_camera:
                try:
                    self._camera = Tau2Grabber()
                    th.Thread(target=self._update_params, daemon=True).start()
                    self._camera.set_params_by_dict(self._camera_params) if self._camera_params else None
                    logger.info('Finished setting parameters.')
                    self._getter_temperature(T_FPA)
                    self._getter_temperature(T_HOUSING)
                    logger.info(f'Initial temperatures {self._fpa.value / 100:.2f}C.')
                    self._event_connected.set()
                    logger.info('Camera connected.')
                    return
                except (RuntimeError, BrokenPipeError, USBError):
                    pass
            sleep(1)

    def _th_ffc_func(self) -> None:
        self._event_connected.wait()
        while self._flag_run:
            self._semaphore_ffc_do.acquire()
            self._ffc_result.value = self._camera.ffc()
            self._semaphore_ffc_finished.release()

    def _getter_temperature(self, t_type: str):  # this function exists for the th_connect function, otherwise redundant
        with self._lock_camera:
            t = self._camera.get_inner_temperature(t_type) if self._camera is not None else None
        if t is not None and t != 0.0 and t != -float('inf'):
            try:
                t = round(t * 100)
                if t_type == T_FPA:
                    self._fpa.value = round(t, -1)  # precision for the fpa is 0.1C
                elif t_type == T_HOUSING:
                    self._housing.value = t  # precision of the housing is 0.01C
            except (BrokenPipeError, RuntimeError):
                pass

    def _th_getter_temperature(self) -> None:
        self._event_connected.wait()
        for t_type in cycle([T_FPA, T_HOUSING]):
            self._getter_temperature(t_type=t_type)
            sleep(TEMPERATURE_ACQUIRE_FREQUENCY_SECONDS)

    def _th_getter_image(self) -> None:
        self._event_connected.wait()
        while self._flag_run:
            with self._lock_camera:
                try:
                    image = self._camera.grab() if self._camera is not None else None
                except Exception as e:
                    logger.error(f'Exception in _th_getter_image: {e}')
                    self.terminate()
            if image is not None:
                with self._lock_image:
                    copyto(self._image_array, image)
                    self._event_new_image.set()

    @property
    def image(self):
        self._event_new_image.wait()
        with self._lock_image:
            self._event_new_image.clear()
            return self._image_array.copy()

    def ffc(self) -> bool:
        self._semaphore_ffc_do.release()
        self._semaphore_ffc_finished.acquire()
        return bool(self._ffc_result.value)

    @property
    def fpa(self) -> float:
        return self._fpa.value

    @property
    def housing(self) -> float:
        return self._housing.value

    @property
    def is_connected(self) -> bool:
        return self._event_connected.is_set()
