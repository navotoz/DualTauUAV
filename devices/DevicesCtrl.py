import multiprocessing as mp
import threading as th
from collections import deque, namedtuple
from ctypes import c_uint
from datetime import datetime
from pathlib import Path
from time import time_ns, sleep
from typing import Union

import numpy as np

from devices.Camera.CameraProcess import CameraCtrl
from devices.gps import GPS
from utils.misc import SyncFlag

TIME_TO_DUMP_NSEC = 12e10  # dump every 2 minutes
namedtuple_gps_results_error = namedtuple(
    field_names=['time', 'latitude', 'longitude', 'altitude'],
    defaults=[np.NaN] * 4, typename='position')


class DeviceCtrlThread(th.Thread):
    _threads_dict = {}
    _flag_run = SyncFlag()
    _deque_rate_cam = deque(maxlen=100)
    _deque_rate_cam.append(time_ns())
    _deque_rate_cam.append(time_ns())

    def __init__(self, path_to_save: Union[str, Path]):
        super().__init__()
        self.daemon = False
        path_to_save = Path(path_to_save)
        self._path_to_save_images = path_to_save / 'images'
        if not self._path_to_save_images.is_dir():
            self._path_to_save_images.mkdir(parents=True)
        self._lock_meas = th.Lock()
        self._semaphore_save = th.Semaphore(value=0)
        self._event_save = th.Event()
        self._event_save.clear()
        self._images = {}

        self._rate_camera = mp.Value(c_uint)
        self._camera = CameraCtrl()
        self._camera.start()
        self._gps = None

    @property
    def event_save(self) -> th.Event:
        return self._event_save

    @property
    def time_to_dump_ns(self) -> float:
        return TIME_TO_DUMP_NSEC

    def run(self) -> None:
        self._threads_dict['gps_conn'] = th.Thread(target=self._th_conn_gps, name='th_conn_gps', daemon=True)
        self._threads_dict['rate'] = th.Thread(target=self._th_rate_camera_function, name='th_dev_rate', daemon=True)
        self._threads_dict['getter'] = th.Thread(target=self._th_image_getter, name='th_dev_getter', daemon=True)
        self._threads_dict['dump'] = th.Thread(target=self._th_dump_data, name='th_dump_data', daemon=False)
        self._threads_dict['timer'] = th.Thread(target=self._th_timer, name='th_timer', daemon=True)
        [p.start() for p in self._threads_dict.values()]

    def _th_conn_gps(self) -> None:
        while self._flag_run:
            try:
                self._gps = GPS()
                break
            except (AttributeError, RuntimeError, KeyError, IndexError, ValueError, ConnectionError):
                sleep(2)

    def terminate(self) -> None:
        try:
            self._flag_run.set(False)
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass
        try:
            self._semaphore_save.release()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass
        try:
            self._camera.terminate()
            self._camera.join()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass
        try:
            self._gps.terminate()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError):
            pass

    def _th_rate_camera_function(self) -> None:
        while not self.is_connected:
            sleep(1)

        while True:
            try:
                self._rate_camera.value = int(1e9 / np.nanmean(np.diff(self._deque_rate_cam)))
                [self._deque_rate_cam.popleft() for _ in range(self._deque_rate_cam.maxlen // 10)]
            except (IndexError, ValueError):
                pass
            sleep(5)

    def _th_timer(self) -> None:
        while not self.is_connected:
            sleep(1)

        timer = time_ns()
        while True:
            if time_ns() - timer >= TIME_TO_DUMP_NSEC:
                self._semaphore_save.release()
                timer = time_ns()
            sleep(5)

    def _th_image_getter(self) -> None:
        while not self.is_connected:
            sleep(1)

        while self._flag_run:
            image = self._camera.image
            time_image = time_ns()  # avoid a race condition on the deque with th_rate_camera_function
            self._deque_rate_cam.append(time_image)
            position = namedtuple_gps_results_error()
            try:
                position = self._gps.position
            except (AttributeError, RuntimeError, KeyError, IndexError, ValueError):
                pass

            # updates the thread-safe _images list in the instance attributes
            with self._lock_meas:
                self._images.setdefault('image', []).append(image)
                self._images.setdefault('fpa', []).append(self._camera.fpa)
                self._images.setdefault('housing', []).append(self._camera.housing)
                self._images.setdefault('position', []).append(position)
                self._images.setdefault('time_ns', []).append(time_image)

    def _th_dump_data(self) -> None:
        while not self.is_connected:
            sleep(1)

        while self._flag_run:  # non-daemon thread, so must use flag_run for loop
            self._semaphore_save.acquire()
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = (self._path_to_save_images / f'{now}').with_suffix('.npz')
            with self._lock_meas:
                if not self._images:
                    continue
                data = self._images
                self._images = {}

            try:
                gps_data = dict(time_gps=np.stack([p.time for p in data['position']]),
                                latitude=np.stack([p.latitude for p in data['position']]),
                                longitude=np.stack([p.longitude for p in data['position']]),
                                altitude=np.stack([p.altitude for p in data['position']]))
            except (AttributeError, RuntimeError, ValueError, IndexError, KeyError):
                gps_data = {}
            np.savez(str(path),
                     images=np.stack(data['image']),
                     cnt_init=np.stack(data['cnt_init']),
                     cnt_final=np.stack(data['cnt_final']),
                     fpa=np.stack(data['fpa']),
                     housing=np.stack(data['housing']),
                     time_ns=np.stack(data['time_ns']),
                     **gps_data)
            self._event_save.set()
            try:
                print(f'Dumped image {"/".join(path.parts[-3:])}')
            except (AttributeError, RuntimeError, ValueError, IndexError, KeyError):
                pass

    @property
    def image(self) -> np.ndarray:
        return self._camera.image

    @property
    def status_gps(self) -> int:
        return isinstance(self._gps, GPS) and self._gps.is_alive

    @property
    def is_connected(self) -> bool:
        return self._camera.is_connected

    @property
    def rate_camera(self) -> int:
        return self._rate_camera.value
