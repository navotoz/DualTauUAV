import multiprocessing as mp
from dataclasses import dataclass
from threading import Thread

from PIL import Image

from devices.gps import GpsData
from utils.misc import SyncFlag, show_image
import numpy as np



class DeviceAbstract(mp.Process):
    _workers_dict = {}

    def __init__(self):
        super().__init__()
        self.daemon = False
        self._flag_run = SyncFlag(init_state=True)
        self._event_terminate = mp.Event()
        self._event_terminate.clear()
        self._workers_dict['terminate'] = Thread(daemon=False, name='term', target=self._terminate)

    def run(self):
        self._run()
        [p.start() for p in self._workers_dict.values()]

    def _run(self):
        raise NotImplementedError

    def _wait_for_threads_to_exit(self):
        for key, t in self._workers_dict.items():
            try:
                if t.daemon:
                    continue
                t.join()
            except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
                pass

    def terminate(self):
        try:
            self._event_terminate.set()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
            pass

    def _terminate(self) -> None:
        self._event_terminate.wait()
        try:
            self._flag_run.set(False)
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
            pass
        self._terminate_device_specifics()
        self._wait_for_threads_to_exit()
        try:
            self.kill()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
            pass

    def _terminate_device_specifics(self) -> None:
        raise NotImplementedError
