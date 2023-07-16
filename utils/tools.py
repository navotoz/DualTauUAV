import logging
import threading as th

import numpy as np


class SyncFlag:
    def __init__(self, init_state: bool = True) -> None:
        self._event = th.Event()
        self._event.set() if init_state else self._event.clear()

    def __call__(self) -> bool:
        return self._event.is_set()

    def set(self, new_state: bool):
        self._event.set() if new_state is True else self._event.clear()

    def __bool__(self) -> bool:
        return self._event.is_set()


def make_logger(name) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create a stdout handler
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    # Add the stdout handler to the logger
    logger.addHandler(stdout_handler)
    return logger


def load_files_from_dir(path):
    data = {}
    list_files = list(path.glob('*.npz'))
    for p in list_files:
        try:
            d = np.load(p)
        except:
            continue
        for k, v in d.items():
            data.setdefault(k, []).extend(v)
    if not data:
        try:
            data = np.load(path.with_suffix('.npz'))
        except:
            raise FileNotFoundError(f'No files found in {path}')
    indices = np.argsort(data['time_ns_start'])
    data = {k: np.stack(v)[indices] for k, v in data.items()}
    return data
