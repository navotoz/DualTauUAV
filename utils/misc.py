import threading as th
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_image(image: Union[Image.Image, np.ndarray], title=None, v_min=None, v_max=None, to_close: bool = True,
               show_axis: bool = False):
    if isinstance(image, Image.Image):
        image = np.array([image])
    if np.any(np.iscomplex(image)):
        image = np.abs(image)
    if len(image.shape) > 2:
        if image.shape[0] == 3 or image.shape[0] == 1:
            image = image.transpose((1, 2, 0))  # CH x W x H -> W x H x CH
        elif image.shape[-1] != 3 and image.shape[-1] != 1:  # CH are not RGB or grayscale
            image = image.mean(-1)
    plt.imshow(image.squeeze(), cmap='gray', vmin=v_min, vmax=v_max)
    if title is not None:
        plt.title(title)
    plt.axis('off' if not show_axis else 'on')
    plt.tight_layout()
    plt.show()
    plt.close() if to_close else None


def normalize_image(image: np.ndarray) -> Image.Image:
    if image.dtype == np.bool:
        return Image.fromarray(image.astype('uint8') * 255)
    image = image.astype('float32')
    if (0 == image).all():
        return Image.fromarray(image.astype('uint8'))
    mask = image > 0
    image[mask] -= image[mask].min()
    image[mask] = image[mask] / image[mask].max()
    image[~mask] = 0
    image *= 255
    return Image.fromarray(image.astype('uint8'))


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


def save_single_npz_file(path_to_files: (str, Path)):
    """
    Save a single npz file to a single npz file.
    :param path_to_files: Path to the npz file to save.
    :return: None
    """
    path_to_files = Path(path_to_files)
    data = {}
    for path in path_to_files.glob('*.npz'):
        file = np.load(path)
        for k, v in file.items():
            data.setdefault(k, []).extend(v)
    indices = np.argsort(data['time'])
    data = {k: np.array(v)[indices] for k, v in data.items()}
    np.savez(path_to_files.with_suffix('.npz'), **data)