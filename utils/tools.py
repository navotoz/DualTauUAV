import logging
import threading as th

import numpy as np
from PIL import ImageDraw, ImageFont, Image
from io import BytesIO
from datetime import datetime


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
        except Exception:
            continue
        for k, v in d.items():
            data.setdefault(k, []).extend(v)
    if not data:
        try:
            data = np.load(path.with_suffix('.npz'))
        except Exception:
            raise FileNotFoundError(f'No files found in {path}')
    indices = np.argsort(data['counter'])
    data = {k: np.stack(v)[indices] for k, v in data.items()}
    return data


def get_text_dimensions(text_string, font):
    # https://stackoverflow.com/a/46220683/9263761
    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3]

    return (text_width, text_height)


def put_text_with_background(img, text, org, font=None, font_scale=0.5, thickness=2):
    draw = ImageDraw.Draw(img)
    font = font or ImageFont.load_default()
    text_width, text_height = get_text_dimensions(text, font)

    # Background rectangle
    rect_left = org[0] - 1
    rect_top = org[1] - 1
    rect_width = text_width + 3
    rect_height = text_height + 3
    rect_right = rect_left + rect_width
    rect_bottom = rect_top + rect_height
    draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill=(255, 255, 255))

    # Text
    draw.text(org, text, font=font, fill=(0, 0, 255))
    return img


def array_to_image(arr, text):
    if not isinstance(arr, np.ndarray):
        return None
    arr = arr - np.min(arr)
    max_val = np.max(arr)
    arr = 255 * (arr / (max_val if max_val > 0 else 1))
    img = Image.fromarray(arr.astype(np.uint8), mode='L')
    img = img.convert('RGB')

    # write the text on the image
    if text is not None and text:
        font = ImageFont.load_default()
        put_text_with_background(img, text, (1, 25), font=font)

    # write the time on the lower-right corner of the image
    text = datetime.now().strftime('%H:%M:%S')
    font = ImageFont.load_default()
    put_text_with_background(img, text, (img.width-60, img.height-20), font=font)

    # encode the image as a JPEG byte string
    with BytesIO() as output:
        img.save(output, format='JPEG')
        return output.getvalue()


class DummyGPIO:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return self

    def __setattr__(self, name, value):
        pass
