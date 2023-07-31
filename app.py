from datetime import datetime
from io import BytesIO
from itertools import product
from pathlib import Path
import shutil
from time import sleep
from flask import Flask, Response, render_template
import threading as th
from PIL import ImageDraw, ImageFont, Image

import numpy as np
from thread_devices import ThreadDevices, NAME_DEVICES_THREAD
SLEEP_GENERATOR_SEC = 0.5


app = Flask(__name__)

# Create a random save folder, to avoid error in RPi timestamp
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
gen = product(alphabet, repeat=3)
path_to_save = Path().cwd() / 'measurements' / str(''.join(next(gen)))
while True:
    if not path_to_save.is_dir():
        path_to_save.mkdir(parents=True)
        break
    else:
        path_to_save = Path().cwd() / 'measurements' / str(''.join(next(gen)))

if NAME_DEVICES_THREAD not in map(lambda x: x.name, th.enumerate()):
    thread_devices = ThreadDevices(path_to_save=path_to_save)
    thread_devices.start()


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


def _array_to_image(arr, text):
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


def generate_image():
    while thread_devices.status_pan != 'Ready':
        jpeg = _array_to_image(np.random.randint(0, 255, size=(256, 256)), '')
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        sleep(SLEEP_GENERATOR_SEC)

    while True:
        # yield the byte string to be sent as the HTTP response
        jpeg = _array_to_image(thread_devices.frame, '')
        if jpeg is None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n')
        else:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

        # get the current time and update the text
        sleep(SLEEP_GENERATOR_SEC)


@app.route('/video')
def video_feed():
    return Response(generate_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/delete_all')
def delete():
    for idx, folder in enumerate(path_to_save.parent.glob("*/")):
        try:
            if folder.is_dir() and folder.parent == path_to_save.parent:
                shutil.rmtree(folder)
        except Exception as e:
            print(e)
    return f'Deleted {idx} folders.'


@app.route('/')
def index():
    return render_template(
        'camera_stats.html',
        path_to_files=str(path_to_save.name),
        pan_status=thread_devices.status_pan,
        pan_rate=thread_devices.rate_pan,
        pan_files=thread_devices.n_files_pan,
        mono_status=thread_devices.status_mono,
        mono_rate=thread_devices.rate_mono,
        mono_files=thread_devices.n_files_mono)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)

# TODO: add reset procedure
