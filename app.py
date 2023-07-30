from datetime import datetime
from itertools import product
from pathlib import Path
import shutil
from time import sleep
import cv2
from flask import Flask, Response, render_template
import threading as th

import numpy as np
from thread_devices import ThreadDevices, NAME_DEVICES_THREAD

app = Flask(__name__)

# Create a random save folder, to avoid error in RPi timestamp
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
gen = product(alphabet, repeat=3)
print(len(list(gen)))
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


def put_text_with_background(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=2):
    cv2.putText(img, text, org, font, font_scale, (255, 255, 255), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)


def _array_to_image(arr, text):
    arr = arr.copy() - np.min(arr)
    arr = 255 * (arr / np.max(arr))
    img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # write the text on the image
    if text is not None and text:
        put_text_with_background(img, text, (1, 25))

    # write the time on the lower-right corner of the image
    text = datetime.now().strftime('%H:%M:%S')
    put_text_with_background(img, text, (img.shape[1] - 75, img.shape[0] - 5))

    # encode the image as a JPEG byte string
    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg


def generate_image():
    while thread_devices.status_pan != 'Ready':
        text = 'Camera is disconnected'
        jpeg = _array_to_image(np.random.randint(0, 255, size=(256, 256)), text)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        sleep(1)

    while True:
        # yield the byte string to be sent as the HTTP response
        jpeg = _array_to_image(thread_devices.frame, '')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        # get the current time and update the text
        sleep(1)


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
