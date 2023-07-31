from itertools import product
from pathlib import Path
import shutil
from flask import Flask, Response, render_template
import threading as th

from thread_devices import ThreadDevices, NAME_DEVICES_THREAD
from time import sleep

from utils.tools import array_to_image
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


def generate_image():
    while True:
        # yield the byte string to be sent as the HTTP response
        jpeg = array_to_image(thread_devices.frame, '')
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
