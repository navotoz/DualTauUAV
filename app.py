from itertools import product
from pathlib import Path
import shutil
from flask import Flask, Response, redirect, render_template, request
import threading as th

import multiprocessing as mp
from ctypes import c_wchar_p

from thread_devices import ThreadDevices, NAME_DEVICES_THREAD
from time import sleep

from utils.tools import array_to_image
SLEEP_GENERATOR_SEC = 0.5

app = Flask(__name__)

# Create a random save folder, to avoid error in RPi timestamp
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
gen = product(alphabet, repeat=3)
path_to_save: mp.Value = mp.Value(c_wchar_p)
path_to_save.value = str(Path().cwd() / 'measurements' / str(''.join(next(gen))))
while True:
    if not Path(path_to_save.value).is_dir():
        Path(path_to_save.value).mkdir(parents=True)
        break
    else:
        path_to_save.value = str(Path().cwd() / 'measurements' / str(''.join(next(gen))))

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


@app.route("/save", methods=["POST"])
def save():
    global path_to_save

    # Get the user input for the folder name
    folder_name = request.form["folder"]

    # Change the folder in the path_to_save variable
    path_to_save = path_to_save.parent / folder_name
    if path_to_save.is_dir():
        counter = 0
        while path_to_save.is_dir():
            counter += 1
            path_to_save = path_to_save.parent / f'{folder_name}_{counter}'
    path_to_save.mkdir(parents=True, exist_ok=True)

    print('Changed folder to save to:', folder_name)
    return redirect(request.referrer)


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
