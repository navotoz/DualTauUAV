from pathlib import Path
import shutil
from time import sleep
from uuid import uuid4
from flask import Flask, render_template
import threading as th
import RPi.GPIO as GPIO
from thread_devices import ThreadDevices, NAME_DEVICES_THREAD

app = Flask(__name__)

# Create a random save folder, to avoid error in RPi timestamp
path_to_save = Path().cwd() / 'measurements' / str(uuid4().hex)
while True:
    if not path_to_save.is_dir():
        path_to_save.mkdir(parents=True)
        break
    else:
        path_to_save = Path().cwd() / 'measurements' / str(uuid4().hex)

if NAME_DEVICES_THREAD not in map(lambda x: x.name, th.enumerate()):
    thread_devices = ThreadDevices(path_to_save=path_to_save)
    thread_devices.start()


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
        pan_status=thread_devices.status_pan,
        pan_rate=thread_devices.rate_pan,
        pan_files=thread_devices.n_files_pan,
        mono_status=thread_devices.status_mono,
        mono_rate=thread_devices.rate_mono,
        mono_files=thread_devices.n_files_mono)


# The sampling rate of the cameras
FREQUENCY_CAMERAS = 10  # Hz
DUTY_CYCLE = 0.01
period = 1 / FREQUENCY_CAMERAS
low_time = period * DUTY_CYCLE  # seconds
high_time = period - low_time  # seconds

# Define a function to toggle the trigger signal at the given rate
def th_rpi_trigger_for_cam():
    global FREQUENCY_CAMERAS
    while True:
        # Set the trigger pin to high
        GPIO.output(PIN_TRIGGER, GPIO.HIGH)
        # Wait for half the period
        sleep(low_time)
        # Set the trigger pin to low
        GPIO.output(PIN_TRIGGER, GPIO.LOW)
        # Wait for the other half of the period
        sleep(high_time)


# Initialize the GPIO pin for the trigger signal
PIN_TRIGGER = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_TRIGGER, GPIO.OUT)
trigger_thread = th.Thread(target=th_rpi_trigger_for_cam, daemon=True)
trigger_thread.start()


if __name__ == '__main__':
    try:
        app.run(debug=False, host='0.0.0.0', port=8080)
    except Exception:
        GPIO.cleanup()