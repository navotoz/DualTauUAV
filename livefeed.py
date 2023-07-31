import io
from time import sleep
from flask import Flask, Response, render_template
import numpy as np
from PIL import Image

from devices.Tau2Grabber import Tau2

from utils.misc import normalize_image
from utils.tools import DummyGPIO

SCALE = 2.5
WIDTH_VIEWER = int(SCALE*336)
HEIGHT_VIEWER = int(SCALE*256)

app = Flask(__name__)


def closer():
    try:
        camera_mono.__del__()
    except (RuntimeError, ValueError, NameError, KeyError, TypeError, AttributeError):
        pass
    try:
        camera_pan.__del__()
    except (RuntimeError, ValueError, NameError, KeyError, TypeError, AttributeError):
        pass


def generate_frames():
    while True:
        # Get frames from cameras
        pan = camera_pan.grab()[0]
        mono = camera_mono.grab()[0]

        if pan is None:
            pan = np.random.rand(256, 256)
        if mono is None:
            mono = np.random.rand(256, 256)

        # Normalize
        pan = normalize_image(pan)
        mono = normalize_image(mono)

        # Combine frames side by side
        combined_frame = np.hstack((pan, mono))

        # Convert ndarray to PIL Image
        pil_image = Image.fromarray(combined_frame)

        # Convert PIL Image to byte stream
        img_io = io.BytesIO()
        pil_image.save(img_io, format='JPEG')
        img_io.seek(0)

        # Yield byte stream as response
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img_io.read() + b'\r\n')

        sleep(1/20)


try:
    import RPi.GPIO as GPIO
    print('Loaded GPIO module for RPi', flush=True)
except (ModuleNotFoundError, RuntimeError):
    print('Could not load GPIO module for RPi', flush=True)
    GPIO = DummyGPIO()

# Configure the trigger pin on the RPi
PIN_TRIGGER = 17
GPIO.setmode(GPIO.BCM)
GPIO.cleanup()
GPIO.setup(PIN_TRIGGER, GPIO.OUT)
GPIO.output(PIN_TRIGGER, GPIO.LOW)  # Set the trigger pin to low -> constant trigger

camera_pan = Tau2()
camera_pan.ffc_mode = 'auto'
camera_mono = Tau2()
camera_mono.ffc_mode = 'auto'


@app.route('/')
def index():
    return render_template('livefeed.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
