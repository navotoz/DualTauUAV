import shutil
from time import sleep, time_ns
from ctypes import c_ushort
from pathlib import Path
import threading as th
from uuid import uuid4
import numpy as np
from flask import Flask, Response
import cv2
from numpy import copyto, frombuffer, uint16
import multiprocessing as mp

from datetime import datetime

from devices.Camera import INIT_CAMERA_PARAMETERS, HEIGHT_IMAGE_TAU2, WIDTH_IMAGE_TAU2, EnumParameterPosition
from devices.Camera.CameraProcess import CameraCtrl
N_SECONDS_TO_SAVE = 30

app = Flask(__name__)


class AcquireProcess(mp.Process):
    def __init__(self, *, path_to_save, camera) -> None:
        super().__init__(daemon=True)
        self._path_to_save = path_to_save
        self._camera = camera
        self._th_saver = th.Thread(target=self._saver, daemon=True)
        self._th_saver.start()
        self._th_acq = th.Thread(target=self._acquire, daemon=True)
        self._th_acq.start()
        self._lock = th.Lock()
        self._frames, self._fpas, self._time_list_ns = [], [], []

        # process-safe image
        self._image_array = mp.RawArray(c_ushort, HEIGHT_IMAGE_TAU2 * WIDTH_IMAGE_TAU2)
        self._image_array = frombuffer(self._image_array, dtype=uint16)
        self._image_array = self._image_array.reshape(HEIGHT_IMAGE_TAU2, WIDTH_IMAGE_TAU2)

        # process-safe counters
        self._cnt_frames = 0
        self._cnt_files = 0
        self._time_start = int(1e9)
        self._time_curr = int(2e9)
        self._param_setting_pos: mp.Value = mp.Value(typecode_or_type=c_ushort)  # uint16
        self._param_setting_pos.value = EnumParameterPosition.DISCONNECTED

    @property
    def text_info(self):
        n_frames = self._cnt_frames
        rate = (n_frames / ((self._time_curr - self._time_start) * 1e-9))
        return f'{self._cnt_files:,} Files | {n_frames:,} Frames | {rate:.0f}Hz'

    @property
    def frame(self):
        self._lock.acquire()
        if self._frames:
            copyto(self._image_array, self._frames[-1])
            self._lock.release()
        else:
            self._lock.release()
            copyto(self._image_array,
                   np.random.randint(0, 255, size=(HEIGHT_IMAGE_TAU2, WIDTH_IMAGE_TAU2), dtype='uint16'))
        return self._image_array

    @property
    def camera_parameters_setting_position(self) -> int:
        return self._camera._param_setting_pos.value

    @property
    def is_camera_ready(self):
        return self._camera.is_connected

    def _acquire(self):
        while not self.is_camera_ready:
            sleep(1)
        self._time_start = time_ns() - int(1e9)  # prevent division by zero

        while True:
            image = self._camera.image
            fpa = self._camera.fpa
            self._time_curr = time_ns()
            with self._lock:
                self._frames.append(image)
                self._fpas.append(fpa)
                self._time_list_ns.append(self._time_curr)
            self._cnt_frames += 1

    def _saver(self):
        while not self.is_camera_ready:
            sleep(1)

        while True:
            sleep(N_SECONDS_TO_SAVE)
            with self._lock:
                frames_arr = np.array(self._frames)
                fpas_arr = np.array(self._fpas)
                time_list_ns_arr = np.array(self._time_list_ns)
                self._frames.clear()
                self._fpas.clear()
                self._time_list_ns.clear()
            self._cnt_files += 1
            np.savez(file=self._path_to_save / f'{self._cnt_files}.npz',
                     frames=frames_arr, fpa=fpas_arr, time=time_list_ns_arr)


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


path_to_save = Path().cwd() / 'measurements' / str(uuid4().hex)
while True:
    if not path_to_save.is_dir():
        path_to_save.mkdir(parents=True)
        break
    else:
        path_to_save = Path().cwd() / 'measurements' / str(uuid4().hex)

params = INIT_CAMERA_PARAMETERS
params['ffc_mode'] = 'auto'
params['ffc_period'] = 1800
camera = CameraCtrl(camera_parameters=params)
camera.start()

mp_acquire = AcquireProcess(path_to_save=path_to_save, camera=camera)
mp_acquire.start()


def generate_image():
    while not mp_acquire.is_camera_ready:
        param_pos = mp_acquire.camera_parameters_setting_position
        if EnumParameterPosition(param_pos) == EnumParameterPosition.DISCONNECTED:
            text = 'Camera is disconnected'
        else:
            text = f'Waiting, Set {param_pos} of {len(EnumParameterPosition) - 1} params'
        jpeg = _array_to_image(np.random.randint(0, 255, size=(256, 256)), text)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        sleep(1)

    while True:
        # yield the byte string to be sent as the HTTP response
        jpeg = _array_to_image(mp_acquire.frame, mp_acquire.text_info)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        # get the current time and update the text
        sleep(1)


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
    return Response(generate_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=False)
