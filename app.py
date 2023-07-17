from pathlib import Path
import shutil
from uuid import uuid4
from flask import Flask, render_template
from functools import partial
from time import sleep
import multiprocessing as mp

from devices import INIT_CAMERA_PARAMETERS, EnumParameterPosition
from devices.CameraProcess import CameraCtrl

app = Flask(__name__)


def _make_status(self, stat: int):
    stat = EnumParameterPosition(stat)
    if stat == EnumParameterPosition.DONE:
        return 'Ready'
    else:
        return stat.name


# Create a random save folder, to avoid error in RPi timestamp
path_to_save = Path().cwd() / 'measurements' / str(uuid4().hex)
while True:
    if not path_to_save.is_dir():
        path_to_save.mkdir(parents=True)
        break
    else:
        path_to_save = Path().cwd() / 'measurements' / str(uuid4().hex)

# Init two cameras - Panchromatic and Monochromatic
params = INIT_CAMERA_PARAMETERS
params['ffc_mode'] = 'auto'
params['ffc_period'] = 3600  # ffc every one minute

# Init cameras
# Cams are synced by the barrier, which releases all cams simultaneously when N_CAMERAS acquire it.
N_CAMERAS = 2
barrier_camera_sync = mp.Barrier(parties=N_CAMERAS)
func_cam = partial(CameraCtrl, camera_parameters=params, is_dummy=False,
                   barrier_camera_sync=barrier_camera_sync,
                   time_to_save=5e9)  # dump to disk every 5 seconds
camera_pan = func_cam(path_to_save=path_to_save / 'pan', name='pan')
camera_pan.start()
camera_mono = func_cam(path_to_save=path_to_save / 'mono', name='mono')
while camera_pan.camera_parameters_setting_position != EnumParameterPosition.DONE.value:
    sleep(1)
camera_mono.start()


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
        pan_status=_make_status(camera_pan._param_setting_pos.value),
        pan_rate=camera_pan.rate_camera,
        pan_files=camera_pan.n_files_saved,
        mono_status=_make_status(camera_mono._param_setting_pos.value),
        mono_rate=camera_mono.rate_camera,
        mono_files=camera_mono.n_files_saved)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
