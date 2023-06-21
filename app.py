from functools import partial
import shutil
from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template

from devices.Camera import INIT_CAMERA_PARAMETERS, EnumParameterPosition
from devices.Camera.CameraProcess import CameraCtrl

app = Flask(__name__)

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
params['ffc_period'] = 1800

func_cam = partial(CameraCtrl, camera_parameters=params, is_dummy=False,
                   time_to_save=12e10)  # dump to disk every 2 minutes
camera_pan = func_cam(path_to_save=path_to_save / 'pan', name='pan')
camera_pan.start()
camera_mono = func_cam(path_to_save=path_to_save / 'mono', name='mono')
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
    # Status of cameras
    pan_status = EnumParameterPosition(camera_pan._param_setting_pos.value)
    if pan_status == EnumParameterPosition.DONE:
        pan_status = 'Ready'
    else:
        pan_status = pan_status.name
    mono_status = EnumParameterPosition(camera_mono._param_setting_pos.value)
    if mono_status == EnumParameterPosition.DONE:
        mono_status = 'Ready'
    else:
        mono_status = mono_status.name

    # Rates
    rate_pan = camera_pan.rate_camera
    rate_mono = camera_mono.rate_camera

    # Number of files saved
    pan_files_saved = camera_pan.n_files_saved
    mono_files_saved = camera_mono.n_files_saved

    return render_template(
        'camera_stats.html', pan_status=pan_status, mono_status=mono_status, pan_rate=rate_pan,
        mono_rate=rate_mono, pan_files=pan_files_saved, mono_files=mono_files_saved)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
