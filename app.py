import shutil
from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template

from devices.Camera import INIT_CAMERA_PARAMETERS, EnumParameterPosition
from devices.Camera.CameraProcess import CameraCtrl
N_SECONDS_TO_SAVE = 30

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

camera_pan = CameraCtrl(camera_parameters=params, path_to_save=path_to_save / 'pan')
camera_pan.start()
camera_mono = CameraCtrl(camera_parameters=params, path_to_save=path_to_save / 'mono')
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
    status_pan = EnumParameterPosition(camera_pan._param_setting_pos.value)
    if status_pan == EnumParameterPosition.CONNECTED:
        status_pan = 'Ready'
    status_mono = EnumParameterPosition(camera_mono._param_setting_pos.value)
    if status_mono == EnumParameterPosition.CONNECTED:
        status_mono = 'Ready'

    # Rates
    rate_pan = camera_pan.rate_camera
    rate_mono = camera_mono.rate_camera

    # Number of files saved
    pan_files_saved = camera_pan.n_files_saved
    mono_files_saved = camera_mono.n_files_saved

    return render_template(
        'camera_stats.html', status_pan=status_pan, status_mono=status_mono, pan_rate=rate_pan, mono_rate=rate_mono,
        pan_files=pan_files_saved, mono_files=mono_files_saved)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
