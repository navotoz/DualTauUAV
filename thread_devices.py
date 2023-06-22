from functools import partial
import threading as th
from time import sleep

from devices import INIT_CAMERA_PARAMETERS, EnumParameterPosition
from devices.CameraProcess import CameraCtrl
NAME_DEVICES_THREAD = 'th_devices'


class ThreadDevices(th.Thread):
    def __init__(self, path_to_save) -> None:
        super().__init__()
        self.name = NAME_DEVICES_THREAD
        self.daemon = False

        # Init two cameras - Panchromatic and Monochromatic
        params = INIT_CAMERA_PARAMETERS
        params['ffc_mode'] = 'auto'
        params['ffc_period'] = 1800

        func_cam = partial(CameraCtrl, camera_parameters=params, is_dummy=False,
                           time_to_save=10e9)  # dump to disk every 15 seconds
        self._camera_pan = func_cam(path_to_save=path_to_save / 'pan', name='pan')
        self._camera_mono = func_cam(path_to_save=path_to_save / 'mono', name='mono')

    def run(self) -> None:
        self._camera_mono.start()
        while self._camera_mono.camera_parameters_setting_position != EnumParameterPosition.DONE.value:
            sleep(1)
        self._camera_pan.start()

    def __del__(self) -> None:
        try:
            self._camera_mono.terminate()
        except:
            pass
        try:
            self._camera_pan.terminate()
        except:
            pass

    @property
    def rate_pan(self):
        return self._camera_pan.rate_camera

    @property
    def rate_mono(self):
        return self._camera_mono.rate_camera

    @property
    def n_files_pan(self):
        return self._camera_pan.n_files_saved

    @property
    def n_files_mono(self):
        return self._camera_mono.n_files_saved

    def _make_status(self, stat: int):
        stat = EnumParameterPosition(stat)
        if stat == EnumParameterPosition.DONE:
            return 'Ready'
        else:
            return stat.name

    @property
    def status_mono(self) -> str:
        return self._make_status(self._camera_mono._param_setting_pos.value)

    @property
    def status_pan(self) -> str:
        return self._make_status(self._camera_pan._param_setting_pos.value)
