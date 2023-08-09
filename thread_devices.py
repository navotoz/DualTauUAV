from functools import partial
import threading as th
import multiprocessing as mp

from devices import INIT_CAMERA_PARAMETERS, EnumParameterPosition
from devices.CameraProcess import CameraCtrl
NAME_DEVICES_THREAD = 'th_devices'


class ThreadDevices(th.Thread):
    def __init__(self, path_to_save: mp.Value) -> None:
        super().__init__()
        self.name = NAME_DEVICES_THREAD
        self.daemon = False

        # Init two cameras - Panchromatic and Monochromatic
        params = INIT_CAMERA_PARAMETERS
        params['ffc_mode'] = 'manual'  # manually control the ffc from the CameraProcess
        params['ffc_period'] = 0

        # Init cameras
        func_cam = partial(CameraCtrl,
                           camera_parameters=params,
                           is_dummy=False,
                           limit_rate_camera=30,
                           path_to_save=path_to_save,
                           time_to_save=2e9)  # dump to disk every 2 seconds
        self._camera = func_cam(name='pan')

    def run(self) -> None:
        self._camera.start()

    def __del__(self) -> None:
        try:
            self._camera.terminate()
        except Exception:
            pass

    @property
    def rate(self):
        return self._camera.rate_camera

    @property
    def n_files(self):
        return self._camera.n_files_saved

    def _make_status(self, stat: int):
        stat = EnumParameterPosition(stat)
        if stat == EnumParameterPosition.DONE:
            return 'Ready'
        else:
            return stat.name

    @property
    def status(self) -> str:
        return self._make_status(self._camera._param_setting_pos.value)

    @property
    def frame(self):
        return self._camera.frame
