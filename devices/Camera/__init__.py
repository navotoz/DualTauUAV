from abc import abstractmethod
from pathlib import Path
from enum import IntEnum, unique
import usb.core
import usb.util
from numpy import ndarray

CAMERA_TAU = 'tau2'

T_FPA = 'fpa'
T_HOUSING = 'housing'

HEIGHT_IMAGE_TAU2 = 256
WIDTH_IMAGE_TAU2 = 336


@unique
class EnumParameterPosition(IntEnum):
    DISCONNECTED = 0
    CONNECTED = 1
    FFC_MODE = 2
    FFC_PERIOD = 3
    ACE = 4
    TLINEAR = 5
    ISOTHERM = 6
    DDE = 7
    GAIN = 8
    AGC = 9
    SSO = 10
    CONTRAST = 11
    BRIGHTNESS = 12
    BRIGHTNESS_BIAS = 13
    CMOS_DEPTH = 14
    FPS = 15
    LENS_NUMBER = 16
    DONE = 17


INIT_CAMERA_PARAMETERS = dict(
    lens_number=2,
    ffc_mode='auto',
    ffc_period=1800,
    isotherm=0x0000,
    dde=0x0000,
    tlinear=0,  # T-Linear disabled. The scene will not represent temperatures, because of the filters.
    gain='high',
    agc='manual',
    ace=0,
    sso=0,
    contrast=0,
    brightness=0,
    brightness_bias=0,
    fps=0x0004,  # 60Hz
    lvds=0x0000,  # disabled
    lvds_depth=0x0000,  # 14bit
    xp=0x0002,  # 14bit w/ 1 discrete
    cmos_depth=0x0000,  # 14bit pre AGC
)


class CameraAbstract:
    def __init__(self):
        self._width = 1
        self._height = 1

    @abstractmethod
    def __del__(self):
        pass

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @abstractmethod
    def get_inner_temperature(self, temperature_type: str) -> float:
        pass

    @property
    @abstractmethod
    def type(self) -> int:
        pass

    @abstractmethod
    def set_params_by_dict(self, yaml_or_dict: (Path, dict)):
        pass

    @property
    @abstractmethod
    def is_dummy(self) -> bool:
        pass

    @abstractmethod
    def grab(self, to_temperature: bool) -> ndarray:
        pass

    @abstractmethod
    def ffc(self):
        pass

    @property
    def correction_mask(self):
        return

    @correction_mask.setter
    def correction_mask(self, mode: str):
        pass

    @property
    def ffc_mode(self):
        return

    @ffc_mode.setter
    def ffc_mode(self, mode: str):
        pass

    @property
    def gain(self):
        return

    @gain.setter
    def gain(self, mode: str):
        pass

    @property
    def agc(self):
        return

    @agc.setter
    def agc(self, mode: str):
        pass

    @property
    def sso(self):
        return

    @sso.setter
    def sso(self, percentage: (int, tuple)):
        pass

    @property
    def contrast(self):
        return

    @contrast.setter
    def contrast(self, value: int):
        pass

    @property
    def brightness(self):
        return

    @brightness.setter
    def brightness(self, value: int):
        pass

    @property
    def brightness_bias(self):
        return

    @brightness_bias.setter
    def brightness_bias(self, value: int):
        pass

    @property
    def isotherm(self):
        return

    @isotherm.setter
    def isotherm(self, value: int):
        pass

    @property
    def dde(self):
        return

    @dde.setter
    def dde(self, value: int):
        pass

    @property
    def tlinear(self):
        return

    @tlinear.setter
    def tlinear(self, value: int):
        pass

    @property
    def lvds(self):
        return

    @lvds.setter
    def lvds(self, mode: int):
        pass

    @property
    def lvds_depth(self):
        return

    @lvds_depth.setter
    def lvds_depth(self, mode: int):
        pass

    @property
    def xp(self):
        return

    @xp.setter
    def xp(self, mode: int):
        pass

    @property
    def cmos_depth(self):
        return

    @cmos_depth.setter
    def cmos_depth(self, mode: int):
        pass

    @property
    def fps(self):
        return

    @fps.setter
    def fps(self, fps_to_set: int):
        pass


def _make_device_from_vid_pid(vid: int, pid: int) -> usb.core.Device:
    device = usb.core.find(idVendor=vid, idProduct=pid)
    if not device:
        raise RuntimeError

    if device.is_kernel_driver_active(0):
        device.detach_kernel_driver(0)

    device.reset()
    for cfg in device:
        for intf in cfg:
            if device.is_kernel_driver_active(intf.bInterfaceNumber):
                try:
                    device.detach_kernel_driver(intf.bInterfaceNumber)
                except usb.core.USBError as e:
                    print(f"Could not detach kernel driver from interface({intf.bInterfaceNumber}): {e}")
    device.set_configuration(1)
    return device
