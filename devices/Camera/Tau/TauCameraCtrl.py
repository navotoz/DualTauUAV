import logging
import struct
from pathlib import Path
from time import sleep

import numpy as np
import serial
from serial.tools.list_ports import comports

from devices.Camera import CameraAbstract, WIDTH_IMAGE_TAU2, HEIGHT_IMAGE_TAU2, CAMERA_TAU, T_FPA, T_HOUSING
from devices.Camera.Tau import tau2_config as ptc
from devices.Camera.Tau.tau2_config import ARGUMENT_FPA, ARGUMENT_HOUSING
logger = logging.getLogger(__name__)


class Tau(CameraAbstract):
    conn = None
    _ffc_mode = None

    def __init__(self, port=None, vid: int = 0x10C4, pid: int = 0xEA60, baud=921600):
        super().__init__()
        self._width = WIDTH_IMAGE_TAU2
        self._height = HEIGHT_IMAGE_TAU2

        if not port:
            port = list(filter(lambda x: x.vid == vid and x.pid == pid, comports()))
            port = port[0].device if port else None
        if not port:
            raise IOError('Tau2 with VPC is not connected via a serial connection.')
        self.conn = serial.Serial(port=port, baudrate=baud)

        if self.conn.is_open:
            logger.info("Connected to camera at {}.".format(port))

            self.conn.flushInput()
            self.conn.flushOutput()
            self.conn.timeout = 1

            self.conn.read(self.conn.in_waiting)
        else:
            logger.critical("Couldn't connect to camera!")
            raise RuntimeError

    def __del__(self):
        if self.conn:
            self.conn.close()

    def send_command(self, command: ptc.Code, argument: (bytes, None)) -> (None, bytes):
        raise NotImplementedError

    def _reset(self):
        self.send_command(command=ptc.CAMERA_RESET, argument=None)

    @property
    def type(self) -> int:
        return CAMERA_TAU

    def get_inner_temperature(self, temperature_type: str):
        if T_FPA in temperature_type:
            arg_hex = ARGUMENT_FPA
        elif T_HOUSING in temperature_type:
            arg_hex = ARGUMENT_HOUSING
        else:
            raise TypeError(f'{temperature_type} was not implemented as an inner temperature of TAU2.')
        command = ptc.READ_SENSOR_TEMPERATURE
        argument = struct.pack(">h", arg_hex)
        res = self.send_command(command=command, argument=argument)
        if res:
            res = struct.unpack(">H", res)[0]
            res /= 10.0 if temperature_type == T_FPA else 100.0
            if not 8.0 <= res <= 99.0:  # camera temperature cannot be > 99C or < 8C, returns None.
                logger.debug(f'Error when recv {temperature_type} - got {res}C')
                return None
        return res

    def _get_values_without_arguments(self, command: ptc.Code) -> int:
        res = self.send_command(command=command, argument=None)
        if res is None:
            return 0xffff
        fmt = 'h' if len(res) == 2 else 'hh'
        return struct.unpack('>' + fmt, res)[0]

    def _set_values_with_2bytes_send_recv(self, value: int, current_value: int, command: ptc.Code) -> bool:
        if value == current_value:
            return True
        res = self.send_command(command=command, argument=struct.pack('>h', value))
        if res and struct.unpack('>h', res)[0] == value:
            return True
        return False

    def _log_set_values(self, value: int, result: bool, value_name: str) -> None:
        if result:
            logger.info(f'Set {value_name} to {value}.')
        else:
            logger.warning(f'Setting {value_name} to {value} failed.')

    def _mode_setter(self, mode: str, current_value: int, setter_code: ptc.Code, code_dict: dict, name: str) -> bool:
        if isinstance(mode, str):
            if not mode.lower() in code_dict:
                raise NotImplementedError(f"{name} mode {mode} is not implemented.")
            mode = code_dict[mode.lower()]
        elif isinstance(mode, int) and mode not in code_dict.values():
            raise NotImplementedError(f"{name} mode {mode} is not implemented.")
        res = self._set_values_with_2bytes_send_recv(mode, current_value, setter_code)
        self._log_set_values(mode, res, f'{name} mode')
        return res

    def set_params_by_dict(self, yaml_or_dict: (Path, dict)):
        pass

    @property
    def is_dummy(self) -> bool:
        return False

    def grab(self, to_temperature: bool) -> np.ndarray:
        pass

    def ffc(self, length: bytes = ptc.FFC_LONG) -> bool:
        prev_mode = self.ffc_mode
        if 'ext' in prev_mode:
            while 'man' not in self.ffc_mode:
                self.ffc_mode = ptc.FFC_MODE_CODE_DICT['manual']
            sleep(0.2)
        res = self.send_command(command=ptc.DO_FFC, argument=length)
        sleep(0.2)
        if 'ext' in prev_mode:
            while 'ext' not in self.ffc_mode:
                self.ffc_mode = ptc.FFC_MODE_CODE_DICT['external']
            sleep(0.2)
        if res and struct.unpack('H', res)[0] == 0xffff:
            t_fpa = self.get_inner_temperature(T_FPA)
            t_housing = self.get_inner_temperature(T_HOUSING)
            f_log = 'FFC.'
            if t_fpa:
                f_log += f' FPA {t_fpa:.2f}C'
            if t_housing:
                f_log += f', Housing {t_housing:.2f}'
            logger.info(f_log)
            return True
        else:
            logger.info('FFC Failed')
            return False

    @property
    def ffc_mode(self) -> str:
        if self._ffc_mode is None:
            res = 0xffff
            while res == 0xffff:
                res = self._get_values_without_arguments(ptc.GET_FFC_MODE)
            self._ffc_mode = {v: k for k, v in ptc.FFC_MODE_CODE_DICT.items()}[res]
        return self._ffc_mode

    @ffc_mode.setter
    def ffc_mode(self, mode: str):
        if self._mode_setter(mode=mode, current_value=ptc.FFC_MODE_CODE_DICT[self.ffc_mode],
                             setter_code=ptc.SET_FFC_MODE, code_dict=ptc.FFC_MODE_CODE_DICT, name='FCC'):
            if isinstance(mode, int):
                self._ffc_mode = {v: k for k, v in ptc.FFC_MODE_CODE_DICT.items()}[mode]
            else:
                self._ffc_mode = mode

    @property
    def ffc_period(self) -> int:
        return self._get_values_without_arguments(ptc.GET_FFC_PERIOD)

    @ffc_period.setter
    def ffc_period(self, period: int):
        if not 0 <= period <= 30000:
            raise ValueError(f'Given FFC period {period} not in 0 <= period <= 30000.')
        res = False
        for _ in range(5):
            res = self._set_values_with_2bytes_send_recv(value=period, current_value=-1, command=ptc.SET_FFC_PERIOD)
            if res is True:
                break
            sleep(1)
        self._log_set_values(value=period, result=res, value_name='FFC Period')

    @property
    def correction_mask(self):
        """ the default value is 2111 (decimal). 0 (decimal) is all off """
        return self._get_values_without_arguments(ptc.GET_CORRECTION_MASK)

    @correction_mask.setter
    def correction_mask(self, mode: str):
        self._mode_setter(mode, self.correction_mask, ptc.SET_CORRECTION_MASK, ptc.FFC_MODE_CODE_DICT, 'FCC')

    @property
    def gain(self):
        return self._get_values_without_arguments(ptc.GET_GAIN_MODE)

    @gain.setter
    def gain(self, mode: str):
        self._mode_setter(mode, self.gain, ptc.SET_GAIN_MODE, ptc.GAIN_CODE_DICT, 'Gain')

    @property
    def agc(self):
        return self._get_values_without_arguments(ptc.GET_AGC_ALGORITHM)  # todo: does this function even works????

    @agc.setter
    def agc(self, mode: str):
        self._mode_setter(mode, self.agc, ptc.SET_AGC_ALGORITHM, ptc.AGC_CODE_DICT, 'AGC')

    @property
    def sso(self) -> int:
        res = self.send_command(command=ptc.GET_AGC_THRESHOLD, argument=struct.pack('>h', 0x0400))
        return struct.unpack('>h', res)[0] if res else 0xffff

    @sso.setter
    def sso(self, percentage: (int, tuple)):
        if percentage == self.sso:
            logger.info(f'Set SSO to {percentage}')
            return
        self.send_command(command=ptc.SET_AGC_THRESHOLD, argument=struct.pack('>hh', 0x0400, percentage))
        if self.sso == percentage:
            logger.info(f'Set SSO to {percentage}%')
            return
        logger.warning(f'Setting SSO to {percentage}% failed.')

    @property
    def contrast(self) -> int:
        return self._get_values_without_arguments(ptc.GET_CONTRAST)

    @contrast.setter
    def contrast(self, value: int):
        self._log_set_values(value, self._set_values_with_2bytes_send_recv(value, self.contrast, ptc.SET_CONTRAST),
                             'AGC contrast')

    @property
    def brightness(self) -> int:
        return self._get_values_without_arguments(ptc.GET_BRIGHTNESS)

    @brightness.setter
    def brightness(self, value: int):
        self._log_set_values(value, self._set_values_with_2bytes_send_recv(value, self.brightness, ptc.SET_BRIGHTNESS),
                             'AGC brightness')

    @property
    def brightness_bias(self) -> int:
        return self._get_values_without_arguments(ptc.GET_BRIGHTNESS_BIAS)

    @brightness_bias.setter
    def brightness_bias(self, value: int):
        result = self._set_values_with_2bytes_send_recv(value, self.brightness_bias, ptc.SET_BRIGHTNESS_BIAS)
        self._log_set_values(value, result, 'AGC brightness_bias')

    @property
    def isotherm(self) -> int:
        return self._get_values_without_arguments(ptc.GET_ISOTHERM)

    @isotherm.setter
    def isotherm(self, value: int):
        result = self._set_values_with_2bytes_send_recv(value, self.isotherm, ptc.SET_ISOTHERM)
        self._log_set_values(value, result, 'IsoTherm')

    @property
    def dde(self) -> int:
        return self._get_values_without_arguments(ptc.GET_SPATIAL_THRESHOLD)

    @dde.setter
    def dde(self, value: int):
        result = self._set_values_with_2bytes_send_recv(value, self.dde, ptc.SET_SPATIAL_THRESHOLD)
        self._log_set_values(value, result, 'DDE')

    @property
    def tlinear(self):
        res = self.send_command(command=ptc.GET_TLINEAR_MODE, argument=struct.pack('>h', 0x0040))
        return struct.unpack('>h', res)[0] if res else 0xffff

    @tlinear.setter
    def tlinear(self, value: int):
        if value == self.tlinear:
            logger.info(f'Set TLinear to {value}.')
            return
        self.send_command(command=ptc.SET_TLINEAR_MODE, argument=struct.pack('>hh', 0x0040, value))
        if value == self.tlinear:
            self._log_set_values(value, True, 'tlinear mode')
            return
        self._log_set_values(value, False, 'tlinear mode')

    def _digital_output_getter(self, command: ptc.Code, argument: bytes):
        res = self.send_command(command=command, argument=argument)
        return struct.unpack('>h', res)[0] if res else 0xffff

    def _digital_output_setter(self, mode: int, current_mode: int, command: ptc.Code, argument: int) -> bool:
        if mode == current_mode:
            return True
        res = self.send_command(command=command, argument=struct.pack('>bb', argument, mode))
        if res and struct.unpack('>bb', res)[-1] == mode:
            return True
        return False

    @property
    def lvds(self):
        return self._digital_output_getter(ptc.GET_LVDS_MODE, struct.pack('>h', 0x0400))

    @lvds.setter
    def lvds(self, mode: int):
        res = self._digital_output_setter(mode, self.lvds, ptc.SET_LVDS_MODE, 0x05)
        self._log_set_values(mode, res, 'lvds mode')

    @property
    def lvds_depth(self):
        return self._digital_output_getter(ptc.GET_LVDS_DEPTH, struct.pack('>h', 0x0900))

    @lvds_depth.setter
    def lvds_depth(self, mode: int):
        res = self._digital_output_setter(mode, self.lvds_depth, ptc.SET_LVDS_DEPTH, 0x07)
        self._log_set_values(mode, res, 'lvds depth')

    @property
    def xp(self):
        return self._digital_output_getter(ptc.GET_XP_MODE, struct.pack('>h', 0x0200))

    @xp.setter
    def xp(self, mode: int):
        res = self._digital_output_setter(mode, self.xp, ptc.SET_XP_MODE, 0x03)
        self._log_set_values(mode, res, 'xp mode')

    @property
    def cmos_depth(self):
        return self._digital_output_getter(ptc.GET_CMOS_DEPTH, struct.pack('>h', 0x0800))

    @cmos_depth.setter
    def cmos_depth(self, mode: int):
        res = self._digital_output_setter(mode, self.cmos_depth, ptc.SET_CMOS_DEPTH, 0x06)
        self._log_set_values(mode, res, 'CMOS Depth')

    @property
    def fps(self):
        return self._get_values_without_arguments(ptc.GET_FPS)

    @fps.setter
    def fps(self, mode: str):
        self._mode_setter(mode, self.fps, ptc.SET_FPS, ptc.FPS_CODE_DICT, 'FPS')

    def reset(self):
        return self.send_command(command=ptc.CAMERA_RESET, argument=None)

    @property
    def ace(self):
        return self._get_values_without_arguments(ptc.GET_AGC_ACE_CORRECT)

    @ace.setter
    def ace(self, value: int):
        if not -8 <= value <= 8:
            return
        for _ in range(5):
            self.send_command(command=ptc.SET_AGC_ACE_CORRECT, argument=struct.pack('>h', value))
            if value == self.ace:
                logger.info(f'Set ACE to {value}.')
                return

    @property
    def lens_number(self):
        return self._get_values_without_arguments(ptc.GET_LENS_NUMBER)

    @lens_number.setter
    def lens_number(self, value: int):
        if not 1 <= value <= 2:
            return
        value -= 1  # the terms of lenses is 0x0001 or 0x0000
        for _ in range(5):
            res = self.send_command(command=ptc.SET_LENS_NUMBER, argument=struct.pack('>h', value))
            try:
                res = struct.unpack('>h', res)[0]
            except (TypeError, struct.error, IndexError, RuntimeError, AttributeError):
                continue
            if value == res:
                logger.info(f'Set Lens number to {value + 1}.')
                return
