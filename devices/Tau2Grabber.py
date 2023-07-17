from time import sleep, time_ns
from typing import Optional, Tuple, Union
from usb.core import USBError
from pyftdi.ftdi import Ftdi, FtdiError
import yaml
import numpy as np
from pathlib import Path
import struct

from devices.utils import (connect_ftdi, is_8bit_image_borders_valid,
                           REPLY_HEADER_BYTES, parse_incoming_message, make_packet)
from devices import HEIGHT_IMAGE_TAU2, T_FPA, T_HOUSING, WIDTH_IMAGE_TAU2, EnumParameterPosition
import devices.tau2_config as ptc
from utils.tools import make_logger

KELVIN2CELSIUS = 273.15
SYNC_MSG = b'SYNC' + struct.pack(4 * 'B', *[0, 0, 0, 0])
TEAX_IN_BYTES = b'TEAX'
TEAX_LEN = len(TEAX_IN_BYTES)


class Tau2:
    _ffc_mode = None

    def __init__(self, vid=0x0403, pid=0x6010, name: str = ''):
        self._logger = make_logger(name=f'{name}Tau2')

        try:
            self._ftdi, address = connect_ftdi(vid, pid)
        except (RuntimeError, USBError):
            raise RuntimeError('Could not connect to the Tau2 camera.')
        self._ftdi.read_data_set_chunksize(chunksize=2**14)
        self._ftdi_read_chunksize = self._ftdi.read_data_get_chunksize()
        self.param_position: EnumParameterPosition = EnumParameterPosition.DISCONNECTED

        self._frame_size = 2 * self.height * self.width + 6 + 4 * self.height  # 6 byte header, 4 bytes pad per row
        self._logger.info(f'Found device in {address}.')

    @property
    def width(self) -> int:
        return WIDTH_IMAGE_TAU2

    @property
    def height(self) -> int:
        return HEIGHT_IMAGE_TAU2

    def __del__(self) -> None:
        if hasattr(self, '_ftdi') and isinstance(self._ftdi, Ftdi):
            self._ftdi.close()

    def get_inner_temperature(self, temperature_type: str):
        if T_FPA in temperature_type:
            arg_hex = ptc.ARGUMENT_FPA
        elif T_HOUSING in temperature_type:
            arg_hex = ptc.ARGUMENT_HOUSING
        else:
            raise TypeError(f'{temperature_type} was not implemented as an inner temperature of TAU2.')
        command = ptc.READ_SENSOR_TEMPERATURE
        argument = struct.pack(">h", arg_hex)
        res = self.send_command(command=command, argument=argument, timeout=1)
        if res:
            res = struct.unpack(">H", res)[0]
            res /= 10.0 if temperature_type == T_FPA else 100.0
            if not 8.0 <= res <= 99.0:  # camera temperature cannot be > 99C or < 8C, returns None.
                self._logger.debug(f'Error when recv {temperature_type} - got {res}C')
                return None
        return res

    def _get_values_without_arguments(self, command: ptc.Code) -> int:
        res = self.send_command(command=command, argument=None, timeout=1)
        if res is None:
            return 0xffff
        fmt = 'h' if len(res) == 2 else 'hh'
        return struct.unpack('>' + fmt, res)[0]

    def _set_values_with_2bytes_send_recv(self, value: int, current_value: int, command: ptc.Code) -> bool:
        if value == current_value:
            return True
        res = self.send_command(command=command, argument=struct.pack('>h', value), timeout=1)
        if res and struct.unpack('>h', res)[0] == value:
            return True
        return False

    def _log_set_values(self, value: int, result: bool, value_name: str) -> None:
        if result:
            self._logger.info(f'Set {value_name} to {value}.')
        else:
            self._logger.warning(f'Setting {value_name} to {value} failed.')

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

    def ffc(self, length: bytes = ptc.FFC_LONG) -> bool:
        prev_mode = self.ffc_mode
        if 'ext' in prev_mode:
            while 'man' not in self.ffc_mode:
                self.ffc_mode = ptc.FFC_MODE_CODE_DICT['manual']
            sleep(0.2)
        res = self.send_command(command=ptc.DO_FFC, argument=length, timeout=1)
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
            self._logger.info(f_log)
            return True
        else:
            self._logger.info('FFC Failed')
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
        res = self.send_command(command=ptc.GET_AGC_THRESHOLD, argument=struct.pack('>h', 0x0400), timeout=1)
        return struct.unpack('>h', res)[0] if res else 0xffff

    @sso.setter
    def sso(self, percentage: Union[int, tuple]):
        if percentage == self.sso:
            self._logger.info(f'Set SSO to {percentage}')
            return
        self.send_command(command=ptc.SET_AGC_THRESHOLD, argument=struct.pack('>hh', 0x0400, percentage), timeout=1)
        if self.sso == percentage:
            self._logger.info(f'Set SSO to {percentage}%')
            return
        self._logger.warning(f'Setting SSO to {percentage}% failed.')

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
        res = self.send_command(command=ptc.GET_TLINEAR_MODE, argument=struct.pack('>h', 0x0040), timeout=1)
        return struct.unpack('>h', res)[0] if res else 0xffff

    @tlinear.setter
    def tlinear(self, value: int):
        if value == self.tlinear:
            self._logger.info(f'Set TLinear to {value}.')
            return
        self.send_command(command=ptc.SET_TLINEAR_MODE, argument=struct.pack('>hh', 0x0040, value), timeout=1)
        if value == self.tlinear:
            self._log_set_values(value, True, 'tlinear mode')
            return
        self._log_set_values(value, False, 'tlinear mode')

    def _digital_output_getter(self, command: ptc.Code, argument: bytes):
        res = self.send_command(command=command, argument=argument, timeout=1)
        return struct.unpack('>h', res)[0] if res else 0xffff

    def _digital_output_setter(self, mode: int, current_mode: int, command: ptc.Code, argument: int) -> bool:
        if mode == current_mode:
            return True
        res = self.send_command(command=command, argument=struct.pack('>bb', argument, mode), timeout=1)
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
        return self.send_command(command=ptc.CAMERA_RESET, argument=None, timeout=1)

    @property
    def ace(self):
        return self._get_values_without_arguments(ptc.GET_AGC_ACE_CORRECT)

    @ace.setter
    def ace(self, value: int):
        if not -8 <= value <= 8:
            return
        for _ in range(5):
            self.send_command(command=ptc.SET_AGC_ACE_CORRECT, argument=struct.pack('>h', value), timeout=1)
            if value == self.ace:
                self._logger.info(f'Set ACE to {value}.')
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
            res = self.send_command(command=ptc.SET_LENS_NUMBER, argument=struct.pack('>h', value), timeout=1)
            try:
                res = struct.unpack('>h', res)[0]
            except (TypeError, struct.error, IndexError, RuntimeError, AttributeError):
                continue
            if value == res:
                self._logger.info(f'Set Lens number to {value + 1}.')
                return

    def set_params_by_dict(self, yaml_or_dict: Union[Path, dict]):
        SLEEP_BETWEEN_PARAMS = 0.4
        if isinstance(yaml_or_dict, Path):
            params = yaml.safe_load(yaml_or_dict)
        else:
            params = yaml_or_dict.copy()
        self.param_position = EnumParameterPosition.CONNECTED
        self.ffc_mode = params.get('ffc_mode', 'manual')
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.FFC_MODE
        self.ffc_period = params.get('ffc_period', 0)  # default is no ffc
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.FFC_PERIOD
        self.ace = params.get('ace', 0)
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.ACE
        self.tlinear = params.get('tlinear', 0)
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.TLINEAR
        self.isotherm = params.get('isotherm', 0)
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.ISOTHERM
        self.dde = params.get('dde', 0)
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.DDE
        self.gain = params.get('gain', 'high')
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.GAIN
        self.agc = params.get('agc', 'manual')
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.AGC
        self.sso = params.get('sso', 0)
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.SSO
        self.contrast = params.get('contrast', 0)
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.CONTRAST
        self.brightness = params.get('brightness', 0)
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.BRIGHTNESS
        self.brightness_bias = params.get('brightness_bias', 0)
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.BRIGHTNESS_BIAS
        self.cmos_depth = params.get('cmos_depth', 0)  # 14bit pre AGC
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.CMOS_DEPTH
        self.fps = params.get('fps', ptc.FPS_CODE_DICT[60])  # 60Hz NTSC
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.FPS
        self.lens_number = params.get('lens_number', 1)
        sleep(SLEEP_BETWEEN_PARAMS)
        self.param_position = EnumParameterPosition.LENS_NUMBER
        # self.correction_mask = params.get('corr_mask', 0)  # Always OFF!!!
        self.param_position = EnumParameterPosition.DONE

    def _write(self, data: bytes) -> None:
        try:
            self._ftdi.write_data(data)
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, FtdiError) as e:
            self._logger.error('Write error ' + str(e))

    def _read(self, length_of_command_in_bytes: int) -> bytes:
        time_start = time_ns()
        buffer = b''
        while time_ns() - time_start < 5e8:
            try:
                data = self._ftdi.read_data(self._ftdi_read_chunksize)
            except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, FtdiError):
                self._logger.error('Reader failed')
                raise RuntimeError('Reader failed')
            if data is not None:
                buffer += data
            if len(buffer) >= length_of_command_in_bytes:
                break
        return buffer

    def send_command(self, command: ptc.Code, argument: Optional[bytes] = None, timeout: float = 1.) -> Optional[bytes]:
        parsed_msg = None
        try:
            data = make_packet(command, argument)
            self._ftdi.set_bitmode(0xFF, Ftdi.BitMode.RESET)
            time_start = time_ns()
            while time_ns() - time_start <= timeout * 1e9:
                self._write(data)
                ret_val = self._read(length_of_command_in_bytes=command.reply_bytes + REPLY_HEADER_BYTES)
                parsed_msg = parse_incoming_message(buffer=ret_val, command=command)
                if parsed_msg is not None:
                    break
        except Exception as e:
            self._logger.error(f"Failed to send command: {str(e)}")
        finally:
            self._ftdi.set_bitmode(0xFF, Ftdi.BitMode.SYNCFF)
            return parsed_msg

    def grab(self, to_temperature: bool = False) -> Tuple[np.ndarray, int, int]:
        # Sync to the next frame by the TEAX magic word
        time_of_frame = time_ns()
        res = bytes(self._ftdi.read_data_bytes(TEAX_LEN + self._frame_size, attempt=1))
        time_of_end = time_ns()
        if not res.startswith(TEAX_IN_BYTES):
            return None, time_of_frame, time_of_end
        res = res[TEAX_LEN:]

        # Data must have at least 8 bytes to find the magic word and frame width
        if len(res) != self._frame_size:
            return None, time_of_frame, time_of_end

        # Check the magic word and frame width
        magic_word = struct.unpack('h', res[6:8])[0]
        frame_width = struct.unpack('h', res[1:3])[0] - 2
        if magic_word != 0x4000 or frame_width != self.width:
            return None, time_of_frame, time_of_end

        # Check the size of the frames
        raw_image_8bit = np.frombuffer(res[6:], dtype='uint8')
        if len(raw_image_8bit) != (2 * (self.width + 2)) * self.height:
            return None, time_of_frame, time_of_end

        # Check the borders
        raw_image_8bit = raw_image_8bit.reshape((-1, 2 * (self.width + 2)))
        if not is_8bit_image_borders_valid(raw_image_8bit, self.height):
            return None, time_of_frame, time_of_end

        # Convert to 16bit and remove the borders
        raw_image_16bit = 0x3FFF & np.array(raw_image_8bit).view('uint16')[:, 1:-1]
        if to_temperature:
            raw_image_16bit = 0.04 * raw_image_16bit - KELVIN2CELSIUS
        return raw_image_16bit, time_of_frame, time_of_end
