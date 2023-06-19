import binascii
import logging
import struct
import sys
import threading as th
import time
from dataclasses import dataclass
from datetime import datetime

import serial
import serial.tools.list_ports

from utils.tools import SyncFlag
logger = logging.getLogger(__name__)


def check_nmea_crc(sentence):
    if isinstance(sentence, bytes):
        sentence = sentence.decode('UTF-8')

    # remove /r/n
    data_to_check = sentence.rstrip()

    # remove crc by removing the chars after *
    try:
        data_to_check, crc = data_to_check.split('*')
    except ValueError:
        return False

    # remove $
    data_to_check = data_to_check.replace('$', '')

    # Initializing our first XOR value
    sum_of_chars = 0

    # For each char in chksumdata, XOR against the previous
    # XOR'd char.  The final XOR of the last char will be our
    # checksum to verify against the checksum we sliced off
    # the NMEA sentence

    for c in data_to_check:
        # XOR'ing value of sum_of_chars against the next char in line
        # and storing the new XOR value in sum_of_chars
        sum_of_chars ^= ord(c)

    # Do we have a validated sentence?
    if hex(sum_of_chars) == hex(int(crc, 16)):
        return True

    return False


# noinspection PyPep8Naming
class _UbxMessage(object):
    def __init__(self, ubx_class, ubx_id, msg_type="rx", **kwargs):
        if msg_type == "rx":
            self._version = kwargs["version"]
            if ubx_class == '05':
                message = {'01': lambda: self._ubx_ACK_ACK(kwargs["dev"]),
                           '00': lambda: self._ubx_ACK_NAK(kwargs["dev"])}
                message[ubx_id]()

        elif msg_type == "tx":
            if ubx_class == '06':
                message = {'08': lambda: self._ubx_CFG_RATE(kwargs["rate"], kwargs["timeRef"])}
                message[ubx_id]()

    def _ubx_CFG_RATE(self, rate, timeRef):
        header, ubx_class, ubx_id, length = 46434, 6, 8, 6

        rate = hex(rate)
        rate = rate[2:]
        while len(rate) < 4:
            rate = '0' + rate

        rate1, rate2 = int(rate[2:4], 16), int(rate[:2], 16)

        navRate = 1  # according to ublox ICD this value is a don't care
        if rate != '0000':
            payload = [length, 0, rate1, rate2, navRate, 0, 0, timeRef]
        else:
            payload = [0, 0]
        checksum = self._calc_checksum(ubx_class, ubx_id, payload)
        payload = payload + checksum
        try:
            self.msg = struct.pack(f'>H{2 + len(payload)}B', header, ubx_class, ubx_id, *payload)
            self.ubx_class = '06'
            self.ubx_id = '08'
        except struct.error:
            print("{} {}".format(sys.exc_info()[0], sys.exc_info()[1]))

    @staticmethod
    def _calc_checksum(ubx_class, ubx_id, payload):
        check1 = (ubx_class + ubx_id) % 256
        check2 = ((2 * ubx_class) + ubx_id) % 256

        for i in range(0, len(payload)):
            check1 = (check1 + payload[i]) % 256
            check2 = (check1 + check2) % 256

        result = [check1, check2]
        return result

    def _ubx_ACK_ACK(self, dev):
        payload = dev.read(size=4)
        payload_cpy = payload

        if self._validate_checksum(5, 1, payload, dev):
            try:
                payload_cpy = payload_cpy[2:]
                self.clsID, self.msgID = struct.unpack('=BB', payload_cpy)
                self.clsID, self.msgID = hex(self.clsID), hex(self.msgID)
                self.ubx_class = '05'
                self.ubx_id = '01'

            except struct.error:
                print("{} {}".format(sys.exc_info()[0], sys.exc_info()[1]))

    # UBX-ACK-NAK (0x05 0x00)
    def _ubx_ACK_NAK(self, dev):
        payload = dev.read(size=4)
        payload_cpy = payload

        if self._validate_checksum(5, 0, payload, dev):
            try:
                payload_cpy = payload_cpy[2:]
                self.clsID, self.msgID = struct.unpack('=BB', payload_cpy)
                self.clsID, self.msgID = hex(self.clsID), hex(self.msgID)
                self.ubx_class = '05'
                self.ubx_id = '00'

            except struct.error:
                print("{} {}".format(sys.exc_info()[0], sys.exc_info()[1]))

    @staticmethod
    def _validate_checksum(ubx_class, ubx_id, payload, dev):
        check1 = (ubx_class + ubx_id) % 256
        check2 = ((2 * ubx_class) + ubx_id) % 256

        chk1 = dev.read()[0]
        chk2 = dev.read()[0]

        for i in range(0, len(payload)):
            check1 = (check1 + payload[i]) % 256
            check2 = (check1 + check2) % 256

        if chk1 == check1 and chk2 == check2:
            return True
        else:
            print("Checksum is incorrect")
            return False


@dataclass(frozen=False)
class GpsData:
    time: float = 0.0
    latitude: float = None
    longitude: float = None
    altitude: float = None
    speed: float = None
    __valid: bool = False

    def __eq__(self, other):
        return self.latitude == other.latitude and \
               self.longitude == other.longitude and \
               self.altitude == other.altitude and \
               self.time == other.time

    def __sub__(self, other):
        return (self.latitude - other.latitude,
                self.longitude - other.longitude,
                self.altitude - other.altitude,
                self.time - other.time)

    def __repr__(self):
        msg = 'GpsData(INVALID)'
        if self.__valid:
            msg = 'GpsData('
            msg += f'{self.get_datetime()},' if self.time else ''
            msg += f' lat: {self.latitude:.4f},' if self.latitude else ''
            msg += f' lng: {self.longitude:.4f},' if self.longitude else ''
            msg += f' alt: {self.altitude:.1f}m,' if self.altitude else ''
            msg += f' speed: {self.speed:.1f}m/s,' if self.speed else ''
            msg += ')'
        return msg

    def __post_init__(self) -> None:
        if self.latitude != 0.0 and self.longitude != 0.0 and self.time != 0.0:
            self.__valid = True

    def get_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.time)

    def __bool__(self) -> bool:
        return self.__valid


def parse_gps_data(gps_data: (tuple, GpsData)) -> (GpsData, None):
    if gps_data is None or not gps_data:
        return None

    if isinstance(gps_data, GpsData):
        return gps_data

    lat, lng, alt, date_str, time_stamp = gps_data
    try:
        if not date_str:
            date_str = 1120
        if isinstance(time_stamp, str) and '.' not in time_stamp:
            time_stamp = datetime.strptime(f'{time_stamp}_{date_str}', "%H%M%S%f_%d%m%y").timestamp()
        else:
            time_stamp = datetime.strptime(f'{time_stamp}_{date_str}', "%H%M%S.%f_%d%m%y").timestamp()
    except ValueError:
        time_stamp = 0.0
    return GpsData(time=float(time_stamp), latitude=float(lat), longitude=float(lng), altitude=float(alt))


class GPS:
    def __init__(self, vid: int = 5446, pid: int = 423):
        """
        Reference to commands at https://aprs.gids.nl/nmea/
        """
        super().__init__()
        self._flag_run = SyncFlag(init_state=True)
        self._lock_dev = th.RLock()
        self._lock_coord = th.Lock()

        dev = [p for p in serial.tools.list_ports.comports() if p.vid == vid and p.pid == pid]
        if not dev:
            raise ConnectionError(f'Serial device with VID {vid} and PID {pid} was not found.')
        try:
            self._dev = serial.Serial(dev[0].device, baudrate=9600, timeout=0.5)
        except serial.SerialException:
            raise RuntimeError(f'Serial was not found at port {dev[0].device}')

        self._coordinates: GpsData = parse_gps_data(())
        self._date = None
        self._rate_hz = 20

        logger.info(f'Initiating GPS and setting rate to {self._rate_hz}Hz')
        self._th_sampler = th.Thread(target=self._th_gps_sampler, name='th_gps_sampler', daemon=True)
        self._th_sampler.start()

    @property
    def is_alive(self) -> bool:
        return self._flag_run()

    def terminate(self) -> None:
        try:
            self._flag_run.set(False)
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
            pass
        try:
            if self._dev.is_open:
                self._dev.close()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
            pass

    def _check_valid(self, data: str, valid_bit: bool) -> bool:
        if not valid_bit:
            # V = Warning, most likely, there are no satellites in view...
            return False
        if not check_nmea_crc(data):
            logger.error('CRC error.')
            return False
        return True

    def _th_gps_sampler(self) -> None:
        try:
            self.sampling_rate_hz = self._rate_hz
            logger.info(f'Sampling rate set to {self._rate_hz}Hz')
        except RuntimeError:
            logger.error(f'Could not set sampling rate to {self._rate_hz}Hz')
        logger.info('Connected.')

        while self._flag_run:
            with self._lock_dev:
                try:
                    data = self._dev.readline()
                except (serial.PortNotOpenError, RuntimeError, NameError, ValueError, OSError,
                        ConnectionError, TypeError):
                    self.terminate()
            if data:
                try:
                    data = data.decode('UTF-8')
                except (UnicodeDecodeError, AttributeError, TypeError, ValueError, RuntimeError, KeyError, NameError):
                    self.terminate()

                message = data[0:6]
                parts = data.split(',')

                if message == '$GPRMC':
                    if not self._check_valid(data=data, valid_bit=parts[2] == 'A'):
                        continue

                    with self._lock_coord:
                        self._date = parts[9]

                elif message == '$GPGGA':
                    if not self._check_valid(data=data, valid_bit=int(parts[6]) > 0):
                        continue

                    latitude = parts[2]
                    # lat_hem = parts[3]  # N or S
                    longitude = parts[4]
                    # long_hem = parts[5]  # E or W
                    altitude = float(parts[9])

                    with self._lock_coord:
                        gps_data = (latitude, longitude, altitude, self._date, parts[1])
                        self._coordinates = parse_gps_data(gps_data=gps_data)

                else:
                    # Handle other NMEA messages and unsupported strings
                    pass

    def _write(self, msg: bytes) -> None:
        with self._lock_dev:
            self._dev.write(msg)

    @property
    def sampling_rate_hz(self) -> int:
        with self._lock_dev:
            self._write(_UbxMessage('06', '08', msg_type="tx", rate=0, timeRef=0).msg)
            now = time.time_ns()
            while (time.time_ns() - now) < 5 * 1e9:
                if self._dev.in_waiting > 0:
                    res = self._dev.read_until(b'\xb5b')
                    ubx_class = binascii.hexlify(res[:1]).decode('UTF-8')
                    ubx_id = binascii.hexlify(res[1:2]).decode('UTF-8')
                    if ubx_id and ubx_class and ubx_class == '06' and ubx_id == '08':
                        res = struct.unpack(f'<{len(res) - 2}BH', res)[2:]
                        if res and len(res) >= 3 and res[0] == 6:
                            self._rate_hz = int((1 / res[2]) * 1e3)
                            break
        return self._rate_hz

    @sampling_rate_hz.setter
    def sampling_rate_hz(self, rate_in_hz_to_set):
        if not rate_in_hz_to_set:
            raise ValueError(f'Rate in Hz must be > 0, got {rate_in_hz_to_set}')

        rate_in_ms = int((1 / rate_in_hz_to_set) * 1e3)
        with self._lock_dev:
            self._write(_UbxMessage('06', '08', msg_type="tx", rate=rate_in_ms, timeRef=0).msg)
            now, answer = time.time_ns(), None
            while (time.time_ns() - now) < 5 * 1e9:
                if self._dev.in_waiting > 0:
                    ubx_class = binascii.hexlify(self._dev.read()).decode('utf-8')
                    ubx_id = binascii.hexlify(self._dev.read()).decode('utf-8')
                    answer = _UbxMessage(ubx_class, ubx_id, dev=self._dev, version=3)
                    if answer is not None and hasattr(answer, 'ubx_class') and hasattr(answer, 'ubx_id'):
                        if answer.ubx_class == '05' and answer.ubx_id == '01':
                            self._rate_hz = rate_in_hz_to_set
                            return
                        elif answer.ubx_class == '05' and answer.ubx_id == '00':
                            break
            raise RuntimeError(f'Failed to set sampling rate in Hz to {rate_in_hz_to_set}.')

    @property
    def position(self) -> GpsData:
        with self._lock_coord:
            return self._coordinates
