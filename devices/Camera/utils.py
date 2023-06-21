import binascii
import re
import struct
import threading as th
from typing import List, Tuple

import numpy as np
import usb
from pyftdi.ftdi import Ftdi

from devices.Camera import make_device_from_vid_pid
from devices.Camera.Tau import tau2_config as ptc
from devices.Camera.Tau.tau2_config import Code

BUFFER_SIZE = int(2 ** 23)  # 8 MBytes
TEAX_LEN = 4
UART_PREAMBLE_LENGTH = 6
REPLY_HEADER_BYTES = 10
BORDER_VALUE = 64


class BytesBuffer:
    def __init__(self, size_to_signal: int = 0) -> None:
        self._buffer: bytes = b''
        self._lock = th.RLock()
        self._size_to_signal = size_to_signal

    def clear_buffer(self) -> None:
        with self._lock:
            self._buffer = b''

    def sync_teax(self) -> bool:
        with self._lock:
            idx_sync = self._buffer.rfind(b'TEAX')
            if idx_sync != -1:
                self._buffer = self._buffer[idx_sync + TEAX_LEN:]
                return True
            return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __add__(self, other: bytes):
        with self._lock:
            self._buffer += other
            if len(self._buffer) > BUFFER_SIZE:
                self._buffer = self._buffer[-BUFFER_SIZE:]
            return self._buffer

    def __iadd__(self, other: bytes):
        with self._lock:
            self._buffer += other
            if len(self._buffer) > BUFFER_SIZE:
                self._buffer = self._buffer[-BUFFER_SIZE:]
            return self

    def __getitem__(self, item: slice) -> bytes:
        with self._lock:
            if isinstance(item, slice):
                return self._buffer[item]

    def __call__(self) -> bytes:
        with self._lock:
            return self._buffer

    @property
    def buffer(self) -> bytes:
        with self._lock:
            return self._buffer


def generate_subsets_indices_in_string(input_string: (BytesWarning, bytes, bytes, map, filter),
                                       subset: (bytes, str)) -> list:
    reg = re.compile(subset)
    if isinstance(input_string, BytesBuffer):
        return [i.start() for i in reg.finditer(input_string())]
    return [i.start() for i in reg.finditer(input_string)]


def generate_overlapping_list_chunks(generator: (map, filter), n: int):
    lst = list(generator)
    subset_generator = map(lambda idx: lst[idx:idx + n], range(len(lst)))
    return filter(lambda sub: len(sub) == n, subset_generator)


def get_crc(data) -> List[int]:
    crc = struct.pack(len(data) * 'B', *data)
    crc = binascii.crc_hqx(crc, 0)
    crc = [((crc & 0xFF00) >> 8).to_bytes(1, 'big'), (crc & 0x00FF).to_bytes(1, 'big')]
    return list(map(lambda x: int.from_bytes(x, 'big'), crc))


def connect_ftdi(vid, pid) -> Tuple[Ftdi, str]:
    device, address = make_device_from_vid_pid(vid, pid)

    usb.util.claim_interface(device, 0)
    usb.util.claim_interface(device, 1)

    ftdi = Ftdi()
    ftdi.open_from_device(device)

    ftdi.set_bitmode(0xFF, Ftdi.BitMode.RESET)
    ftdi.set_bitmode(0xFF, Ftdi.BitMode.SYNCFF)
    return ftdi, address


def is_8bit_image_borders_valid(raw_image_8bit: np.ndarray, height: int) -> bool:
    if raw_image_8bit is None:
        return False
    try:
        if np.nonzero(raw_image_8bit[:, 0] != 0)[0]:
            return False
    except ValueError:
        return False
    valid_idx = np.nonzero(raw_image_8bit[:, -1] != BORDER_VALUE)
    if len(valid_idx) != 1:
        return False
    valid_idx = int(valid_idx[0])
    if valid_idx != height - 1:  # the different value should be in the bottom of the border
        return False
    return True


def parse_incoming_message(buffer: bytes, command: Code) -> (List, None):
    len_in_bytes = command.reply_bytes + REPLY_HEADER_BYTES
    argument_length = len_in_bytes * UART_PREAMBLE_LENGTH

    idx_list = generate_subsets_indices_in_string(buffer, b'UART')
    if not idx_list:
        return None
    try:
        data = map(lambda idx: buffer[idx:idx + argument_length][5::6], idx_list)
        data = map(lambda d: d[0], data)
        data = generate_overlapping_list_chunks(data, len_in_bytes)
        data = filter(lambda res: len(res) >= len_in_bytes, data)  # length of message at least as expected
        data = filter(lambda res: res[0] == 110, data)  # header is 0x6E (110)
        data = list(filter(lambda res: res[3] == command.code, data))
    except IndexError:
        data = None
    if not data:
        return None
    data = data[-1]
    crc_1 = get_crc(data[:6])
    crc_2 = get_crc(data[8:8 + command.reply_bytes])
    if not crc_1 == data[6:8] or not crc_2 == data[-2:]:
        return None
    ret_value = data[8:8 + command.reply_bytes]
    ret_value = struct.pack('<' + len(ret_value) * 'B', *ret_value)
    return ret_value


def make_packet(command: ptc.Code, argument: (bytes, None) = None) -> bytes:
    if argument is None:
        argument = []

    # Refer to Tau 2 Software IDD
    # Packet Protocol (Table 3.2)
    packet_size = len(argument)
    assert (packet_size == command.cmd_bytes)

    process_code = int(0x6E).to_bytes(1, 'big')
    status = int(0x00).to_bytes(1, 'big')
    function = command.code.to_bytes(1, 'big')

    # First CRC is the first 6 bytes of the packet
    # 1 - Process code
    # 2 - Status code
    # 3 - Reserved
    # 4 - Function
    # 5 - N Bytes MSB
    # 6 - N Bytes LSB

    packet = [process_code,
              status,
              function,
              ((packet_size & 0xFF00) >> 8).to_bytes(1, 'big'),
              (packet_size & 0x00FF).to_bytes(1, 'big')]
    crc_1 = binascii.crc_hqx(struct.pack("ccxccc", *packet), 0)

    packet.append(((crc_1 & 0xFF00) >> 8).to_bytes(1, 'big'))
    packet.append((crc_1 & 0x00FF).to_bytes(1, 'big'))

    if packet_size > 0:

        # Second CRC is the CRC of the data (if any)
        crc_2 = binascii.crc_hqx(argument, 0)
        packet.append(argument)
        packet.append(((crc_2 & 0xFF00) >> 8).to_bytes(1, 'big'))
        packet.append((crc_2 & 0x00FF).to_bytes(1, 'big'))

        fmt = ">cxcccccc{}scc".format(packet_size)

    else:
        fmt = ">cxccccccxxx"

    data = struct.pack(fmt, *packet)
    return data
