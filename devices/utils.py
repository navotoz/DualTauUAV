import binascii
import re
import struct
from typing import List, Optional, Tuple

import numpy as np
import usb
from pyftdi.ftdi import Ftdi
import usb.core

from devices.tau2_config import Code

_list_of_connected_camera = []


REPLY_HEADER_BYTES = 10
BORDER_VALUE = 64


def generate_subsets_indices_in_string(input_string: bytes) -> list:
    reg = re.compile(b'UART')
    return [i.start() for i in reg.finditer(input_string)]


def generate_overlapping_list_chunks(lst: list, n: int):
    subset_generator = map(lambda idx: lst[idx:idx + n], range(len(lst)))
    return filter(lambda sub: len(sub) == n, subset_generator)


def get_crc(data) -> List[int]:
    crc = struct.pack(len(data) * 'B', *data)
    crc = binascii.crc_hqx(crc, 0)
    crc = [((crc & 0xFF00) >> 8).to_bytes(1, 'big'), (crc & 0x00FF).to_bytes(1, 'big')]
    return list(map(lambda x: int.from_bytes(x, 'big'), crc))


def make_device_from_vid_pid(vid: int, pid: int) -> Tuple[usb.core.Device, str]:
    device = None
    devices = usb.core.find(idVendor=vid, idProduct=pid, find_all=True)
    for d in devices:
        if d.address in _list_of_connected_camera:
            continue
        device = d
        _list_of_connected_camera.append(d.address)
        break
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
    return device, d.address


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


def parse_incoming_message(buffer: bytes, command: Code) -> Optional[list]:
    len_in_bytes = command.reply_bytes + REPLY_HEADER_BYTES
    buffer = [p for p in buffer]
    if not buffer:
        return None

    try:
        data = generate_overlapping_list_chunks(buffer, len_in_bytes)
        data = filter(lambda res: len(res) >= len_in_bytes, data)  # length of message at least as expected
        data = filter(lambda res: res[0] == 110, data)  # header is 0x6E (110)
        data = list(filter(lambda res: res[3] == command.code, data))
    except IndexError:
        data = None
    if not data:
        return None
    for idx, d in enumerate(data):
        crc_1 = get_crc(d[:6])
        crc_2 = get_crc(d[8:8 + command.reply_bytes])
        if not crc_1 == d[6:8] or not crc_2 == d[-2:]:
            data[idx] = None
    if not data or all([p is None for p in data]):
        return None
    data = data[-1]
    ret_value = data[8:8 + command.reply_bytes]
    ret_value = struct.pack('<' + len(ret_value) * 'B', *ret_value)
    return ret_value


def make_packet(command: Code, argument: Optional[bytes] = None) -> bytes:
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
