class Code:
    def __init__(self, code: int = 0, cmd_bytes: int = 0, reply_bytes: int = 0) -> None:
        self.__code = code
        self.cmd_bytes = cmd_bytes
        self.reply_bytes = reply_bytes

    def __repr__(self) -> str:
        return f"Code(decimal={self.__code}, hex=0x{self.__code:x})"

    @property
    def code(self) -> int:
        return self.__code


# Tau Status codes
CAM_OK = 0x00
CAM_NOT_READY = 0x02
CAM_RANGE_ERROR = 0x03
CAM_UNDEFINED_ERROR = 0x04
CAM_UNDEFINED_PROCESS_ERROR = 0x05
CAM_UNDEFINED_FUNCTION_ERROR = 0x06
CAM_TIMEOUT_ERROR = 0x07
CAM_BYTE_COUNT_ERROR = 0x09
CAM_FEATURE_NOT_ENABLED = 0x0A
ARGUMENT_FPA = 0x00
ARGUMENT_HOUSING = 0x0A
TIMEOUT_IN_NSEC = 1e9  # 1 seconds
MAGIC_IMAGE_ENDING = 0x0ff0

# General Commands
NO_OP = Code(0x00, 0, 0)
SET_DEFAULTS = Code(0x01, 0, 0)
CAMERA_RESET = Code(0x02, 0, 0)
RESTORE_FACTORY_DEFAULTS = Code(0x03, 0, 0)
SERIAL_NUMBER = Code(0x04, 0, 8)
GET_REVISION = Code(0x05, 0, 8)

# Gain Commands
GET_GAIN_MODE = Code(0x0A, 0, 2)
GAIN_CODE_DICT = dict(auto=0x0000, low=0x0001, high=0x0002, manual=0x0003)
SET_GAIN_MODE = Code(0x0A, 2, 2)

# FFC Commands
GET_FFC_MODE = Code(0x0B, 0, 2)
SET_FFC_MODE = Code(0x0B, 2, 2)
FFC_MODE_CODE_DICT = dict(auto=0x0001, manual=0x0000, external=0x0002)

GET_FFC_NFRAMES = Code(0x0B, 4, 2)
SET_FFC_NFRAMES = Code(0x0B, 4, 2)

DO_FFC = Code(0x0C, 2, 2)
FFC_SHORT = b'\x00\x00'
FFC_LONG = b'\x00\x01'

GET_FFC_PERIOD = Code(0x0D, 0, 4)
SET_FFC_PERIOD_LOW_GAIN = Code(0x0D, 4, 4)
SET_FFC_PERIOD_HIGH_GAIN = Code(0x0D, 4, 4)
SET_FFC_PERIOD = Code(0x0D, 2, 2)

GET_FFC_TEMP_DELTA = Code(0x0E, 0, 4)
SET_FFC_TEMP_DELTA_LOW_GAIN = Code(0x0E, 2, 2)
SET_FFC_TEMP_DELTA_HIGH_GAIN = Code(0x0E, 2, 2)
SET_FFC_TEMP_DELTA = Code(0x0E, 4, 4)

# Video Mode Commands
GET_VIDEO_MODE = Code(0x0F, 0, 2)
SET_VIDEO_MODE = Code(0x0F, 2, 2)

GET_VIDEO_SYMBOLOGY_DIGITAL = Code(0x0F, 4, 2)
SET_VIDEO_SYMBOLOGY_DIGITAL = Code(0x0F, 4, 4)
GET_VIDEO_SYMBOLOGY_ANALOG = Code(0x0F, 4, 2)
SET_VIDEO_SYMBOLOGY_ANALOG = Code(0x0F, 4, 4)

GET_VIDEO_PALETTE = Code(0x10, 0, 2)
SET_VIDEO_PALETTE = Code(0x10, 2, 2)

GET_VIDEO_ORIENTATION = Code(0x11, 0, 2)
SET_VIDEO_ORIENTATION = Code(0x11, 2, 2)

# AGC
SET_CONTRAST = Code(0x14, 2, 2)
GET_CONTRAST = Code(0x14, 0, 2)

SET_BRIGHTNESS = Code(0x15, 2, 2)
GET_BRIGHTNESS = Code(0x15, 0, 2)

SET_BRIGHTNESS_BIAS = Code(0x18, 2, 2)
GET_BRIGHTNESS_BIAS = Code(0x18, 0, 2)

SET_AGC_TAIL_SIZE = Code(0x1B, 0, 2)
GET_AGC_TAIL_SIZE = Code(0x1B, 2, 2)

SET_AGC_ACE_CORRECT = Code(0x1C, 2, 0)
GET_AGC_ACE_CORRECT = Code(0x1C, 0, 2)

AGC_CODE_DICT = dict(plateau=0x0000, once_bright=0x0001, auto_bright=0x0002, manual=0x0003, linear=0x0005,
                     information_based=0x0009, information_based_eq=0x000A)
GET_AGC_ALGORITHM = Code(0x13, 0, 2)
SET_AGC_ALGORITHM = Code(0x13, 2, 2)

GET_AGC_THRESHOLD = Code(0x13, 2, 2)
SET_AGC_THRESHOLD = Code(0x13, 4, 0)

GET_AGC_OPTIMISATION_PERCENT = Code(0x13, 2, 2)
SET_AGC_OPTIMISATION_PERCENT = Code(0x13, 4, 0)

GET_CORRECTION_MASK = Code(0xB1, 0, 2)
SET_CORRECTION_MASK = Code(0xB1, 2, 2)
CORRECTION_MASK_DICT = dict(off=0)

# Lens

GET_LENS_NUMBER = Code(0x1E, 0, 2)
SET_LENS_NUMBER = Code(0x1E, 2, 2)

GET_LENS_GAIN_SWITCH = Code(0x1E, 2, 2)
SET_LENS_GAIN_SWITCH = Code(0x1E, 4, 4)

GET_LENS_GAIN_MAPPING = Code(0x1E, 2, 2)
SET_LENS_GAIN_MAPPING = Code(0x1E, 4, 4)

# Spot Meter

SET_SPOT_METER_MODE = Code(0x1F, 0, 2)
GET_SPOT_METER_MODE = Code(0x1F, 2, 2)

# Onboard sensors

READ_SENSOR_TEMPERATURE = Code(0x20, 2, 2)
READ_SENSOR_ACCELEROMETER = Code(0x20, 2, 8)
READ_SENSOR_STATUS = Code(0x20, 2, 2)

# Sync

GET_EXTERNAL_SYNC = Code(0x21, 0, 2)
SET_EXTERNAL_SYNC = Code(0x21, 2, 2)

# Isotherm

GET_ISOTHERM = Code(0x22, 0, 2)
SET_ISOTHERM = Code(0x22, 2, 2)

GET_ISOTHERM_THRESHOLD = Code(0x23, 0, 6)
SET_ISOTHERM_THRESHOLD = Code(0x23, 6, 6)

GET_ISOTHERM_THRESHOLD_FOUR = Code(0x23, 4, 4)
SET_ISOTHERM_THRESHOLD_FOUR = Code(0x23, 4, 4)

# Test Pattern

GET_TEST_PATTERN = Code(0x25, 0, 2)
SET_TEST_PATTERN = Code(0x25, 2, 2)

SET_VIDEO_COLOR_MODE = Code(0x26, 0, 2)
GET_VIDEO_COLOR_MODE = Code(0x26, 2, 2)

# Shutter
GET_SHUTTER_POSITION = Code(0x79, 0, 2)
SET_SHUTTER_POSITION = Code(0x79, 2, 2)

GET_SHUTTER_TEMPERATURE = Code(0x4D, 0, 2)
SET_SHUTTER_TEMPERATURE = Code(0x4D, 2, 0)

# Frame transfer

TRANSFER_FRAME = Code(0x82, 4, 4)

GET_MEMORY_ADDRESS = Code(0xD6, 4, 8)
GET_NV_MEMORY_SIZE = Code(0xD5, 2, 8)

READ_MEMORY_256 = Code(0xD2, 6, 256)

ERASE_BLOCK = Code(0xD4, 2, 2)

MEMORY_STATUS = Code(0xC4, 0, 2)

# Radiometry

GET_PLANCK_COEFFICIENTS = Code(0xB9, 0, 16)
SET_PLANCK_COEFFICIENTS = Code(0xB9, 18, 18)

GET_SPATIAL_THRESHOLD = Code(0x2C, 0, 2)  # dde
SET_SPATIAL_THRESHOLD = Code(0x2C, 2, 2)  # dde

SET_TLINEAR_MODE = Code(0x8E, 4, 0)
GET_TLINEAR_MODE = Code(0x8E, 2, 2)

# digital output modes
GET_DIGITAL_OUTPUT_MODE = Code(0x12, 0, 2)
SET_DIGITAL_OUTPUT_MODE = Code(0x12, 2, 2)

GET_LVDS_MODE = Code(0x12, 2, 2)
SET_LVDS_MODE = Code(0x12, 2, 2)
GET_LVDS_DEPTH = Code(0x12, 2, 2)
SET_LVDS_DEPTH = Code(0x12, 2, 2)

GET_CMOS_DEPTH = Code(0x12, 2, 2)
SET_CMOS_DEPTH = Code(0x12, 2, 2)

GET_XP_MODE = Code(0x12, 2, 2)
SET_XP_MODE = Code(0x12, 2, 2)

# FPS
GET_FPS = Code(0x72, 0, 2)
FPS_CODE_DICT = {30: 0x0000, 60: 0x0004}
SET_FPS = Code(0x72, 2, 2)
