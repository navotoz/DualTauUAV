from enum import IntEnum, unique


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
