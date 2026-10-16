"""All spatial unit and height-reference conversions live here."""
import numpy as np

GROUND_ALTITUDE_M = 166.0
DEFAULT_POS_SCALE = 10.0
FRAME_HZ = 3.0


def altitude_to_height(altitude_m, ground_altitude_m=GROUND_ALTITUDE_M):
    return np.asarray(altitude_m) - ground_altitude_m


def to_internal(position_m, pos_scale=DEFAULT_POS_SCALE):
    if pos_scale <= 0:
        raise ValueError("pos_scale must be positive")
    return position_m / pos_scale


def to_metres(position, pos_scale=DEFAULT_POS_SCALE):
    if pos_scale <= 0:
        raise ValueError("pos_scale must be positive")
    return position * pos_scale
