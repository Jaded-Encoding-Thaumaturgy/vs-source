from dataclasses import dataclass

__all__ = ['a52_syncinfo']

RATE = [32, 40, 48, 56, 64, 80, 96, 112,
        128, 160, 192, 224, 256, 320, 384, 448,
        512, 576, 640]

HALFRATE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]


@dataclass
class A52SyncInfo:
    data_size: int
    sample_rate: int


def a52_syncinfo(buf: bytes) -> A52SyncInfo:
    assert buf[0] == 0x0B
    assert buf[1] == 0x77

    half = HALFRATE[buf[5] >> 3]

    frmsizecod = buf[4] & 63
    bitrate = RATE[frmsizecod >> 1]

    asd = buf[4] & 0xc0
    if asd == 0:
        sample_rate = 48000 >> half
        data_size = 4 * bitrate
    elif asd == 0x40:
        sample_rate = 44100 >> half
        data_size = 2 * (320 * bitrate // 147 + (frmsizecod & 1))
    elif asd == 0x80:
        sample_rate = 32000 >> half
        data_size = 6 * bitrate
    else:
        assert False

    return A52SyncInfo(data_size, sample_rate)
