from struct import unpack
from typing import Any, List, Tuple
import io


PTS_FLAG = 0b0010
PTSDTS_PTS_FLAG = 0b0011
PTSDTS_DTS_FLAG = 0b0001

__all__ = [
    'PTS_FLAG',
    'PTSDTS_PTS_FLAG',
    'PTSDTS_DTS_FLAG',
    'unpack_byte',
    'parse_pts',
    'get_pes',
    'pes_payload',
    'get_pts',
    'get_start'
]

def unpack_byte(buf, n: int | List[int]) -> tuple[Any, ...]:
    stra = ">"

    if isinstance(n, int):
        n = [n]
    bytecnt = 0
    for a in n:
        bytecnt += a
        stra += {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}.get(a, 'B')

    assert len(buf) == bytecnt
    return unpack(stra, buf)

def parse_pts(flag,b):
    assert(((b[0] & 0b11110000) >> 4) == flag)
    assert(((b[0] & 0b00000001)) == 1)
    assert(((b[2] & 0b00000001)) == 1)
    assert(((b[4] & 0b00000001)) == 1)

    p0 = (b[0] & 0b1110) >> 1
    p1 = (b[1] << 8) + (b[2] & 0b11111110) >> 1
    p2 = (b[3] << 8) + (b[4] & 0b11111110) >> 1
    pts = p2 + (p1 << 15) + (p0 << 30)
    return pts

def get_pes(b: io.BytesIO):
    lenb = b.read(2) 
    length = unpack_byte(lenb,2)[0]
    assert length < 2048 - 4

    inner = b.read(length)
    return inner

def pes_payload(inner) -> bytes:
    hdr_len = inner[2]
    inner_data = inner[3 + hdr_len:]
    return inner_data

def get_pts(inner) -> Tuple[int | None, int | None]:
    pts_dts_ind = (inner[1] & 0b11000000) >> 6
    hdr_len = inner[2]
    inner_data = inner[3 + hdr_len:]

    pts = None
    dts = None

    if pts_dts_ind == 0b00:
        pass
    elif pts_dts_ind == 0b10:
        pts = parse_pts(PTS_FLAG, inner[3:3 + 5])
    elif  pts_dts_ind == 0b11:
        pts = parse_pts(PTSDTS_PTS_FLAG, inner[3:3 + 5])
        dts = parse_pts(PTSDTS_DTS_FLAG, inner[3+5:3 + 5+5])
    else:
        assert False
    return pts,dts

def get_start(b: io.BytesIO):
    b = b.read(4)
    assert (0,0,1) == tuple(b[0:3])
    return b[3]
