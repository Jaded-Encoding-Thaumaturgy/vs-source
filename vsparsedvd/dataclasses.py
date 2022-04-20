from __future__ import annotations

from fractions import Fraction
from dataclasses import dataclass, field
from typing import List, Tuple, Union, NamedTuple

from .utils.types import Matrix
from .utils.spathlib import SPath


class _SetItemMeta:
    def __setitem__(self, key: str, value: float | int) -> None:
        return self.__setattr__(key, value)


class IFOFileInfo(NamedTuple):
    chapters: Matrix[int]
    fps: Fraction
    is_multiple_IFOs: bool


class IndexFileVideo(NamedTuple):
    path: SPath
    size: int
    num_frames: int


@dataclass
class IndexFileFrameData(_SetItemMeta):
    matrix: int
    pic_type: str
    vob: int | None
    cell: int | None


@dataclass
class _IndexFileInfoBase(_SetItemMeta):
    path: SPath
    file_idx: int
    videos: List[IndexFileVideo]


@dataclass
class IndexFileInfo(_IndexFileInfoBase):
    frame_data: List[IndexFileFrameData]


@dataclass
class D2VIndexHeader(_SetItemMeta):
    stream_type: int = 0
    MPEG_type: int = 0
    iDCT_algorithm: int = 0
    YUVRGB_scale: int = 1
    luminance_filter: Tuple[int, ...] = (0, 0)
    clipping: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    aspect: Fraction = Fraction(16, 9)
    pic_size: str = ''
    field_op: int = 0
    frame_rate: Fraction = Fraction(30000, 1001)
    location: List[int] = field(default_factory=lambda: [0, 0, 0, 0])


@dataclass
class D2VIndexFrameData(IndexFileFrameData):
    info: str
    skip: int
    position: int


@dataclass
class DGIndexHeader(_SetItemMeta):
    device: int = 0
    decode_modes: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    stream: Tuple[int, ...] = (1, 0)
    ranges: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    depth: int = 8
    aspect: Fraction = Fraction(16, 9)
    colorimetry: Tuple[int, ...] = (2, 2, 2)
    packet_size: int | None = None
    vpid: int | None = None


@dataclass
class DGIndexFrameData(IndexFileFrameData):
    pass


@dataclass
class DGIndexFooter(_SetItemMeta):
    film: float = 0.0
    frames_coded: int = 0
    frames_playback: int = 0
    order: int = 0


@dataclass
class D2VIndexFileInfo(_IndexFileInfoBase):
    header: D2VIndexHeader
    frame_data: List[D2VIndexFrameData]


@dataclass
class DGIndexFileInfo(_IndexFileInfoBase):
    header: DGIndexHeader
    frame_data: List[DGIndexFrameData]
    footer: DGIndexFooter


IndexFileType = Union[D2VIndexFileInfo, DGIndexFileInfo]
