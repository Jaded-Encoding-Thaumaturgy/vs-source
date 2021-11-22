from pathlib import Path
from fractions import Fraction
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class IFOFileInfo:
    chapters: List[List[int]]
    fps: Fraction
    is_multiple_IFOs: bool


@dataclass
class IndexFileVideo:
    path: Path
    size: int


@dataclass
class IndexFileFrameData:
    matrix: int
    pic_type: str
    vob: Optional[int]
    cell: Optional[int]


@dataclass
class __IndexFileInfoBase:
    path: Path
    file_idx: int
    videos: List[IndexFileVideo]


@dataclass
class IndexFileInfo(__IndexFileInfoBase):
    frame_data: List[IndexFileFrameData]


@dataclass
class D2VIndexHeader:
    pass


@dataclass
class D2VIndexFrameData(IndexFileFrameData):
    info: str
    skip: int
    position: int


@dataclass
class DGIndexHeader:
    device: int = 0
    decode_modes: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    stream: Tuple[int, ...] = (1, 0)
    ranges: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    depth: int = 8
    aspect: Fraction = Fraction(16, 9)
    colorimetry: Tuple[int, ...] = (2, 2, 2)
    packet_size: Optional[int] = None
    vpid: Optional[int] = None


@dataclass
class DGIndexFrameData(IndexFileFrameData):
    pass


@dataclass
class DGIndexFooter:
    film: float = 0.0
    frames_coded: int = 0
    frames_playback: int = 0
    order: int = 0


@dataclass
class D2VIndexFileInfo(__IndexFileInfoBase):
    header: D2VIndexHeader
    frame_data: List[D2VIndexFrameData]


@dataclass
class DGIndexFileInfo(__IndexFileInfoBase):
    header: DGIndexHeader
    frame_data: List[DGIndexFrameData]
    footer: DGIndexFooter


IndexFileType = Union[D2VIndexFileInfo, DGIndexFileInfo]
