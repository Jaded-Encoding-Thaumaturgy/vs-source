from pathlib import Path
from fractions import Fraction
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class IFOFile:
    chapters: List[List[int]]
    fps: Fraction
    is_multiple_IFOs: bool


@dataclass
class IndexFileVideo:
    path: Path
    size: int


@dataclass
class IndexFileInfo:
    path: Path
    file_idx: int
    videos: List[IndexFileVideo]


@dataclass
class IndexFileFrameData:
    matrix: int
    pic_type: str
    vob: Optional[int]
    cell: Optional[int]


@dataclass
class D2VIndexHeader:
    pass


@dataclass
class D2VIndexFrameData(IndexFileFrameData):
    info: str
    skip: int
    position: int


@dataclass
class D2VIndexFooter:
    pass


@dataclass
class DGIndexHeader:
    device: int
    decode_modes: List[int]
    stream: List[int]
    ranges: List[int]
    depth: int
    aspect: List[int]
    colorimetry: Tuple[int, int, int]
    packet_size: int
    vpid: int


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
class D2VIndexFileInfo(IndexFileInfo):
    header: D2VIndexHeader
    frame_data: List[D2VIndexFrameData]
    footer: D2VIndexFooter


@dataclass
class DGIndexFileInfo(IndexFileInfo):
    header: DGIndexHeader
    frame_data: List[DGIndexFrameData]
    footer: DGIndexFooter
