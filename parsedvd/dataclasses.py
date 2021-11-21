from pathlib import Path
from fractions import Fraction
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IFOInfo:
    chapters: List[List[int]]
    fps: Fraction
    is_multiple_IFOs: bool


@dataclass
class IndexFileData:
    info: Optional[str]
    matrix: Optional[int]
    position: Optional[int]
    skip: Optional[int]
    vob: Optional[int]
    cell: Optional[int]
    pic_type: Optional[str]


@dataclass
class IndexFileVideo:
    path: Path
    size: int


@dataclass
class IndexFileVideoInfo:
    film: float
    frames_coded: int
    frames_playback: int
    order: int


@dataclass
class IndexFileInfo:
    path: Path
    videos: List[IndexFileVideo]
    data: List[IndexFileData]
    file_idx: int
    video_info: Optional[IndexFileVideoInfo] = None
