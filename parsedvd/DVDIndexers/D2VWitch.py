import re
import vapoursynth as vs
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
from typing import Callable, List, Union, Optional


from ..dataclasses import D2VIndexFileInfo, D2VIndexFrameData, D2VIndexHeader, IndexFileVideo

from .DVDIndexer import DVDIndexer


core = vs.core


class D2VWitch(DVDIndexer):
    """Built-in d2vwitch indexer"""

    def __init__(
        self, path: Union[Path, str] = 'd2vwitch',
        vps_indexer: Optional[Callable[..., vs.VideoNode]] = None, ext: str = '.d2v'
    ) -> None:
        super().__init__(path, vps_indexer or core.d2v.Source, ext)

    def get_cmd(self, files: List[Path], output: Path) -> List[str]:
        return list(map(str, [self._check_path(), *files, '--output', output]))

    def update_idx_file(self, index_path: Path, filepaths: List[Path]) -> None:
        with open(index_path, 'r') as file:
            file_content = file.read()

        str_filepaths = [str(path) for path in filepaths]

        firstsplit_idx = file_content.index('\n\n')

        if "DGIndex" not in file_content[:firstsplit_idx]:
            self.file_corrupted(index_path)

        maxsplits = file_content[:firstsplit_idx].count('\n') + 1

        content = file_content.split('\n', maxsplits)

        n_files = int(content[1])

        if not n_files or n_files != len(str_filepaths):
            self.file_corrupted(index_path)

        if content[2:maxsplits] == str_filepaths:
            return

        content[2:maxsplits] = str_filepaths

        file_content = '\n'.join(content)

        with open(index_path, 'w') as file:
            file.write(file_content)

    @lru_cache
    def get_info(self, index_path: Path, file_idx: int = 0) -> D2VIndexFileInfo:
        with index_path.open(mode="r", encoding="utf8") as f:
            file_content = f.read()

        lines = file_content.split('\n')

        head, lines = lines[:3], lines[3:]

        if "DGIndex" not in head[0]:
            self.file_corrupted(index_path)

        vid_lines, lines = self._split_lines(lines)
        raw_header, lines = self._split_lines(lines)

        videos = [IndexFileVideo(path, path.stat().st_size) for path in map(Path, vid_lines)]

        header = D2VIndexHeader()

        for rlin in raw_header:
            if split_val := rlin.rstrip().split('='):
                key: str = split_val[0].upper()
                values: List[str] = ','.join(split_val[1:]).split(',')
            else:
                continue

            if key == 'STREAM_TYPE':
                header.stream_type = int(values[0])
            elif key == 'MPEG_TYPE':
                header.MPEG_type = int(values[0])
            elif key == 'IDCT_ALGORITHM':
                header.iDCT_algorithm = int(values[0])
            elif key == 'YUVRGB_SCALE':
                header.YUVRGB_scale = int(values[0])
            elif key == 'LUMINANCE_FILTER':
                header.luminance_filter = tuple(map(int, values))
            elif key == 'CLIPPING':
                header.clipping = list(map(int, values))
            elif key == 'ASPECT_RATIO':
                header.aspect = Fraction(*list(map(int, values[0].split(':'))))
            elif key == 'PICTURE_SIZE':
                header.pic_size = str(values[0])
            elif key == 'FIELD_OPERATION':
                header.field_op = int(values[0])
            elif key == 'FRAME_RATE':
                if matches := re.search(r".*\((\d+\/\d+)", values[0]):
                    header.frame_rate = Fraction(matches.group(1))
            elif key == 'LOCATION':
                header.location = list(map(int, values))

        frame_data = []

        for rawline in lines:
            if len(rawline) == 0:
                break

            line = rawline.split(" ", maxsplit=7)

            ffile_idx = int(line[2])

            if ffile_idx < file_idx:
                continue
            elif ffile_idx > file_idx:
                break

            frame_data.append(D2VIndexFrameData(
                int(line[1]), 'I', int(line[5]),
                int(line[6]), bin(int(line[0], 16))[2:].zfill(8),
                int(line[4]), int(line[3])
            ))

        return D2VIndexFileInfo(index_path, file_idx, videos, header, frame_data)
