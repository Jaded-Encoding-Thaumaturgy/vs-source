import re
import tempfile
import vapoursynth as vs
from fractions import Fraction
from functools import lru_cache
from typing import Callable, List, Union, Optional


from ..utils.spathlib import SPath
from ..dataclasses import D2VIndexFileInfo, D2VIndexFrameData, D2VIndexHeader, IndexFileVideo

from .DVDIndexer import DVDIndexer


core = vs.core


class D2VWitch(DVDIndexer):
    """Built-in d2vwitch indexer"""

    ffflength_key = "FirstFileFrameLength"

    def __init__(
        self, path: Union[SPath, str] = 'd2vwitch',
        vps_indexer: Optional[Callable[..., vs.VideoNode]] = None, ext: str = 'd2v'
    ) -> None:
        super().__init__(path, vps_indexer or core.d2v.Source, ext)

        return list(map(str, [self._check_path(), *files, '--output', output]))
    def get_cmd(self, files: List[SPath], output: SPath) -> List[str]:

    def update_video_filenames(self, index_path: SPath, filepaths: List[SPath]) -> None:
        with open(index_path, 'r') as file:
            file_content = file.read()

        lines = file_content.split('\n')

        str_filepaths = list(map(str, filepaths))

        if "DGIndex" not in lines[0]:
            self.file_corrupted(index_path)

        if not (n_files := int(lines[1])) or n_files != len(str_filepaths):
            self.file_corrupted(index_path)

        end_videos = lines.index('')

        if lines[2:end_videos] == str_filepaths:
            return

        lines[2:end_videos] = str_filepaths

        with open(index_path, 'w') as file:
            file.write('\n'.join(lines))

    def write_idx_file_ffflength(self, index_path: Path) -> int:
        with index_path.open(mode="r", encoding="utf8") as f:
            file_content = f.read()

        lines = file_content.split('\n')

        prev_lines, lines = lines[:2], lines[2:]

        if "DGIndex" not in prev_lines[0]:
            self.file_corrupted(index_path)

        vid_lines, lines = self._split_lines(lines)
        prev_lines += vid_lines + ['']

        if path := Path(vid_lines[0]):
            video = IndexFileVideo(path, path.stat().st_size)

        temp_idx_file = Path(tempfile.gettempdir()) / f'{video.path.name}_{video.size}.{self.ext}'

        self.index([video.path], temp_idx_file, '--single-input')

        first_file = self.vps_indexer(temp_idx_file)

        ffflength = first_file.num_frames

        raw_header, lines = self._split_lines(lines)

        raw_header = [line for line in raw_header if self.ffflength_key not in line]

        raw_header += [f"{self.ffflength_key}={ffflength}", '']

        with open(index_path, 'w') as file:
            file.write('\n'.join(prev_lines + raw_header + lines))

        return ffflength

    @lru_cache
    def get_info(self, index_path: Path, file_idx: int = 0) -> D2VIndexFileInfo:
        with index_path.open(mode="r", encoding="utf8") as f:
            file_content = f.read()

        lines = file_content.split('\n')

        head, lines = lines[:2], lines[2:]

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
            elif key == self.ffflength_key.upper():
                header.ffflength = int(values[0])

        if header.ffflength < 0:
            header.ffflength = self.write_idx_file_ffflength(index_path)

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
