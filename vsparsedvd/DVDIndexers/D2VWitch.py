from __future__ import annotations

import re
import tempfile
import vapoursynth as vs
from fractions import Fraction
from functools import lru_cache
from typing import Any, List, Literal


from ..utils.spathlib import SPath
from ..utils.types import SPathLike
from ..dataclasses import D2VIndexFileInfo, D2VIndexFrameData, D2VIndexHeader, IndexFileVideo

from .DVDIndexer import DVDIndexer


core = vs.core


class D2VWitch(DVDIndexer):
    """Built-in d2vwitch indexer"""

    frame_lengths_key = "FilesFrameLengths"

    def __init__(self, **kwargs: Any) -> None:
        if 'bin_path' not in kwargs:
            kwargs['bin_path'] = 'd2vwitch'
        if 'vps_indexer' not in kwargs:
            kwargs['vps_indexer'] = core.d2v.Source
        if 'ext' not in kwargs:
            kwargs['ext'] = 'd2v'
        super().__init__(**kwargs)

    def get_cmd(self, files: List[SPath], output: SPath) -> List[str]:
        return list(map(str, [self._get_bin_path(), *files, '--output', output]))

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

    def write_idx_file_videoslength(self, index_path: SPath) -> List[int]:
        with open(index_path, 'r') as f:
            file_content = f.read()

        lines = file_content.split('\n')

        prev_lines, lines = lines[:2], lines[2:]

        if "DGIndex" not in prev_lines[0]:
            self.file_corrupted(index_path)

        vid_lines, lines = self._split_lines(lines)
        prev_lines += vid_lines + ['']

        vids_frame_lenghts = []

        for path in map(SPath, vid_lines):
            temp_idx_files = self.index([path], True, False, tempfile.gettempdir(), False)[0].to_str()

            if path.to_str().lower().endswith('_0.vob'):
                with open(temp_idx_files, 'r') as f:
                    idx_file_content = f.read()

                if len(idx_file_content.splitlines()) < 20:
                    vids_frame_lenghts += [1]
                    continue

            vid_file = self.vps_indexer(temp_idx_files)

            vids_frame_lenghts += [vid_file.num_frames]

        raw_header, lines = self._split_lines(lines)

        raw_header = [line for line in raw_header if self.frame_lengths_key not in line]

        raw_header += [f"{self.frame_lengths_key}={','.join(map(str, vids_frame_lenghts))}", '']

        with open(index_path, 'w') as file:
            file.write('\n'.join(prev_lines + raw_header + lines))

        return vids_frame_lenghts

    @lru_cache
    def get_info(self, index_path: SPath, file_idx: int = -1) -> D2VIndexFileInfo:
        with open(index_path, 'r') as f:
            file_content = f.read()

        lines = file_content.split('\n')

        head, lines = lines[:2], lines[2:]

        if "DGIndex" not in head[0]:
            self.file_corrupted(index_path)

        vid_lines, lines = self._split_lines(lines)
        raw_header, lines = self._split_lines(lines)

        video_frame_lenghts = []

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
            elif key == self.frame_lengths_key.upper():
                video_frame_lenghts = list(map(int, values))

        if len(video_frame_lenghts) != len(vid_lines):
            video_frame_lenghts = self.write_idx_file_videoslength(index_path)

        videos = [
            IndexFileVideo(path, path.stat().st_size, vidlen)
            for path, vidlen in zip(map(SPath, vid_lines), video_frame_lenghts)
        ]

        frame_data = []

        if file_idx >= 0:
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

    def index(
        self, files: List[SPath], force: bool = False, split_files: bool = False,
        output_folder: SPathLike | Literal[False] | None = None, single_input: bool = False, *cmd_args: str
    ) -> List[SPath]:
        return super().index(files, force, split_files, output_folder, single_input, *cmd_args, '--single-input')
