from pathlib import Path
import vapoursynth as vs
from functools import lru_cache
from typing import Any, Callable, List, Union, Optional

from .DVDIndexer import DVDIndexer
from ..dataclasses import IndexFileInfo, IndexFileData, IndexFileVideo, IndexFileVideoInfo

core = vs.core


class DGIndexNV(DVDIndexer):
    """Built-in DGIndexNV indexer"""

    def __init__(
        self, path: Union[Path, str] = 'DGIndexNV',
        vps_indexer: Optional[Callable[..., vs.VideoNode]] = None, ext: str = '.dgi'
    ) -> None:
        vps_indexer = vps_indexer or core.dgdecodenv.DGSource
        super().__init__(path, vps_indexer, ext)

    def get_cmd(self, files: List[Path], output: Path) -> List[Any]:
        self._check_path()
        return [self.path, '-i', ','.join(map(str, files)), '-o', output, '-h']

    def update_idx_file(self, index_path: Path, filepaths: List[Path]) -> None:
        with open(index_path, 'r') as file:
            file_content = file.read()

        str_filepaths = list(map(str, filepaths))

        firstsplit_idx = file_content.index('\n\n')

        if "DGIndexNV" not in file_content[:firstsplit_idx]:
            self.file_corrupted(index_path)

        cut_content = file_content[firstsplit_idx + 2:]

        firstsplit = file_content[:firstsplit_idx].count('\n') + 2

        maxsplits = cut_content[:cut_content.index('\n\n')].count('\n') + firstsplit + 1

        content = file_content.split('\n', maxsplits)

        if maxsplits - firstsplit != len(str_filepaths):
            self.file_corrupted(index_path)

        splitted = [content[i].split(' ') for i in range(firstsplit, maxsplits)]

        if [split[0] for split in splitted] == str_filepaths:
            return

        content[firstsplit:maxsplits] = [
            f"{filepaths[i]} {splitted[i][1]}" for i in range(maxsplits - firstsplit)
        ]

        file_content = '\n'.join(content)

        with open(index_path, 'w') as file:
            file.write(file_content)

    @lru_cache
    def get_info(self, index_path: Path, file_idx: int = 0) -> IndexFileInfo:
        f = index_path.open(mode="r", encoding="utf8")

        file_content = f.read()

        firstsplit_idx = file_content.index('\n\n')

        if "DGIndexNV" not in file_content[:firstsplit_idx]:
            self.file_corrupted(index_path)

        cut_content = file_content[firstsplit_idx + 2:]

        firstsplit = file_content[:firstsplit_idx].count('\n') + 2

        maxsplits = cut_content[:cut_content.index('\n\n')].count('\n') + firstsplit + 1

        f.close()

        f = index_path.open(mode="r", encoding="utf8")

        for _ in range(3):
            f.readline().strip()

        videos = [
            IndexFileVideo(Path(' '.join(line[:-1])), int(line[-1]))
            for line in [
                f.readline().strip().split(' ') for _ in range(maxsplits - firstsplit)
            ]
        ]

        if len(f.readline().strip()) > 0:
            self.file_corrupted(index_path)

        while True:
            if len(f.readline().strip()) == 0:
                break

        idx_file_sector = [0, 0]

        for video in videos[:file_idx + 1]:
            idx_file_sector[1] += video.size

        idx_file_sector[0] = idx_file_sector[1] - videos[file_idx].size

        curr_SEQ = 0

        data = []
        while True:
            rawline = f.readline().strip()

            if len(rawline) == 0:
                break

            line = rawline.split(" ", maxsplit=6)

            if line[0] == 'SEQ':
                curr_SEQ = int(line[1])

            if curr_SEQ < idx_file_sector[0]:
                continue
            elif curr_SEQ > idx_file_sector[1]:
                break

            try:
                int(line[0].split(':')[0])
            except ValueError:
                continue

            data.append(IndexFileData(
                info=None, matrix=int(line[2]) + 2,
                vob=int(line[4]), cell=int(line[5]),
                position=None, skip=0, pic_type=line[1]
            ))

        f.close()

        vinfo_dict = {"film": 0.0, "frames_coded": 0, "frames_playback": 0, "order": 0}

        def getfilmval(val: str) -> float:
            return float(val.replace('%', ''))

        for rlin in list(reversed(cut_content[-120:].split('\n')))[:8]:
            values = rlin.rstrip().split(' ')

            for key in vinfo_dict.keys():
                if key.replace('frames_', '').upper() in values:

                    if key == 'film':
                        value = getfilmval(values[0]) if '%' in values[0] else getfilmval(values[1])
                    else:
                        value = int(values[1])

                    vinfo_dict[key] = value

        video_info = IndexFileVideoInfo(**vinfo_dict)  # type: ignore

        return IndexFileInfo(videos, data, file_idx, video_info)
