from fractions import Fraction
import vapoursynth as vs
from pathlib import Path
from functools import lru_cache, reduce as funcreduce
from typing import Callable, List, Union, Optional, Tuple


from ..dataclasses import (
    DGIndexFileInfo, DGIndexFooter,
    DGIndexHeader, DGIndexFrameData, IndexFileVideo
)

from .DVDIndexer import DVDIndexer
from .utils import opt_int, opt_ints


core = vs.core


class DGIndexNV(DVDIndexer):
    """Built-in DGIndexNV indexer"""

    def __init__(
        self, path: Union[Path, str] = 'DGIndexNV',
        vps_indexer: Optional[Callable[..., vs.VideoNode]] = None, ext: str = '.dgi'
    ) -> None:
        super().__init__(path, vps_indexer or core.dgdecodenv.DGSource, ext)

    def get_cmd(self, files: List[Path], output: Path) -> List[str]:
        return list(map(str, [self._check_path(), '-i', ','.join(map(str, files)), '-o', output, '-h']))

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

        # with open(index_path, 'w') as file:
        #     file.write(file_content)

    @lru_cache
    def get_info(self, index_path: Path, file_idx: int = 0) -> DGIndexFileInfo:
        with index_path.open(mode="r", encoding="utf8") as f:
            file_content = f.read()

        lines = file_content.split('\n')

        head, lines = self.__split_lines(lines)

        if "DGIndexNV" not in head[0]:
            self.file_corrupted(index_path)

        vid_lines, lines = self.__split_lines(lines)
        raw_header, lines = self.__split_lines(lines)

        print(raw_header)

        header = DGIndexHeader(
            0, [0, 0, 0, 0, 0],
            (1, 0),
            [0, 0, 4480309247, 0],
            8, Fraction(10, 11),
            (6, 6, 6),
            192, 4113
        )

        videos = [
            IndexFileVideo(Path(' '.join(line[:-1])), int(line[-1]))
            for line in map(lambda a: a.split(' '), vid_lines)
        ]

        max_sector = funcreduce(lambda a, b: a + b, [v.size for v in videos[:file_idx + 1]], 0)

        idx_file_sector = [max_sector - videos[file_idx].size, max_sector]

        curr_SEQ, frame_data = 0, []

        for rawline in lines:
            if len(rawline) == 0:
                break

            line: List[Optional[str]] = [*rawline.split(" ", maxsplit=6), *([None] * 6)]

            name = str(line[0])

            if name == 'SEQ':
                curr_SEQ = opt_int(line[1]) or 0

            if curr_SEQ < idx_file_sector[0]:
                continue
            elif curr_SEQ > idx_file_sector[1]:
                break

            try:
                int(name.split(':')[0])
            except ValueError:
                continue

            frame_data.append(DGIndexFrameData(
                int(line[2] or 0) + 2, str(line[1]), *opt_ints(line[4:6])
            ))

        footer = DGIndexFooter()

        for rlin in lines[-10:]:
            if split_val := rlin.rstrip().split(' '):
                values = [split_val[0], ' '.join(split_val[1:])]
            else:
                continue

            for key in footer.__dict__.keys():
                if key.split('_')[-1].upper() in values:
                    if key == 'film':
                        try:
                            value = [float(v.replace('%', '')) for v in values if '%' in v][0]
                        except IndexError:
                            value = 0
                    else:
                        value = int(values[1])

                    footer.__setattr__(key, value)

        return DGIndexFileInfo(index_path, file_idx, videos, header, frame_data, footer)

    @staticmethod
    def __split_lines(buff: List[str]) -> Tuple[List[str], List[str]]:
        split_idx = buff.index('')
        return buff[:split_idx], buff[split_idx + 1:]
