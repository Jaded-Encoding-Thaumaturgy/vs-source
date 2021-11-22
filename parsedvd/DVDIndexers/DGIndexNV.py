from pathlib import Path
import vapoursynth as vs
from functools import lru_cache, reduce as funcreduce
from typing import Any, Callable, List, Union, Optional, Tuple

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

        # with open(index_path, 'w') as file:
        #     file.write(file_content)

    @lru_cache
    def get_info(self, index_path: Path, file_idx: int = 0) -> IndexFileInfo:
        with index_path.open(mode="r", encoding="utf8") as f:
            file_content = f.read()

        lines = file_content.split('\n')

        head, lines = self.__split_lines(lines)

        if "DGIndexNV" not in head[0]:
            self.file_corrupted(index_path)

        vid_lines, lines = self.__split_lines(lines)

        videos = [
            IndexFileVideo(Path(' '.join(line[:-1])), int(line[-1]))
            for line in map(lambda a: a.split(' '), vid_lines)
        ]

        _, lines = self.__split_lines(lines)

        idx_file_sector = [0, funcreduce(lambda a, b: a + b, [v.size for v in videos[:file_idx + 1]], 0)]

        idx_file_sector[0] = idx_file_sector[1] - videos[file_idx].size

        curr_SEQ = 0

        data = []

        for rawline in lines:
            if len(rawline) == 0:
                break

            line: List[Optional[str]] = rawline.split(" ", maxsplit=6) + ([None] * 6)  # type: ignore

            name = str(line[0])

            def getint(idx: int) -> Optional[int]:
                item = line[idx]
                return None if item is None else int(item)

            if name == 'SEQ':
                curr_SEQ = getint(1) or 0

            if curr_SEQ < idx_file_sector[0]:
                continue
            elif curr_SEQ > idx_file_sector[1]:
                break

            try:
                int(name.split(':')[0])
            except ValueError:
                continue

            matrix = getint(2)

            if matrix is not None:
                matrix += 2

            data.append(
                IndexFileData(
                    info=None, matrix=matrix,
                    vob=getint(4), cell=getint(5),
                    position=None, skip=0, pic_type=line[1]
                )
            )

        vinfo_dict = {"film": 0.0, "frames_coded": 0, "frames_playback": 0, "order": 0}

        for rlin in lines[-10:]:
            values = rlin.rstrip().split(' ')

            for key in vinfo_dict.keys():
                if key.split('_')[-1].upper() in values:

                    if key == 'film':
                        value = self.__getfilmval(values[0]) if '%' in values[0] else self.__getfilmval(values[1])
                    else:
                        value = int(values[1])

                    vinfo_dict[key] = value

        return IndexFileInfo(
            index_path, videos, data, file_idx,
            IndexFileVideoInfo(**vinfo_dict)  # type: ignore
        )

    @staticmethod
    def __split_lines(buff: List[str]) -> Tuple[List[str], List[str]]:
        split_idx = buff.index('')
        return buff[:split_idx], buff[split_idx + 1:]

    @staticmethod
    def __getfilmval(val: str) -> float:
        return float(val.replace('%', ''))
