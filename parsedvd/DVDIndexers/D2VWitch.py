from pathlib import Path
import vapoursynth as vs
from functools import lru_cache
from typing import Any, Callable, List, Union, Optional

from .DVDIndexer import DVDIndexer
from ..dataclasses import IndexFileInfo, IndexFileData, IndexFileVideo

core = vs.core


class D2VWitch(DVDIndexer):
    """Built-in d2vwitch indexer"""

    def __init__(
        self, path: Union[Path, str] = 'd2vwitch',
        vps_indexer: Optional[Callable[..., vs.VideoNode]] = None, ext: str = '.d2v'
    ) -> None:
        vps_indexer = vps_indexer or core.d2v.Source
        super().__init__(path, vps_indexer, ext)

    def get_cmd(self, files: List[Path], output: Path) -> List[Any]:
        self._check_path()
        return [self.path, *files, '--output', output]

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
    def get_info(self, index_path: Path, file_idx: int = 0) -> IndexFileInfo:
        f = index_path.open(mode="r", encoding="utf8")

        f.readline().strip()
        video_paths = [Path(f.readline().strip()) for _ in range(int(f.readline().strip()))]
        videos = [IndexFileVideo(path, path.stat().st_size) for path in video_paths]

        if len(f.readline().strip()) > 0:
            self.file_corrupted(index_path)

        while True:
            if len(f.readline().strip()) == 0:
                break

        data = []
        while True:
            rawline = f.readline().strip()
            if len(rawline) == 0:
                break

            line = rawline.split(" ", maxsplit=7)

            ffile_idx = int(line[2])

            if ffile_idx < file_idx:
                continue
            elif ffile_idx > file_idx:
                break

            data.append(IndexFileData(
                info=bin(int(line[0], 16))[2:].zfill(8),
                matrix=int(line[1]), vob=int(line[5]),
                skip=int(line[4]), cell=int(line[6]),
                position=int(line[3]), pic_type='I'
            ))

        return IndexFileInfo(index_path, videos, data, file_idx)
