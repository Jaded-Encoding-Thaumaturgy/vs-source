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
        vps_indexer = vps_indexer or core.d2v.Source  # type:ignore
        super().__init__(path, vps_indexer, ext)

    def get_cmd(self, files: List[Path], output: Path) -> List[Any]:
        self._check_path()
        return [self.path, *files, '--output', output]

    def update_idx_file(self, index_path: Path, filepaths: List[Path]):
        with open(index_path, 'r') as file:
            content = file.read()

        str_filepaths = [str(path) for path in filepaths]

        firstsplit_idx = content.index('\n\n')

        if "DGIndex" not in content[:firstsplit_idx]:
            raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

        maxsplits = content[:firstsplit_idx].count('\n') + 1

        content = content.split('\n', maxsplits)

        n_files = int(content[1])

        if not n_files or n_files != len(str_filepaths):
            raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

        if content[2:maxsplits] == str_filepaths:
            return

        content[2:maxsplits] = str_filepaths

        content = '\n'.join(content)

        with open(index_path, 'w') as file:
            file.write(content)

    @lru_cache
    def get_info(self, index_path: Path, file_idx: int = 0) -> IndexFileInfo:
        f = index_path.open(mode="r", encoding="utf8")

        f.readline().strip()
        videos = [Path(f.readline().strip()) for _ in range(int(f.readline().strip()))]
        videos = [IndexFileVideo(path, path.stat().st_size) for path in videos]

        if len(f.readline().strip()) > 0:
            raise ValueError("IsoFile: Index file corrupted! Delete it and retry.")

        while True:
            if len(f.readline().strip()) == 0:
                break

        data = []
        while True:
            line = f.readline().strip()
            if len(line) == 0:
                break

            line = line.split(" ", maxsplit=7)

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

        return IndexFileInfo(videos, data, file_idx)
