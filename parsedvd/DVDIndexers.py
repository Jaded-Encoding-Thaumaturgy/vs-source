import shutil
import subprocess
from pathlib import Path
import vapoursynth as vs
from functools import lru_cache
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Union, Optional

core = vs.core

# Contains portion of code from
# https://github.com/Varde-s-Forks/lvsfunc/blob/patches/source/lvsfunc/source.py
# Will be replaced with vardefunc's import when it's going to be available in it


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
class IndexFileInfo:
    videos: List[IndexFileVideo]
    data: List[IndexFileData]
    file_idx: int


class DVDIndexer(ABC):
    """Abstract DVD indexer interface."""

    def __init__(self, path: Union[Path, str], vps_indexer: Callable[..., vs.VideoNode], ext: str) -> None:
        self.path = Path(path)
        self.vps_indexer = vps_indexer
        self.ext = ext
        super().__init__()

    @abstractmethod
    def get_cmd(self, files: List[Path], output: Path) -> List[Any]:
        """Returns the indexer command"""
        raise NotImplementedError

    @abstractmethod
    def get_info(self, index_path: Path, file_idx: int = 0) -> IndexFileInfo:
        """Returns info about the indexing file"""
        raise NotImplementedError

    @abstractmethod
    def update_idx_file(self, index_path: Path, filepaths: List[Path]):
        raise NotImplementedError

    def _check_path(self) -> None:
        if not shutil.which(str(self.path)):
            raise FileNotFoundError(f'DVDIndexer: `{self.path}` was not found!')

    def index(self, files: List[Path], output: Path) -> None:
        subprocess.run(
            list(map(str, self.get_cmd(files, output))),
            check=True, text=True, encoding='utf-8',
            stdout=subprocess.PIPE, cwd=files[0].parent
        )

    def get_idx_file_path(self, path: Path) -> Path:
        return path.with_suffix(self.ext)


class D2VWitch(DVDIndexer):
    """Built-in d2vwitch indexer"""

    def __init__(
        self, path: Union[Path, str] = 'd2vwitch',
        vps_indexer: Callable[..., vs.VideoNode] = core.d2v.Source, ext: str = '.d2v'  # type:ignore
    ) -> None:
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

    def update_idx_file(self, index_path: Path, filepaths: List[Path]):
        with open(index_path, 'r') as file:
            content = file.read()

        str_filepaths = list(map(str, filepaths))

        firstsplit_idx = content.index('\n\n')

        if "DGIndexNV" not in content[:firstsplit_idx]:
            raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

        cut_content = content[firstsplit_idx + 2:]

        firstsplit = content[:firstsplit_idx].count('\n') + 2

        maxsplits = cut_content[:cut_content.index('\n\n')].count('\n') + firstsplit + 1

        content = content.split('\n', maxsplits)

        if maxsplits - firstsplit != len(str_filepaths):
            raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

        splitted = [content[i].split(' ') for i in range(firstsplit, maxsplits)]

        if [split[0] for split in splitted] == str_filepaths:
            return

        content[firstsplit:maxsplits] = [
            f"{filepaths[i]} {splitted[i][1]}" for i in range(maxsplits - firstsplit)
        ]

        content = '\n'.join(content)

        with open(index_path, 'w') as file:
            file.write(content)

    @lru_cache
    def get_info(self, index_path: Path, file_idx: int = 0) -> IndexFileInfo:
        f = index_path.open(mode="r", encoding="utf8")

        content = f.read()

        firstsplit_idx = content.index('\n\n')

        if "DGIndexNV" not in content[:firstsplit_idx]:
            raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

        cut_content = content[firstsplit_idx + 2:]

        firstsplit = content[:firstsplit_idx].count('\n') + 2

        maxsplits = cut_content[:cut_content.index('\n\n')].count('\n') + firstsplit + 1

        content = content.split('\n', maxsplits)

        f = index_path.open(mode="r", encoding="utf8")

        for _ in range(3):
            f.readline().strip()

        videos = [
            IndexFileVideo(Path(line[0]), int(line[1]))
            for line in [
                f.readline().strip().split(' ') for _ in range(maxsplits - firstsplit)
            ]
        ]

        if len(f.readline().strip()) > 0:
            raise ValueError("IsoFile: Index file corrupted! Delete it and retry.")

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
            line = f.readline().strip()

            if len(line) == 0:
                break

            line = line.split(" ", maxsplit=6)

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

        return IndexFileInfo(videos, data, file_idx)


class DGIndex(DGIndexNV):
    """Built-in dgindex indexer"""

    def __init__(
        self, path: Union[Path, str] = 'dgindex',
        vps_indexer: Optional[Callable[..., vs.VideoNode]] = None, ext: str = '.d2v'
    ) -> None:
        vps_indexer = vps_indexer or core.d2v.Source  # type:ignore
        super().__init__(path, vps_indexer, ext)
        print(RuntimeWarning("\n\tDGIndex is bugged, it will probably not work on your system/version.\n"))

    def get_cmd(
        self, files: List[Path], output: Path,
        idct_algo: int = 5, field_op: int = 2, yuv_to_rgb: int = 1
    ) -> List[Any]:
        self._check_path()

        filepaths = '[' + ','.join([f'"{str(path)}"' for path in files]) + ']'

        return [
            str(self.path), "-AIF", filepaths,
            "-IA", str(idct_algo), "-FO", str(field_op), "-YR", str(yuv_to_rgb),
            "-OM", "0", "-HIDE", "-EXIT", "-O", str(output)
        ]
