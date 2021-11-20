import shutil
import subprocess
from pathlib import Path
import vapoursynth as vs
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Union

from ..dataclasses import IndexFileInfo

core = vs.core

# Contains portion of code from
# https://github.com/Varde-s-Forks/lvsfunc/blob/patches/source/lvsfunc/source.py
# Will be replaced with vardefunc's import when it's going to be available in it


class DVDIndexer(ABC):
    """Abstract DVD indexer interface."""

    def __init__(
        self, path: Union[Path, str], vps_indexer: Callable[..., vs.VideoNode], ext: str, force: bool = True
    ) -> None:
        self.path = Path(path)
        self.vps_indexer = vps_indexer
        self.ext = ext
        self.force = force
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
    def update_idx_file(self, index_path: Path, filepaths: List[Path]) -> None:
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
