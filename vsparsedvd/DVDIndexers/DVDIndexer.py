from __future__ import annotations

import re
import shutil
import tempfile
import subprocess
from hashlib import md5
import vapoursynth as vs
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple, Sequence, Literal


from ..utils.spathlib import SPath
from ..utils.types import SPathLike
from ..dataclasses import IndexFileType


core = vs.core


class DVDIndexer(ABC):
    """Abstract DVD indexer interface."""
    index_folder_name = '.vsparsedvd'

    def __init__(
        self, bin_path: SPathLike, vps_indexer: Callable[..., vs.VideoNode],
        ext: str, force: bool = True, **indexer_kwargs: Any
    ) -> None:
        self.bin_path = SPath(bin_path)
        self.vps_indexer = vps_indexer
        self.ext = ext
        self.force = force
        self.indexer_kwargs = indexer_kwargs
        super().__init__()

    @abstractmethod
    def get_cmd(self, files: List[SPath], output: SPath) -> List[str]:
        """Returns the indexer command"""
        raise NotImplementedError

    @abstractmethod
    def get_info(self, index_path: SPath, file_idx: int = 0) -> IndexFileType:
        """Returns info about the indexing file"""
        raise NotImplementedError

    @abstractmethod
    def update_video_filenames(self, index_path: SPath, filepaths: List[SPath]) -> None:
        raise NotImplementedError

    def _get_bin_path(self) -> SPath:
        if not (bin_path := shutil.which(str(self.bin_path))):
            raise FileNotFoundError(f'DVDIndexer: `{self.bin_path}` was not found!')
        return SPath(bin_path)

    def _run_index(self, files: List[SPath], output: SPath, cmd_args: Sequence[str]) -> None:
        output.mkdirp()

        status = subprocess.Popen(
            list(map(str, self.get_cmd(files, output))) + list(cmd_args),
            text=True, encoding='utf-8', shell=True, cwd=output.get_folder().to_str()
        ).wait()

        if status:
            raise RuntimeError(f"There was an error while running the {self.bin_path} command!")

    def get_out_folder(
        self, output_folder: SPathLike | Literal[False] | None = None, file: SPath | None = None
    ) -> SPath:
        if output_folder is None:
            return SPath(file).get_folder() if file else self.get_out_folder(False)
        elif not output_folder:
            return SPath(tempfile.gettempdir())

        return SPath(output_folder)

    def index(
        self, files: List[SPath], force: bool = False, split_files: bool = False,
        output_folder: SPathLike | Literal[False] | None = None, single_input: bool = False, *cmd_args: str
    ) -> List[SPath]:
        if len(unique_folders := list(set([f.get_folder().to_str() for f in files]))) > 1:
            return [
                c for s in (
                    self.index([
                        f for f in files if f.get_folder().to_str() == folder
                    ], force, split_files, output_folder, single_input)
                    for folder in unique_folders
                ) for c in s
            ]

        source_folder = files[0].get_folder()
        dest_folder = self.get_out_folder(output_folder, files[0])

        if single_input:
            for file in list(files):
                if matches := re.search(r"VTS_([0-9]{2})_([0-9])\.VOB", file.name, re.IGNORECASE):
                    files += source_folder.glob(
                        f'[vV][tT][sS]_[{matches[1][0]}-9][{matches[1][1]}-9]_[{matches[2]}-9].[vV][oO][bB]'
                    )

        files = list(sorted(set(files)))

        hash_str = self.get_videos_hash(files)

        def _index(files: List[SPath], output: SPath) -> None:
            if output.is_file():
                if output.stat().st_size == 0 or force:
                    output.unlink()
                    return self._run_index(files, output, cmd_args)

                return self.update_video_filenames(output, files)
            return self._run_index(files, output, cmd_args)

        if not split_files:
            output = self.get_video_idx_path(dest_folder, hash_str, 'JOINED' if len(files) > 1 else 'SINGLE')
            _index(files, output)
            return [output]

        outputs = [self.get_video_idx_path(dest_folder, hash_str, file.name) for file in files]

        for file, output in zip(files, outputs):
            _index([file], output)

        return outputs

    def get_idx_file_path(self, path: SPath) -> SPath:
        return path.with_suffix(f'.{self.ext}')

    def file_corrupted(self, index_path: SPath) -> None:
        if self.force:
            try:
                index_path.unlink()
            except OSError:
                raise RuntimeError("IsoFile: Index file corrupted, tried to delete it and failed.")
        else:
            raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

    def get_video_idx_path(self, folder: SPath, file_hash: str, video_name: SPathLike) -> SPath:
        vid_name = SPath(video_name).stem
        filename = '_'.join([file_hash, vid_name])
        return self.get_idx_file_path(folder / self.index_folder_name / filename)

    @staticmethod
    def _split_lines(buff: List[str]) -> Tuple[List[str], List[str]]:
        return buff[:(split_idx := buff.index(''))], buff[split_idx + 1:]

    @staticmethod
    def get_joined_names(files: List[SPath]) -> str:
        return '_'.join([file.name for file in files])

    @staticmethod
    def get_videos_hash(files: List[SPath]) -> str:
        lenght = sum(file.stat().st_size for file in files)
        to_hash = lenght.to_bytes(32, 'little') + DVDIndexer.get_joined_names(files).encode()
        return md5(to_hash).hexdigest()

    def source(
        self, file: str | SPath, force: bool = False, output_folder: SPathLike | Literal[False] | None = None,
        single_input: bool = True, *indexer_args: str, **vps_indexer_kwargs: Any
    ) -> vs.VideoNode:
        return core.std.Splice([
            self.vps_indexer(idx_filename.to_str(), **vps_indexer_kwargs)
            for idx_filename in self.index(
                [SPath(file)], force, False, output_folder, single_input, *indexer_args
            )
        ])
