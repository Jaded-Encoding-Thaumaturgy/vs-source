from __future__ import annotations
from os import path
from typing import Any

__all__ = ['SPath']

from pathlib import Path


class SPath(Path):
    """Modified version of pathlib.Path"""
    _flavour = type(Path())._flavour  # type: ignore

    def format(self, *args: Any, **kwargs: Any) -> SPath:
        return SPath(self.to_str().format(*args, **kwargs))

    def to_str(self) -> str:
        return str(self)

    def get_folder(self) -> SPath:
        folder_path = self.resolve()

        if folder_path.is_dir():
            return folder_path

        return SPath(path.dirname(folder_path))

    def mkdirp(self) -> None:
        return self.get_folder().mkdir(parents=True, exist_ok=True)
