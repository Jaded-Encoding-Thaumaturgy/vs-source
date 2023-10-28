from __future__ import annotations

import os
from abc import ABC, abstractmethod
from io import BufferedReader
from pprint import pformat
from struct import unpack
from typing import Any

from vstools import SPath, SPathLike

__all__ = [
    'SectorReadHelper',
]


class SectorReadHelper(ABC):
    def __init__(self, ifo: SPathLike | BufferedReader) -> None:
        if not isinstance(ifo, BufferedReader):
            file = SPath(ifo)
            ifo = file.open('rb')
        else:
            file = None

        self.ifo = ifo

        try:
            self._load()
        except Exception as e:
            raise e
        finally:
            if file is not None and self.ifo and not self.ifo.closed:
                ifo.close()

    def _goto_sector_ptr(self, pos: int):
        self.ifo.seek(pos, os.SEEK_SET)
        ptr, = self._unpack_byte(4)
        self.ifo.seek(ptr * 2048, os.SEEK_SET)

    def __repr__(self) -> str:
        return pformat(vars(self), sort_dicts=False)

    @abstractmethod
    def _load(self) -> None:
        ...

    def _seek_unpack_byte(self, addr: int, n: int | list[int]) -> tuple[Any, ...]:
        self.ifo.seek(addr, os.SEEK_SET)
        return self._unpack_byte(n)

    def _unpack_byte(self, n: int | list[int]) -> tuple[Any, ...]:
        stra = ">"

        if isinstance(n, int):
            n = [n]
        bytecnt = 0
        for a in n:
            bytecnt += a
            stra += {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}.get(a, 'B')

        buf = self.ifo.read(bytecnt)

        assert len(buf) == bytecnt
        return unpack(stra, buf)
