from __future__ import annotations

from abc import ABC, abstractmethod
from io import BufferedReader
from pprint import pformat
from struct import unpack
from typing import Any, List

from vstools import CustomIntEnum, SPath, SPathLike

import os

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

    def _seek_unpack_byte(self, addr: int, n: int | List[int]) -> tuple[Any, ...]:
        self.ifo.seek(addr, os.SEEK_SET)
        return self._unpack_byte(n)

    def _unpack_byte(self, n: int | List[int]) -> tuple[Any, ...]:
        stra = ">"


        if isinstance(n, int):
            nn = n
            stra += {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}.get(n, 'B')
        else:
            nn = 0
            for a in n:
                nn += a
                stra += {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}.get(a, 'B')
        #print(stra,nn)
        buf = self.ifo.read(nn)
        assert len(buf) == nn
        return unpack(stra, buf)
