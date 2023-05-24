from __future__ import annotations

from abc import ABC, abstractmethod
from io import BufferedReader
from pprint import pformat
from struct import unpack
from typing import Any

from vstools import CustomIntEnum, SPath, SPathLike

__all__ = [
    'Sector',

    'SectorOffset'
]


class Sector(ABC):
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

    def __repr__(self) -> str:
        return pformat(vars(self), sort_dicts=False)

    @abstractmethod
    def _load(self) -> None:
        ...

    def _unpack_byte(self, n: int) -> tuple[Any, ...]:
        return unpack({1: '>B', 2: '>H', 4: '>I', 8: '>Q'}.get(n, '>B'), self.ifo.read(n))


class SectorOffset(CustomIntEnum):
    DVDVIDEO_VTS = 0x0000
    LAST_SECTOR_TITLE_SET = 0x000C
    LAST_SECTOR_IFO = 0x001C
    VERSION_NB = 0x0020
    VTS_CATEGORY = 0x0022
    # 0x0026
    # 0x0028
    # 0x002A
    # 0x003E
    # 0x0040
    # 0x0060
    VTS_MAT_END_BYTE = 0x0080
    # 0x0084
    START_VECTOR_MENU_VOB = 0x00C0
    START_VECTOR_TITLE_VOB = 0x00C4
    SECTOR_POINTER_VTS_PTT_SRPT = 0x00C8
    SECTOR_POINTER_VTS_PGCI = 0x00CC
    SECTOR_POINTER_VTSM_PGCI_UT = 0x00D0
    SECTOR_POINTER_VTS_TMAPTI = 0x00D4
    SECTOR_POINTER_VTSM_C_ADT = 0x00D8
    SECTOR_POINTER_VTSM_VOBU_ADMAP = 0x00DC
    SECTOR_POINTER_VTS_C_ADT = 0x00E0
    SECTOR_POINTER_VTS_VOBU_ADMAP = 0x00E4

    VID_ATT_VTSM_VOBS = 0x0100
    NB_AUDIOS_VTSM_VOBS = 0x0102
    AUD_ATT_VTSM_VOBS = 0x0104
    # 0x0144
    NB_SUBPIC_STREAMS_VTSM_VOBS = 0x0154
    SUBPIC_ATT_VTSM_VOBS = 0x0156
    RESERVED = 0x015A
