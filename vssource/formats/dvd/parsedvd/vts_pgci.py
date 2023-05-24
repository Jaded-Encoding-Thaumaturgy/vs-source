from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from pprint import pformat

from vstools import Region

from .sector import Sector, SectorOffset

__all__ = [
    'VTSPGCI', 'ProgramChain', 'PlaybackTime', 'PGCOffset', 'VTS_FRAMERATE'
]


class PGCOffset(IntEnum):
    """http://dvd.sourceforge.net/dvdinfo/pgc.html"""

    # 0x0000
    NB_PROGRAMS = 0x0002
    NB_CELLS = 0x0003
    PLAYBACK_TIME = 0x0004
    UOPS = 0x0008
    PGC_AST_CTL = 0x000C
    PGC_SPST_CTL = 0x001C
    NEXT_PGCN = 0x009C
    PREVIOUS_PGCN = 0x009E
    GOUP_PGCN = 0x00A0
    PGC_STILL_TIME = 0x00A2
    PG_PLAYBACK_MODE = 0x00A3
    PALETTE = 0x00A4

    COMMANDS_OFFSET = 0x00E4
    PROGRAM_MAP_OFFSET = 0x00E6
    CELL_PLAYBACK_INFO_TABLE_OFFSET = 0x00E8
    CELL_POS_INFO_TABLE_OFFSET = 0x00EA


@dataclass
class PlaybackTime:
    fps: int
    hours: int
    minutes: int
    seconds: int
    frames: int

    def __repr__(self) -> str:
        return pformat(vars(self), sort_dicts=False)


@dataclass
class ProgramChain:
    duration: PlaybackTime
    nb_program: int
    playback_times: list[PlaybackTime]

    def __repr__(self) -> str:
        return pformat(vars(self), sort_dicts=False)


class VTSPGCI(Sector):
    nb_program_chains: int
    program_chains: list[ProgramChain]

    chain_offset: int

    def _load(self) -> None:
        self.ifo.seek(SectorOffset.SECTOR_POINTER_VTS_PGCI, os.SEEK_SET)
        offset, = self._unpack_byte(4)

        self.ifo.seek(2048 * offset + 0x01, os.SEEK_SET)
        self.nb_program_chains, = self._unpack_byte(1)

        pcgit_pos = offset * 0x800

        self.ifo.seek(SectorOffset.SECTOR_POINTER_VTS_PGCI, os.SEEK_SET)

        self.program_chains = []

        for nbpgc in range(self.nb_program_chains):
            self.ifo.seek(pcgit_pos + (8 * (nbpgc + 1)) + 4, os.SEEK_SET)
            self.chain_offset, = self._unpack_byte(4)

            offset = pcgit_pos + self.chain_offset

            self.ifo.seek(offset + PGCOffset.NB_PROGRAMS, os.SEEK_SET)
            nb_program, = self._unpack_byte(1)

            self.ifo.seek(offset + PGCOffset.NB_CELLS, os.SEEK_SET)
            nb_cells, = self._unpack_byte(1)

            self.ifo.seek(offset + PGCOffset.PLAYBACK_TIME, os.SEEK_SET)
            duration = self._get_timespan(self.ifo.read(4))

            self.ifo.seek(offset + PGCOffset.PROGRAM_MAP_OFFSET, os.SEEK_SET)
            program_map_offset, = self._unpack_byte(2)

            self.ifo.seek(offset + PGCOffset.CELL_PLAYBACK_INFO_TABLE_OFFSET, os.SEEK_SET)
            cell_table_offset, = self._unpack_byte(2)

            playback_times: list[PlaybackTime] = []

            for program in range(nb_program):
                self.ifo.seek(offset + program_map_offset + program, os.SEEK_SET)
                entry_cell, = self._unpack_byte(1)

                exit_cell = entry_cell

                if program < nb_program - 1:
                    self.ifo.seek(offset + program_map_offset + program + 0x01, os.SEEK_SET)
                    exit_cell, = self._unpack_byte(1)
                    exit_cell -= 1
                else:
                    exit_cell = nb_cells

                for cell in range(entry_cell, exit_cell + 1):
                    cell_start = cell_table_offset + (cell - 1) * 0x18

                    self.ifo.seek(offset + cell_start, os.SEEK_SET)
                    cell_type = self.ifo.read(4)[0] >> 6

                    if cell_type in {0x02, 0x03}:
                        print('found different angle block:', cell_type)

                    self.ifo.seek(offset + cell_start + 0x0004, os.SEEK_SET)
                    playback_time = self._get_timespan(self.ifo.read(4))

                    playback_times.append(playback_time)

            self.program_chains.append(
                ProgramChain(duration, nb_program, playback_times)
            )

    def _get_timespan(self, data: bytes) -> PlaybackTime:
        frames = self._get_frames(data[3])
        fps = data[3] >> 6

        if fps not in VTS_FRAMERATE:
            raise ValueError

        hours, minutes, seconds = [self._bcd_to_int(data[i]) for i in range(3)]

        return PlaybackTime(fps, hours, minutes, seconds, frames)

    def _get_frames(self, byte: int) -> int:
        if ((byte >> 6) & 0x01) == 1:
            frames = self._bcd_to_int(byte & 0x3F)
        else:
            raise ValueError
        return frames

    @staticmethod
    def _bcd_to_int(bcd: int) -> int:
        return ((0xFF & (bcd >> 4)) * 10) + (bcd & 0x0F)


VTS_FRAMERATE = {
    0x01: Region.PAL.framerate,
    0x03: Region.NTSC.framerate
}
