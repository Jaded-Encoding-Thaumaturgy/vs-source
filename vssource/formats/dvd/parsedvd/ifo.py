from .sector import SectorReadHelper
import os
from vstools import Region


class IFO0(SectorReadHelper):
    def _load(self) -> None:
        crnt = dict()

        self.ifo.seek(0x3E, os.SEEK_SET)
        self.num_vts, = self._unpack_byte(2)


        # tt_srpt
        self._goto_sector_ptr(0x00C4)
        num, _, end = self._unpack_byte([2, 2, 4])

        tt_srpt = []
        for a in range(num):
            type, nr_of_angles, nr_of_ptts, parental, title_set_nr, vts_ttn, title_set_sector = self._unpack_byte([
                                                                                                                  1, 1, 2, 2, 1, 1, 4])
            entry = dict()

            entry["title_set_nr"] = title_set_nr
            entry["title_set_sector"] = title_set_sector
            entry["nr_of_angles"] = nr_of_angles
            entry["nr_of_ptts"] = nr_of_ptts
            entry["vts_ttn"] = vts_ttn
            tt_srpt += [entry]

        crnt["tt_srpt"] = tt_srpt

        crnt["vts_ptt_srpt"] = []
        crnt["pgci_ut"] = []
        crnt["vts_c_adt"] = []
        crnt["vts_pgcit"] = []

        self.crnt = crnt


class IFOX(SectorReadHelper):
    def _load(self) -> None:
        self.crnt = dict()
        self.vtsi_mat()
        self.vts_pgci()
        self.vobu_admap()
        self.vts_c_adt()
        self.vts_ptt_srpt()

        self.crnt["pgci_ut"] = []
        self.crnt["tt_srpt"] = []

    def vts_pgci(self):
        self._goto_sector_ptr(0x00CC)
        posn = self.ifo.tell()
        nr_pgcs, res, end = self._unpack_byte([2, 2, 4])

        pgcs = []

        for _ in range(nr_pgcs):
            cat, offset = self._unpack_byte([4, 4])
            bk = self.ifo.tell()

            audio_control = []

            pgc_base = posn + offset
            self.ifo.seek(pgc_base, os.SEEK_SET)
            res, num_programs, num_cells = self._unpack_byte([2, 1, 1])
            self._unpack_byte([4, 4])

            for a in range(8):
                ac, res = self._unpack_byte([1, 1])

                avalible = (ac & 0x80) != 0
                number = ac & 7

                jj = {}
                jj["available"] = avalible
                jj["number"] = number
                audio_control += [jj]
            for a in range(32):
                self._unpack_byte([4])
            next_pgcn, prev_pgcn, group_pgcn = self._unpack_byte([2, 2, 2])
            playback_mode, still_time = self._unpack_byte([1, 1])
            self._unpack_byte([4] * 16)

            offset_commands, offset_program, offset_playback, offset_position = self._unpack_byte([2, 2, 2, 2])

            self.ifo.seek(pgc_base + offset_program, os.SEEK_SET)
            program_map = list(self._unpack_byte([1] * num_programs))

            self.ifo.seek(pgc_base + offset_position, os.SEEK_SET)
            cell_position = [self._unpack_byte([2, 1, 1]) for _ in range(num_cells)]
            cell_position = [{"cell_nr": a[2], "vob_id_nr": a[0]} for a in cell_position]

            self.ifo.seek(pgc_base + offset_playback, os.SEEK_SET)
            cell_playback = [self._unpack_byte([1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4]) for _ in range(num_cells)]
            cell_playback = [{
                "interleaved": (a[0] & 0b100) != 0,
                "seamless_play": (a[0] & 0b1000) != 0,
                "seamless_angle": (a[0] & 0b1) != 0,


                "block_mode": ((a[0] & 0b11000000) >> 6),
                "block_type": ((a[0] & 0b00110000) >> 4),
                "playback_time": self._get_timespan(bytes([a[4], a[5], a[6], a[7]])),
                "first_sector": a[5 + 3],
                "last_sector": a[8 + 3],
                "first_ilvu_end_sector": a[6 + 3],
                "last_vobu_start_sector": a[7 + 3],
            } for a in cell_playback]

            self.ifo.seek(bk, os.SEEK_SET)

            pgcs += [{
                "nr_of_cells": num_cells,
                "nr_of_programs": num_programs,
                "next_pgc_nr": next_pgcn,
                "prev_pgc_nr": prev_pgcn,
                "goup_pgc_nr": group_pgcn,
                "program_map": program_map,
                "cell_position": cell_position,
                "cell_playback": cell_playback,

                "audio_control": audio_control,
            }]

        self.crnt["vts_pgcit"] = pgcs

    def vobu_admap(self):
        self._goto_sector_ptr(0x00E4)
        end, = self._unpack_byte(4)

        vobu_admap = []
        cnt = (end + 1 - 4) // 4
        for _ in range(cnt):
            vobu_admap += [self._unpack_byte(4)[0]]

        self.crnt["vts_vobu_admap"] = vobu_admap

    def vts_ptt_srpt(self):
        self._goto_sector_ptr(0x00C8)
        num, _res, end = self._unpack_byte([2, 2, 4])

        # not really sure with this
        correction = num * 4 + 8
        offsets = list(self._unpack_byte([4] * num))
        offsets = [x - correction for x in offsets]

        total_ptts = (end - correction + 4 + 1 - 4) // 4

        all_ptts_x = list(self._unpack_byte([2, 2] * total_ptts))
        all_ptts = []
        for i in range(len(all_ptts_x) // 2):
            all_ptts += [(all_ptts_x[i * 2 + 0], all_ptts_x[i * 2 + 1])]
        offsets = [a // 4 for a in offsets] + [len(all_ptts)]

        titles = []
        for a in range(num):
            ptts = all_ptts[offsets[a]:offsets[a + 1]]
            ptts = [{"pgcn": p[0], "pgn": p[1]} for p in ptts]
            titles += [ptts]
        self.crnt["vts_ptt_srpt"] = titles

    def vts_c_adt(self):
        self._goto_sector_ptr(0x00E0)
        vobcnt, res, end = self._unpack_byte([2, 2, 4])

        vts_c_adt = []
        cnt = (end + 1 - 6) // 12
        for _ in range(cnt):
            vob_id, cell_id, _res, start_sector, last_sector = self._unpack_byte([2, 1, 1, 4, 4])
            vts_c_adt += [{
                "vob_id": vob_id,
                "cell_id": cell_id,
                "start_sector": start_sector,
                "last_sector": last_sector,
            }]
        self.crnt["vts_c_adt"] = vts_c_adt

    def vtsi_mat(self):
        vtsi_mat = {}

        vb0, vb1, = self._seek_unpack_byte(0x0200, [1, 1])
        vts_video_attr = {}

        vts_video_attr["mpeg_version"] = (vb0 & 0b11000000) >> 6
        vts_video_attr["video_format"] = (vb0 & 0b00110000) >> 4
        vts_video_attr["picture_size"] = (vb1 & 0b00110000) >> 4

        vtsi_mat["vts_video_attr"] = vts_video_attr

        vts_audio_attr = []
        num_audio, = self._seek_unpack_byte(0x0202, 2)
        for _ in range(num_audio):
            buf = self.ifo.read(8)
            lang_type = (buf[0] & 0b1100) >> 2
            audio_format = (buf[0] & 0b11100000) >> 5
            if lang_type:
                lang = chr(buf[2]) + chr(buf[3])
            else:
                lang = "xx"

            vts_audio_attr += [{"audio_format": audio_format, "language": lang}]

        vtsi_mat["vts_audio_attr"] = vts_audio_attr

        self.crnt["vtsi_mat"] = vtsi_mat

    def _get_timespan(self, data: bytes) -> dict:
        _ = self._get_frames(data[3])  # frames
        fps = data[3] >> 6

        if fps not in VTS_FRAMERATE:
            raise ValueError

        hours, minutes, seconds = [self._bcd_to_int(data[i]) for i in range(3)]

    #    return PlaybackTime(fps, hours, minutes, seconds, frames)
        return {
            "hour": hours,
            "minute": minutes,
            "second": seconds,
            # "frames": frames,
            "frame_u": data[3],
            # "fps": fps,
        }

    def _get_frames(self, byte: int) -> int:
        if ((byte >> 6) & 0x01) == 1:
            frames = self._bcd_to_int(byte & 0x3F)
        else:
            raise ValueError
        return frames

    @staticmethod
    def _bcd_to_int(bcd: int) -> int:
        return bcd
        # libdvdread does not ??
        # return ((0xFF & (bcd >> 4)) * 10) + (bcd & 0x0F)


VTS_FRAMERATE = {
    0x01: Region.PAL.framerate,
    0x03: Region.NTSC.framerate
}
