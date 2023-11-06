from __future__ import annotations

import datetime
import io
import json
import os
import subprocess
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from itertools import count
from typing import Any, Callable, Sequence, SupportsFloat, cast

from vstools import (
    CustomValueError, FuncExceptT, SPath, SPathLike, SupportsString, T, copy_signature, get_prop, remap_frames,
    set_output, to_arr, vs
)

from ...dataclasses import D2VIndexFrameData
from ...indexers import D2VWitch, DGIndex, ExternalIndexer
from ...rff import apply_rff_array, apply_rff_video, cut_array_on_ranges
from .parsedvd import (
    AUDIO_FORMAT_AC3, AUDIO_FORMAT_LPCM, IFO0, IFOX, SectorReadHelper, to_json,
    BLOCK_MODE_FIRST_CELL, BLOCK_MODE_IN_BLOCK, BLOCK_MODE_LAST_CELL
)

__all__ = [
    'IsoFileCore', 'Title'
]

DVD_DEBUG = "DVD_DEBUG" in os.environ


@copy_signature(print)
def debug_print(*args: Any, **kwargs: Any) -> None:
    if DVD_DEBUG:
        print(*args, **kwargs)


# d2vwitch needs this patch applied
# https://gist.github.com/jsaowji/ead18b4f1b90381d558eddaf0336164b

# https://gist.github.com/jsaowji/2bbf9c776a3226d1272e93bb245f7538
def double_check_dvdnav(iso: SupportsString, title: int) -> list[float] | None:
    try:
        ap = subprocess.check_output(["dvdsrc_dvdnav_title_ptt_test", str(iso), str(title)])

        return list(map(float, ap.splitlines()))
    except FileNotFoundError:
        ...

    return None


def absolute_time_from_timecode(timecodes: Sequence[SupportsFloat]) -> list[float]:
    absolutetime = list[float]([0.0])

    for i, a in enumerate(timecodes):
        absolutetime.append(absolutetime[i] + float(a))

    return absolutetime


def get_sectorranges_for_vobcellpair(current_vts: IFOX, pair_id: tuple[int, int]) -> list[tuple[int, int]]:
    return [
        (e.start_sector, e.last_sector)
        for e in current_vts.vts_c_adt.cell_adr_table
        if (e.vob_id, e.cell_id) == pair_id
    ]


@dataclass
class AllNeddedDvdFrameData:
    vobids: list[int]
    tff: list[int]
    rff: list[int]
    prog: list[int]
    progseq: list[int]


@dataclass
class SplitTitle:
    video: vs.VideoNode
    audio: list[vs.AudioNode] | None
    chapters: list[int]

    _title: Title
    _split_chpts: tuple[int, int]  # inclusive inclusive

    def ac3(self, outfile: str, audio_i: int = 0) -> float:
        a = self._split_chpts
        return SplitHelper.split_range_ac3(self._title, a[0], a[1], audio_i, outfile)

    def __repr__(self) -> str:
        # TODO: use absolutetime from title
        _absolute_time = absolute_time_from_timecode(
            [1 / float(self.video.fps)] * len(self.video)
        )

        chapters = self.chapters + [len(self.video) - 1]

        chapter_lengths = [
            _absolute_time[chapters[i + 1]] - _absolute_time[chapters[i]]
            for i in range(len(self.chapters))
        ]

        chapter_lengths_str = [
            str(datetime.timedelta(seconds=x)) for x in chapter_lengths
        ]

        timestrings = [
            str(datetime.timedelta(seconds=_absolute_time[x])) for x in self.chapters
        ]

        to_print = ["Chapters:"]

        to_print.extend([
            f'{i:02} {tms:015} {cptls:015} {cpt}'
            for i, tms, cptls, cpt in zip(count(1), timestrings, chapter_lengths_str, self.chapters)
        ])

        to_print.append("Audios: (fz)")

        if self.audio is not None:
            to_print.extend([f'{i} {a}' for i, a in enumerate(self.audio)])

        return '\n'.join(to_print)


@dataclass
class Title:
    node: vs.VideoNode
    chapters: list[int]

    # only for reference for gui or sth
    cell_changes: list[int]

    _core: IsoFileCore
    _title: int
    _vts: int
    _vobidcellids_to_take: list[tuple[int, int]]
    _dvdsrc_ranges: list[int]
    _absolute_time: list[float]
    _duration_times: list[float]
    _audios: list[str]
    _patched_end_chapter: int | None

    def split_at(self, splits: list[int], audio: int | list[int] | None = None) -> tuple[SplitTitle, ...]:
        output_cnt = SplitHelper._sanitize_splits(self, splits)
        video = SplitHelper.split_video(self, splits)
        chapters = SplitHelper.split_chapters(self, splits)

        audios: list[list[vs.AudioNode] | None]

        if audio is not None and (audio := to_arr(audio)):
            audio_per_output_cnt = len(audio)

            auds = [SplitHelper.split_audio(self, splits, a) for a in audio]

            audios = [
                [auds[j][i] for j in range(audio_per_output_cnt)] for i in range(output_cnt)
            ]
        else:
            audios = [None] * output_cnt

        fromy = 1
        from_to_s = list[tuple[int, int]]()

        for j in splits:
            from_to_s += [(fromy, j - 1)]
            fromy = j

        from_to_s += [(fromy, len(self.chapters) - 1)]

        return tuple(
            SplitTitle(v, a, c, self, f) for v, a, c, f in zip(video, audios, chapters, from_to_s)
        )

    def split_ranges(
        self, split: list[tuple[int, int]], audio: list[int] | int | None = None
    ) -> tuple[SplitTitle, ...]:
        return tuple(self.split_range(start, end, audio) for start, end in split)

    def split_range(self, start: int, end: int, audio: list[int] | int | None = None) -> SplitTitle:
        """
        starting from 1

        from: inclusive
        to: inclusive
        """
        if start < 0:
            start = len(self.chapters) + start

        if end < 0:
            end = len(self.chapters) + end

        if start == 1 and end == len(self.chapters) - 1:
            return self.split_at([], audio)[0]

        if start == 1:
            return self.split_at([end + 1], audio)[0]

        if end == len(self.chapters) - 1:
            return self.split_at([start], audio)[1]

        return self.split_at([start, end + 1], audio)[1]

    def preview(self, split: SplitTitle | Sequence[SplitTitle] | None = None) -> None:
        set_output(self.video(), f"title v {self._title}")

        if split is not None:
            split = to_arr(split)

            for i, s in enumerate(split):
                set_output(s.video, f'split {i}')

            for i, s in enumerate(split):
                if s.audio:
                    for j, audio in enumerate(s.audio):
                        set_output(audio, f'split {i} - {j}')

    def video(self) -> vs.VideoNode:
        return self.node

    def audio(self, i: int = 0) -> vs.AudioNode:
        self._assert_dvdsrc2(self.audio)

        asd = self._audios[i]

        anode: vs.AudioNode
        args = (self._core.iso_path, self._vts, i, self._dvdsrc_ranges)
        if asd.startswith("ac3"):
            anode = vs.core.dvdsrc2.FullVtsAc3(*args)  # type: ignore
        elif asd.startswith("lpcm"):
            anode = vs.core.dvdsrc2.FullVtsLpcm(*args)  # type: ignore
        else:
            raise CustomValueError('Invalid index for audio node!', self.audio)

        prps = anode.get_frame(0).props

        strt = (get_prop(prps, 'Stuff_Start_PTS', int) * anode.sample_rate) / 90_000
        endd = (get_prop(prps, 'Stuff_End_PTS', int) * anode.sample_rate) / 90_000

        debug_print(
            "splice", round((strt / anode.sample_rate) * 1000 * 10) / 10,
            "ms", round((endd / anode.sample_rate) * 1000 * 10) / 10, "ms"
        )

        start, end = int(strt), int(endd)
        anode = anode[start:len(anode) - end]

        total_dura = (self._absolute_time[-1] + self._duration_times[-1])
        delta = abs(total_dura - anode.num_samples / anode.sample_rate) * 1000

        debug_print(f"delta {delta} ms")

        if delta > 50:
            debug_print(f'Rather big audio/video lenght delta might be indecator that something is off {delta}!')

        return anode

    def _assert_dvdsrc2(self, func: FuncExceptT) -> None:
        if self._dvdsrc_ranges is None or len(self._dvdsrc_ranges) == 0:
            raise CustomValueError("Title needts to be opened with dvdsrc2!", func)

        if not self._core.use_dvdsrc:
            raise CustomValueError("This feature requires dvdsrc2!", func)

    def dump_ac3(self, a: str, audio_i: int = 0, only_calc_delay: bool = False) -> float:
        self._assert_dvdsrc2(self.dump_ac3)

        if not self._audios[audio_i].startswith("ac3"):
            raise CustomValueError(f"Autio at {audio_i} is not ac3", self.dump_ac3)

        nd = cast(vs.AudioNode, vs.core.dvdsrc2.RawAc3(self._core.iso_path, self._vts, audio_i, self._dvdsrc_ranges))
        p0 = nd.get_frame(0).props

        if not only_calc_delay:
            with open(a, 'wb') as wrt:
                for f in nd.frames():
                    wrt.write(bytes(f[0]))

        return float(get_prop(p0, 'Stuff_Start_PTS', int)) / 90_000

    def __repr__(self) -> str:
        chapters = [*self.chapters, len(self.node) - 1]
        chapter_lengths = [
            self._absolute_time[chapters[i + 1]] - self._absolute_time[chapters[i]]
            for i in range(len(self.chapters))
        ]

        chapter_lengths_str = [str(datetime.timedelta(seconds=x)) for x in chapter_lengths]
        timestrings = [str(datetime.timedelta(seconds=self._absolute_time[x])) for x in self.chapters]

        to_print = 'Chapters:\n'
        for i in range(len(timestrings)):
            to_print += f'{i + 1:02} {timestrings[i]:015} {chapter_lengths_str[i]:015} {self.chapters[i]}'

            if i == 0:
                to_print += ' (faked)'

            if self._patched_end_chapter is not None and i == len(timestrings) - 1:
                delta = self.chapters[i] - self._patched_end_chapter
                to_print += f' (originally {self._patched_end_chapter} delta {delta})'

            to_print += '\n'

        to_print += f'\ncellchange: {self.cell_changes}\n'
        to_print += '\nAudios: (fz)\n'
        for i, a in enumerate(self._audios):
            to_print += f'{i} {a}\n'

        return to_print.strip()


class SplitHelper:
    @staticmethod
    def split_range_ac3(title: Title, f: int, t: int, audio_i: int, outfile: str) -> float:
        nd = cast(vs.AudioNode, vs.core.dvdsrc2.RawAc3(title._core.iso_path, title._vts, audio_i, title._dvdsrc_ranges))
        prps = nd.get_frame(0).props

        start, end = (get_prop(prps, f'Stuff_{x}_PTS', int) for x in ('Start', 'End'))

        debug_print(f"Stuff_Start_PTS pts {start} Stuff_End_PTS {end}")

        raw_start = (title._absolute_time[title.chapters[f - 1]] * 90_000)
        raw_end = ((title._absolute_time[title.chapters[t]] + title._duration_times[title.chapters[t]]) * 90_000)

        start_pts = raw_start + start
        end_pts = start_pts + (raw_end - raw_start)

        audio_offset_pts = 0.0

        with open(outfile, 'wb') as outf:
            debug_print(f'start_pts  {start_pts} end_pts {end_pts}')

            start = int(start_pts / 2880)

            debug_print("first ", start, len(nd))

            for i in range(start, len(nd)):
                pkt_start_pts = i * 2880
                pkt_end_pts = (i + 1) * 2880

                assert pkt_end_pts > start_pts

                if pkt_start_pts < start_pts:
                    audio_offset_pts = start_pts - pkt_start_pts

                outf.write(bytes(nd.get_frame(i)[0]))

                if pkt_end_pts > end_pts:
                    break

        debug_print("wrote", (i - (start_pts // 2880)))
        debug_print("offset is", (audio_offset_pts) / 90, "ms")

        return audio_offset_pts / 90_000

    @staticmethod
    def split_chapters(title: Title, splits: list[int]) -> list[list[int]]:
        out = list[list[int]]()

        rebase = title.chapters[0]  # normally 0
        chaps = list[int]()

        for i, a in enumerate(title.chapters):
            chaps += [a - rebase]
            if (i + 1) in splits:
                rebase = a

                out += [chaps]
                chaps = [0]

        if len(chaps) >= 1:
            out += [chaps]

        assert len(out) == len(splits) + 1

        return out

    @staticmethod
    def split_video(title: Title, splits: list[int]) -> tuple[vs.VideoNode, ...]:
        reta = SplitHelper._cut_split(title, splits, title.node, SplitHelper._cut_fz_v)
        assert len(reta) == len(splits) + 1
        return reta

    @staticmethod
    def split_audio(title: Title, splits: list[int], i: int = 0) -> tuple[vs.AudioNode, ...]:
        reta = SplitHelper._cut_split(title, splits, title.audio(i), SplitHelper._cut_fz_a)
        assert len(reta) == len(splits) + 1
        return reta

    @staticmethod
    def _sanitize_splits(title: Title, splits: list[int]) -> int:
        assert isinstance(splits, list)

        lasta = -1

        for a in splits:
            assert isinstance(a, int)
            assert a > lasta
            assert a <= len(title.chapters)

            lasta = a

        return len(splits) + 1

    @staticmethod
    def _cut_split(title: Title, splits: list[int], a: T, b: Callable[[Title, T, int, int], T]) -> tuple[T, ...]:
        out, last = list[T](), 0

        for s in splits:
            index = s - 1
            out += [b(title, a, last, index)]
            last = index

        out += [b(title, a, last, len(title.chapters) - 1)]

        return tuple(out)

    @staticmethod
    def _cut_fz_v(title: Title, vnode: vs.VideoNode, f: int, t: int) -> vs.VideoNode:
        # starting 0
        # end inclusive
        #  0 0 -> chapter 0
        f = title.chapters[f]
        t = title.chapters[t]
        assert f >= 0
        assert t <= len(vnode) - 1
        assert f <= t
        return vnode[f:t]

    @staticmethod
    def _cut_fz_a(title: Title, anode: vs.AudioNode, start: int, end: int) -> vs.AudioNode:
        chapter_idxs = [title.chapters[i] for i in (start, end)]
        timecodes = [title._absolute_time[i] for i in chapter_idxs]
        sample_cuts = [
            min(round(i * anode.sample_rate), anode.num_samples)
            for i in timecodes
        ]

        samples_start, samples_end, *_ = sample_cuts

        return anode[samples_start:samples_end]


class IsoFileCore:
    _subfolder = "VIDEO_TS"
    ifo0: IFO0
    vts: list[IFOX]

    def __init__(
        self, path: SPath | str,
        indexer: ExternalIndexer | type[ExternalIndexer] | None = None,
    ):
        """
        Only external indexer supported D2VWitch and DGIndex

        If the indexer is None dvdsrc is used

        """
        self.force_root = False
        self.output_folder = "/tmp" if os.name != "nt" else "C:/tmp"

        self._mount_path: SPath | None = None
        self._vob_files: list[SPath] | None = None
        self._ifo_files: list[SPath] | None = None

        self.has_dvdsrc1 = hasattr(vs.core, "dvdsrc")
        self.has_dvdsrc2 = hasattr(vs.core, "dvdsrc2")
        self.use_dvdsrc = indexer is None

        if self.use_dvdsrc and not self.has_dvdsrc2:
            indexer = DGIndex() if os.name == "nt" else D2VWitch()
            self.use_dvdsrc = False

        self.iso_path = SPath(path).absolute()

        #this check seems stupid
        if not (self.iso_path.is_dir() or self.iso_path.is_file()):
            raise CustomValueError('"path" needs to point to a .ISO or a dir root of DVD', str(path), self.__class__)

#        ifo0: SPathLike | io.BufferedReader | None = None
#        ifos: Sequence[SPathLike | io.BufferedReader] = []
#        if self.use_dvdsrc:
#            _ifo0, *_ifos = [
#                cast(bytes, vs.core.dvdsrc2.Ifo(self.iso_path, i)) for i in range(self.ifo0.num_vts + 1)
#            ]
#
#            if len(_ifo0) <= 30:
#                warnings.warn('Newer VapourSynth is required for dvdsrc2 information gathering without mounting!')
#            else:
#                ifo0, *ifos = [io.BufferedReader(io.BytesIO(x)) for x in (_ifo0, *_ifos)]  # type: ignore
#
#        if not ifo0:
#            ifo0, *ifos = self.ifo_files
#
#        self.ifo0 = IFO0(SectorReadHelper(ifo0))
#        self.vts = [IFOX(SectorReadHelper(ifo)) for ifo in ifos]

        read_ifo_from_mount = False

        if self.use_dvdsrc:
            i0bytes = vs.core.dvdsrc2.Ifo(self.iso_path, 0)
            if len(i0bytes) <= 30:
                read_ifo_from_mount = True
                print('newer Vapoursynth is required for dvdsrc2 information gathering without mounting')
            else:
                self.ifo0 = IFO0(SectorReadHelper(io.BufferedReader(io.BytesIO(i0bytes))))
                self.vts = []

                for i in range(1, self.ifo0.num_vts + 1):
                    rh = SectorReadHelper(io.BufferedReader(io.BytesIO(vs.core.dvdsrc2.Ifo(self.iso_path, i))))
                    self.vts += [IFOX(rh)]

        if not self.use_dvdsrc or read_ifo_from_mount:
            for i, a in enumerate(self.ifo_files):
                if i == 0:
                    self.ifo0 = IFO0(SectorReadHelper(a))
                    self.vts = []
                else:
                    self.vts += [IFOX(SectorReadHelper(a))]

        self.json = to_json(self.ifo0, self.vts)

        if self.has_dvdsrc1:
            dvdsrc_json = json.loads(cast(str, vs.core.dvdsrc.Json(self.iso_path)))

            for key in ('dvdpath', 'current_vts', 'current_domain'):
                dvdsrc_json.pop(key, None)

            for ifo in dvdsrc_json.get('ifos', []):
                ifo['pgci_ut'] = []

            ja = json.dumps(dvdsrc_json, sort_keys=True)
            jb = json.dumps(self.json, sort_keys=True)

            if ja != jb:
                warnings.warn(
                    f"libdvdread json does not match python implentation\n"
                    f"json a,b have been written to {self.output_folder}"
                )

                for k, v in [('a', ja), ('b', jb)]:
                    with open(os.path.join(self.output_folder, f'{k}.json'), 'wt') as file:
                        file.write(v)
        else:
            debug_print("We don't have dvdsrc and can't double check the json output with libdvdread.")

        if not self.use_dvdsrc:
            assert indexer
            self.indexer = indexer if isinstance(indexer, ExternalIndexer) else indexer()

        self.title_count = len(self.ifo0.tt_srpt)

    def get_vts(self, title_set_nr: int = 1, d2v_our_rff: bool = False) -> vs.VideoNode:
        """
        Gets a full vts.
        only works with dvdsrc2 and d2vsource usese our rff for dvdsrc and d2source rff for d2vsource

        mainly useful for debugging and checking if our rff algorithm is good
        """

        if not self.use_dvdsrc or d2v_our_rff:
            rawnode = cast(vs.VideoNode, vs.core.dvdsrc2.FullVts(self.iso_path, vts=title_set_nr))
            staff = IsoFileCore._dvdsrc2_extract_data(rawnode)

        if self.use_dvdsrc:
            vob_input_files = self._get_title_vob_files_for_vts(title_set_nr)
            index_file = self.indexer.index(vob_input_files, output_folder=self.output_folder)[0]

            if d2v_our_rff:
                return apply_rff_video(
                    self.indexer._source_func(index_file, rff=False), staff.rff, staff.tff, staff.prog, staff.progseq
                )

            return self.indexer._source_func(index_file, rff=True)

        return apply_rff_video(rawnode, staff.rff, staff.tff, staff.prog, staff.progseq)

    def get_title(self, title_nr: int = 1, angle_nr: int | None = None, rff_mode: int = 0) -> Title:
        """
        Gets a title.

        :param title_nr:            title nr starting from 1
        :param angle_nr:            starting from 1
        :param rff_mode:            0 apply rff soft telecine (default)
                                    1 calculate per frame durations based on rff
                                    2 set average fps on global clip
        """
        # TODO: assert angle_nr range
        disable_rff = rff_mode >= 1

        tt_srpt = self.ifo0.tt_srpt
        title_idx = title_nr - 1

        if title_idx < 0 or title_idx >= len(tt_srpt):
            raise CustomValueError('"title_nr" out of range', self.get_title)

        tt = tt_srpt[title_idx]

        if tt.nr_of_angles != 1 and angle_nr is None:
            raise CustomValueError('no angle_nr given for multi angle title', self.get_title)

        target_vts = self.vts[tt.title_set_nr - 1]
        target_title = target_vts.vts_ptt_srpt[tt.vts_ttn - 1]

        assert len(target_title) == tt.nr_of_ptts

        for ptt in target_title[1:]:
            if ptt.pgcn != target_title[0].pgcn:
                warnings.warn('title is not one program chain (untested currently)')

        vobidcellids_to_take = list[tuple[int, int]]()
        is_chapter = []

        i = 0
        while i < len(target_title):
            ptt_to_take_for_pgc = 0

            for j in target_title[i:]:
                if target_title[i].pgcn != j.pgcn:
                    break
                ptt_to_take_for_pgc += 1
            assert ptt_to_take_for_pgc >= 1

            title_programs = [a.pgn for a in target_title[i:i + ptt_to_take_for_pgc]]
            target_pgc = target_vts.vts_pgci.pgcs[target_title[i].pgcn - 1]
            pgc_programs = target_pgc.program_map

            if title_programs[0] != 1 or pgc_programs[0] != 1:
                warnings.warn('Open Title does not start at the first cell (open issue in github with sample)\n')

            target_programs = [a[1]
                               for a in list(filter(lambda x: (x[0] + 1) in title_programs, enumerate(pgc_programs)))]

            if target_programs != pgc_programs:
                warnings.warn('Open the program chain does not include all ptts\n')

            current_angle = 1
            angle_start_cell_i: int

            for cell_i in range(len(target_pgc.cell_position)):
                cell_position = target_pgc.cell_position[cell_i]
                cell_playback = target_pgc.cell_playback[cell_i]

                block_mode = cell_playback.block_mode

                if block_mode == BLOCK_MODE_FIRST_CELL:
                    current_angle = 1
                    angle_start_cell_i = cell_i
                elif block_mode == BLOCK_MODE_IN_BLOCK or block_mode == BLOCK_MODE_LAST_CELL:
                    current_angle += 1

                if block_mode == 0:
                    take_cell = True
                    angle_start_cell_i = cell_i
                else:
                    take_cell = current_angle == angle_nr

                if take_cell:
                    vobidcellids_to_take += [(cell_position.vob_id_nr, cell_position.cell_nr)]
                    is_chapter += [(angle_start_cell_i + 1) in target_programs]

            i += ptt_to_take_for_pgc

        assert len(is_chapter) == len(vobidcellids_to_take)

        # should set rnode, vobids and rff, dvdsrc_ranges
        if self.use_dvdsrc:
            admap = target_vts.vts_vobu_admap

            all_ranges = []

            for a in vobidcellids_to_take:
                all_ranges += get_sectorranges_for_vobcellpair(target_vts, a)
            idxx = []
            for a in all_ranges:
                start_index = admap.index(a[0])
                try:
                    end_index = admap.index(a[1] + 1) - 1
                except ValueError:
                    end_index = len(admap) - 1
                idxx += [start_index, end_index]

            rawnode = cast(vs.VideoNode, vs.core.dvdsrc2.FullVts(self.iso_path, vts=tt.title_set_nr, ranges=idxx))
            staff = IsoFileCore._dvdsrc2_extract_data(rawnode)

            if not disable_rff:
                rnode = apply_rff_video(rawnode, staff.rff, staff.tff, staff.prog, staff.progseq)
                vobids = apply_rff_array(staff.vobids, staff.rff, staff.tff, staff.progseq)
            else:
                rnode = rawnode
                vobids = staff.vobids
            dvdsrc_ranges = idxx
            rff = staff.rff
        else:
            dvdsrc_ranges = []
            vob_input_files = self._get_title_vob_files_for_vts(tt.title_set_nr)
            dvddd = self._d2v_vobid_frameset(tt.title_set_nr)

            if len(dvddd.keys()) == 1 and (0, 0) in dvddd.keys():
                raise CustomValueError(
                    'Youre indexer created a d2v file with only zeros for vobid cellid; '
                    'This usually means outdated/unpatched D2Vwitch', self.get_title
                )

            frameranges = []
            for a in vobidcellids_to_take:
                frameranges += dvddd[a]

            fflags, vobids, progseq = self._d2v_collect_all_frameflags(tt.title_set_nr)

            index_file = self.indexer.index(vob_input_files, output_folder=self.output_folder)[0]
            node = self.indexer._source_func(index_file, rff=False)

            assert len(node) == len(fflags) == len(vobids) == len(progseq)

            progseq = cut_array_on_ranges(progseq, frameranges)
            fflags = cut_array_on_ranges(fflags, frameranges)
            vobids = cut_array_on_ranges(vobids, frameranges)
            node = remap_frames(node, frameranges)

            rff = [(a & 1) for a in fflags]

            if not disable_rff:
                tff = [(a & 2) >> 1 for a in fflags]
                prog = [(a & 0b01000000) != 0 for a in fflags]

                # just be sure
                prog = [int(a) for a in prog]
                tff = [int(a) for a in tff]

                rnode = apply_rff_video(node, rff, tff, prog, progseq)
                vobids = apply_rff_array(vobids, rff, tff, progseq)
            else:
                rnode = node
                vobids = vobids

        rfps = float(rnode.fps)
        abs1 = abs(25 - rfps)
        abs2 = abs(30 - rfps)
        if abs1 < abs2:
            fpsnum, fpsden = 25, 1
        else:
            fpsnum, fpsden = 30000, 1001

        if not disable_rff:
            rnode = vs.core.std.AssumeFPS(rnode, fpsnum=fpsnum, fpsden=fpsden)
            durationcodes = [fpsden / fpsnum for a in range(len(rnode))]
            absolutetime = [a * (fpsden / fpsnum) for a in range(len(rnode))]
        else:
            if rff_mode == 1:
                timecodes = [Fraction(fpsden * (a + 2), fpsnum * 2) for a in rff]
                durationcodes = timecodes
                absolutetime = absolute_time_from_timecode(timecodes)

                def _apply_timecodes(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                    f = f.copy()

                    f.props._DurationNum = timecodes[n].numerator
                    f.props._DurationDen = timecodes[n].denominator
                    f.props._AbsoluteTime = absolutetime[n]

                    return f

                rnode = rnode.std.ModifyFrame(rnode, _apply_timecodes)
            else:
                rffcnt = 0
                for a in rff:
                    if a:
                        rffcnt += 1

                asd = (rffcnt * 3 + 2 * (len(rff) - rffcnt)) / len(rff)

                fcc = len(rnode) * 5
                new_fps = Fraction(fpsnum * fcc * 2, int(fcc * fpsden * asd),)

                rnode = vs.core.std.AssumeFPS(rnode, fpsnum=new_fps.numerator, fpsden=new_fps.denominator)

                timecodes = [Fraction(fpsden * (a + 2), fpsnum * 2) for a in rff]
                durationcodes = timecodes
                absolutetime = absolute_time_from_timecode(timecodes)

        changes = []
        for a in range(1, len(vobids)):
            if vobids[a] != vobids[a - 1]:
                changes += [a]

        changes += [len(rnode) - 1]

        assert len(changes) == len(is_chapter)

        last_chapter_i = 0
        for i, a in reversed(list(enumerate(is_chapter))):
            if a:
                last_chapter_i = i
                break

        output_chapters = []
        for i in range(len(is_chapter)):
            a = is_chapter[i]

            if not a:
                continue

            broke = False

            for j in range(i + 1, len(is_chapter)):
                if is_chapter[j]:
                    broke = True
                    break
            output_chapters += [changes[last_chapter_i] if not broke else changes[j - 1]]

        dvnavchapters = double_check_dvdnav(self.iso_path, title_nr)

        if dvnavchapters is not None:  # and (rff_mode == 0 or rff_mode == 2):
            # ???????
            if fpsden == 1001:
                dvnavchapters = [a * 1.001 for a in dvnavchapters]

            adjusted = [absolutetime[i] for i in output_chapters]  # [1:len(output_chapters)-1] ]
            if len(adjusted) != len(dvnavchapters):
                warnings.warn(f'dvdnavchapters length do not match our chapters {len(adjusted)} {len(dvnavchapters)}'
                              ' (open an issue in github)')
                print(adjusted, "\n\n\n", dvnavchapters)
            else:
                framelen = fpsden / fpsnum
                for i in range(len(adjusted)):
                    # tolerance no idea why so big
                    # on hard telecine ntcs it matches up almost perfectly
                    # but on ~24p pal rffd it does not lol
                    if abs(adjusted[i] - dvnavchapters[i]) > framelen * 20:
                        warnings.warn(
                            f'dvdnavchapters do not match our chapters {len(adjusted)} {len(dvnavchapters)}'
                            ' (open an issue in github)\n'
                            f' index: {i} {adjusted[i]}'
                        )
                        print(adjusted, "\n\n\n", dvnavchapters)
                        break
        else:
            debug_print("Skipping sanity check with dvdnav")

        patched_end_chapter = None
        # only the chapter | are defined by dvd
        # (the splitting logic assumes though that there is a chapter at the start and end)
        # TODO: verify these claims and check the splitting logic and figure out what the best solution is
        # you could either always add the end as chapter or stretch the last chapter till the end
        output_chapters = [0] + output_chapters

        lastframe = len(rnode) - 1
        if output_chapters[-1] != lastframe:
            patched_end_chapter = output_chapters[-1]
            output_chapters[-1] = lastframe

        audios = []
        for i, ac in enumerate(target_pgc.audio_control):
            if ac.available:
                audio = target_vts.vtsi_mat.vts_audio_attr[i]

                if audio.audio_format == AUDIO_FORMAT_AC3:
                    aformat = "ac3"
                elif audio.audio_format == AUDIO_FORMAT_LPCM:
                    aformat = "lpcm"
                else:
                    aformat = "unk"

                audios += [f'{aformat}({audio.language})']
            else:
                audios += ["none"]

        return Title(
            rnode, output_chapters, changes, self, title_nr, tt.title_set_nr,
            vobidcellids_to_take, dvdsrc_ranges, absolutetime, durationcodes,
            audios, patched_end_chapter
        )

    def _d2v_collect_all_frameflags(self, title_set_nr: int) -> tuple[list[int], list[tuple[int, int]], list[int]]:
        files = self._get_title_vob_files_for_vts(title_set_nr)
        index_file = self.indexer.index(files, output_folder=self.output_folder)[0]
        index_info = self.indexer.get_info(index_file)

        frameflagslst = list[int]()
        vobidlst = list[tuple[int, int]]()
        progseqlst = list[int]()

        for iframe in index_info.frame_data:
            assert isinstance(iframe, D2VIndexFrameData)

            vobcell = (iframe.vob, iframe.cell)

            progseq = int(((iframe.info & 0b1000000000) != 0))

            for a in iframe.frameflags:
                if a != 0xFF:
                    frameflagslst += [a]
                    vobidlst += [vobcell]
                    progseqlst += [progseq]

        return frameflagslst, vobidlst, progseqlst

    def _d2v_vobid_frameset(self, title_set_nr: int) -> dict[tuple[int, int], list[list[int]]]:
        _, vobids, _ = self._d2v_collect_all_frameflags(title_set_nr)

        vobidset = dict[tuple[int, int], list[list[int]]]()
        for i, a in enumerate(vobids):
            if a not in vobidset:
                vobidset[a] = [[i, i - 1]]

            last = vobidset[a][-1]

            if last[1] + 1 == i:
                last[1] += 1
                continue

            vobidset[a] += [[i, i]]

        return vobidset

    def _get_title_vob_files_for_vts(self, vts: int) -> Sequence[SPath]:
        return [
            vob for vob in self.vob_files
            if f'VTS_{vts:02}_' in (s := vob.to_str()) and not s.upper().endswith("0.VOB")
        ]

    def _mount_folder_path(self) -> SPath:
        if self.force_root:
            return self.iso_path

        if self.iso_path.name.upper() == self._subfolder:
            self.iso_path = self.iso_path.parent

        return self.iso_path / self._subfolder

    @staticmethod
    def _dvdsrc2_extract_data(rawnode: vs.VideoNode) -> AllNeddedDvdFrameData:
        dd = bytes(get_prop(rawnode, 'InfoFrame', vs.VideoFrame)[0])  # type: ignore

        assert len(dd) == len(rawnode) * 4  # type: ignore

        vobids = list[tuple[int, int]]()
        tff = list[int]()
        rff = list[int]()
        prog = list[int]()
        progseq = list[int]()

        for i in range(len(rawnode)):
            sb = dd[i * 4 + 0]

            vobids += [((dd[i * 4 + 1] << 8) + dd[i * 4 + 2], dd[i * 4 + 3])]
            tff += [(sb & (1 << 0)) >> 0]
            rff += [(sb & (1 << 1)) >> 1]
            prog += [(sb & (1 << 2)) >> 2]
            progseq += [(sb & (1 << 3)) >> 3]

        return AllNeddedDvdFrameData(vobids, tff, rff, prog, progseq)

    def __repr__(self) -> str:
        to_print = f"Path: {self.iso_path}\n"
        to_print += f"Mount: {self._mount_path}\n"
        for i, tt in enumerate(self.ifo0.tt_srpt):
            target_vts = self.vts[tt.title_set_nr - 1]
            ptts = target_vts.vts_ptt_srpt[tt.vts_ttn - 1]

            current_time = 0.0
            timestrings = []
            absolutestrings = []
            for a in ptts:
                target_pgc = target_vts.vts_pgci.pgcs[a.pgcn - 1]
                cell_n = target_pgc.program_map[a.pgn - 1]

                chap_time = target_pgc.cell_playback[cell_n - 1].playback_time.get_seconds_float()
                current_time += chap_time

                timestrings += [str(datetime.timedelta(seconds=chap_time))]
                absolutestrings += [str(datetime.timedelta(seconds=current_time))]

            to_print += f"Title: {i+1:02}\n"
            to_print += f"length: {timestrings}\n"
            to_print += f"end   : {absolutestrings}\n"
            to_print += "\n"

        return to_print.strip()

    @property
    def mount_path(self) -> SPath:
        if self._mount_path is not None:
            return self._mount_path

        if self.iso_path.is_dir():
            return self._mount_folder_path()

        disc = self._get_mounted_disc() or self._mount()
        if not disc:
            raise RuntimeError("IsoFile: Couldn't mount ISO file!")

        self._mount_path = disc / self._subfolder

        return self._mount_path

    @property
    def vob_files(self) -> list[SPath]:
        if self._vob_files is not None:
            return self._vob_files

        vob_files = [
            f for f in sorted(self.mount_path.glob('*.[vV][oO][bB]')) if f.stem != 'VIDEO_TS'
        ]

        if not len(vob_files):
            raise FileNotFoundError('IsoFile: No VOBs found!')

        self._vob_files = vob_files

        return self._vob_files

    @property
    def ifo_files(self) -> list[SPath]:
        if self._ifo_files is not None:
            return self._ifo_files

        ifo_files = [
            f for f in sorted(self.mount_path.glob('*.[iI][fF][oO]'))
        ]

        if not len(ifo_files):
            raise FileNotFoundError('IsoFile: No IFOs found!')

        self._ifo_files = ifo_files

        return self._ifo_files

    @abstractmethod
    def _get_mounted_disc(self) -> SPath | None:
        raise NotImplementedError()

    @abstractmethod
    def _mount(self) -> SPath | None:
        raise NotImplementedError()
