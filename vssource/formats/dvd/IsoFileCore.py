from __future__ import annotations

import datetime
import io
from itertools import count
import json
import os
import warnings
import subprocess
from abc import abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import Any, Sequence, SupportsFloat
from vstools import CustomValueError, SPath, remap_frames, set_output, vs, copy_signature

from ...indexers import D2VWitch, DGIndex, ExternalIndexer
from ...rff import apply_rff_array, apply_rff_video, cut_array_on_ranges
from .parsedvd.ifo import IFO0, IFOX, SectorReadHelper, to_json
from .parsedvd.vtsi_mat import AUDIO_FORMAT_LPCM, AUDIO_FORMAT_AC3

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
def double_check_dvdnav(iso: str, title: int) -> list[float] | None:
    try:
        ap = subprocess.check_output(["dvdsrc_dvdnav_title_ptt_test", iso, str(title)])

        return list(map(float, ap.splitlines()))
    except FileNotFoundError:
        ...

    return None


def absolute_time_from_timecode(timecodes: Sequence[SupportsFloat]) -> list[float]:
    absolutetime = list[float]([0.0])

    for i, a in enumerate(timecodes):
        absolutetime.append(absolutetime[i] + float(a))

    return absolutetime


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
    audio: vs.AudioNode | list[vs.AudioNode] | None
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

    def split_at(self, splits: list[int], audio: list[int] | int | None = None) -> tuple[SplitTitle, ...] | SplitTitle:
        output_cnt = SplitHelper._sanitize_splits(self, splits)
        video = SplitHelper.split_video(self, splits)
        chapters = SplitHelper.split_chapters(self, splits)
        audios = None
        if audio is not None:
            if isinstance(audio, int):
                audio = [audio]
            audio_per_output_cnt = len(audio)

            auds = []
            for a in audio:
                auds += [SplitHelper.split_audio(self, splits, a)]

            audios = []

            for i in range(output_cnt):
                lst = []
                for j in range(audio_per_output_cnt):
                    lst += [auds[j][i]]
                audios += [lst]

        fromy = 1
        from_to_s = []
        for j in splits:
            from_to_s += [(fromy, j - 1)]
            fromy = j
        from_to_s += [(fromy, len(self.chapters) - 1)]

        reta = []
        for i in range(output_cnt):
            a = None
            if audios is not None:
                a = audios[i]
                if len(a) == 1:
                    a = a[0]

            reta += [SplitTitle(video[i], a, chapters[i], self, from_to_s[i])]

        if len(reta) == 1:
            return reta[0]
        return tuple(reta)

    def split_ranges(self, split: list[tuple[int, int] | int],
                     audio: list[int] | int | None = None) -> tuple[SplitTitle, ...]:
        assert isinstance(split, list)

        return tuple([self.split_range(s[0], s[1], audio) for s in split])

    def split_range(self, f: int, t: int, audio: list[int] | int | None = None) -> SplitTitle:
        '''
        starting from 1

        from: inclusive
        to: inclusive
        '''
        if t == -1:
            # -1 because last chapter marks the end
            # and not because 1 indexed
            t = len(self.chapters) - 1

        if f == 1 and t == len(self.chapters) - 1:
            return self.split_at([], audio)

        if f == 1:
            return self.split_at([t + 1], audio)[0]

        if t == len(self.chapters) - 1:
            return self.split_at([f], audio)[1]
        return self.split_at([f, t + 1], audio)[1]

    def preview(self, splt=None):
        set_output(self.video(), f"title v {self._title}")
        if splt is not None:
            if not isinstance(splt, tuple):
                splt = [splt]
            for i, a in enumerate(list(splt)):
                set_output(a.video, f"split {i}")
            for i, a in enumerate(list(splt)):
                if isinstance(a.audio, vs.AudioNode):
                    set_output(a.audio, f"split {i}")

    def video(self) -> vs.VideoNode:
        return self.node

    def audio(self, i: int = 0) -> vs.AudioNode:
        self.assert_dvdsrc2()
        asd = self._audios[i]

        if asd.startswith("ac3"):
            anode = vs.core.dvdsrc2.FullVtsAc3(self._core.iso_path, self._vts, i, self._dvdsrc_ranges)
        elif asd.startswith("lpcm"):
            anode = vs.core.dvdsrc2.FullVtsLpcm(self._core.iso_path, self._vts, i, self._dvdsrc_ranges)
        else:
            raise CustomValueError('invalid audio at index', self.__class__)
        anode: vs.AudioNode

        prps = anode.get_frame(0).props
        strt = (prps["Stuff_Start_PTS"] * anode.sample_rate) / 90_000
        endd = (prps["Stuff_End_PTS"] * anode.sample_rate) / 90_000
        debug_print("splice", round((strt / anode.sample_rate) * 1000 * 10) / 10,
                    "ms", round((endd / anode.sample_rate) * 1000 * 10) / 10, "ms")
        strt = int(strt)
        endd = int(endd)
        anode = anode[strt:len(anode) - endd]

        total_dura = (self._absolute_time[-1] + self._duration_times[-1])
        delta = abs(total_dura - anode.num_samples / anode.sample_rate) * 1000

        debug_print(f"delta {delta} ms")

        if delta > 50:
            debug_print(f"WARNING rather big audio/video lenght delta might be indecator that sth is off {delta}")

        return anode

    def assert_dvdsrc2(self):
        if self._dvdsrc_ranges is None or len(self._dvdsrc_ranges) == 0:
            raise CustomValueError("Title needts to be opened with dvdsrc2", __class__)
        if self._core.use_dvdsrc != 2:
            raise CustomValueError("This feature requires dvdsrc2", __class__)

    def dump_ac3(self, a: str, audio_i: int = 0, only_calc_delay: bool = False) -> float:
        self.assert_dvdsrc2()
        if not self._audios[audio_i].startswith("ac3"):
            raise CustomValueError("autio at {audio_i} is not ac3", __class__)

        nd = vs.core.dvdsrc2.RawAc3(self._core.iso_path, self._vts, audio_i, self._dvdsrc_ranges)
        p0 = nd.get_frame(0).props

        # print(p0["Stuff_Start_PTS"])
        # print(p0["Stuff_End_PTS"])

        if not only_calc_delay:
            wrt = open(a, "wb")
            for f in nd.frames():
                wrt.write(bytes(f[0]))
            wrt.close()

        return float(p0["Stuff_Start_PTS"]) / 90_000

    def __repr__(self) -> str:
        chapters = self.chapters + [len(self.node) - 1]
        chapter_lengths = [self._absolute_time[chapters[i + 1]] - self._absolute_time[chapters[i]]
                           for i in range(len(self.chapters))]

        chapter_lengths = [str(datetime.timedelta(seconds=x)) for x in chapter_lengths]
        timestrings = [str(datetime.timedelta(seconds=self._absolute_time[x])) for x in self.chapters]

        to_print = "Chapters:\n"
        for i in range(len(timestrings)):
            to_print += "{:02} {:015} {:015} {}".format(i + 1, timestrings[i], chapter_lengths[i], self.chapters[i])

            if i == 0:
                to_print += " (faked)"

            if self._patched_end_chapter is not None and i == len(timestrings) - 1:
                delta = self.chapters[i] - self._patched_end_chapter
                to_print += f" (originally {self._patched_end_chapter} delta {delta})"

            to_print += "\n"

        to_print += "\n"
        to_print += f"cellchange: {self.cell_changes}\n"
        to_print += "\n"
        to_print += "Audios: (fz)\n"
        for i, a in enumerate(self._audios):
            to_print += "{} {}\n".format(i, a)

        return to_print.strip()


class SplitHelper:
    def split_range_ac3(title: Title, f: int, t: int, audio_i: int, outfile: str) -> float:
        nd = vs.core.dvdsrc2.RawAc3(title._core.iso_path, title._vts, audio_i, title._dvdsrc_ranges)
        prps = nd.get_frame(0).props

        strt = prps["Stuff_Start_PTS"]
        endd = prps["Stuff_End_PTS"]
        debug_print(f"Stuff_Start_PTS pts {strt} Stuff_End_PTS {endd}")
        raw_start = (title._absolute_time[title.chapters[f - 1]] * 90_000)
        raw_end = ((title._absolute_time[title.chapters[t]] + title._duration_times[title.chapters[t]]) * 90_000)

        start_pts = raw_start + strt
        end_pts = start_pts + (raw_end - raw_start)

        audio_offset_pts = 0

        outf = open(outfile, "wb")
        debug_print(f"start_pts  {start_pts} end_pts {end_pts}")

        i = start_pts // 2880

        debug_print("first ", i, len(nd))

        while i < len(nd):
            pkt_start_pts = i * 2880
            pkt_end_pts = (i + 1) * 2880

            assert pkt_end_pts > start_pts

            if pkt_start_pts < start_pts:
                audio_offset_pts = start_pts - pkt_start_pts
            outf.write(bytes(nd.get_frame(i)[0]))

            if pkt_end_pts > end_pts:
                break

            i += 1
        debug_print("wrote", (i - (start_pts // 2880)))
        debug_print("offset is", (audio_offset_pts) / 90, "ms")
        return audio_offset_pts / 90_000

    def split_chapters(title: Title, splits: list[int]) -> tuple[list[int]]:
        out = []

        rebase = title.chapters[0]  # normally 0
        chaps = []

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

    def split_video(title: Title, splits: list[int]) -> tuple[vs.VideoNode, ...]:
        reta = SplitHelper._cut_split(title, splits, title.node, SplitHelper._cut_fz_v)
        assert len(reta) == len(splits) + 1
        return reta

    def split_audio(title: Title, splits: list[int], i: int = 0) -> tuple[vs.AudioNode, ...]:
        reta = SplitHelper._cut_split(title, splits, title.audio(i), SplitHelper._cut_fz_a)
        assert len(reta) == len(splits) + 1
        return reta

    def _sanitize_splits(title: Title, splits: list[int]):
        # assert len(splits) >= 1
        assert isinstance(splits, list)
        lasta = -1
        for a in splits:
            assert isinstance(a, int)
            assert a > lasta
            assert a <= len(title.chapters)
            lasta = a
        return len(splits) + 1

    def _cut_split(title: Title, splits: list[int], a, b) -> tuple[vs.VideoNode, ...]:
        out = []
        last = 0
        for s in splits:
            index = s - 1
            out += [b(title, a, last, index)]
            last = index
        out += [b(title, a, last, len(title.chapters) - 1)]

        return tuple(out)

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

    def _cut_fz_a(title: Title, anode: vs.AudioNode, f: int, t: int) -> vs.AudioNode:
        ft = [f, t]

        ft = [title.chapters[i] for i in ft]
        ft = [title._absolute_time[i] for i in ft]
        ft = [i * anode.sample_rate for i in ft]
        ft = [round(i) for i in ft]
        ft = [min(i, anode.num_samples) for i in ft]

        f, t = ft[0], ft[1]
        return anode[f:t]


class IsoFileCore:
    _subfolder = "VIDEO_TS"
    ifo0: IFO0
    vts: list[IFOX]

    def __init__(
        self, path: SPath | str,
        use_dvdsrc: None | int = None,
        indexer: ExternalIndexer | type[ExternalIndexer] = None,
    ):
        '''
        Only external indexer supported D2VWitch and DGIndex

        indexer only used if use_dvdsrc == 0

        '''
        self.force_root = False
        self.output_folder = "/tmp" if os.name != "nt" else "C:/tmp"

        if indexer is None:
            indexer = DGIndex() if os.name == "nt" else D2VWitch()

        self._mount_path: SPath | None = None
        self._vob_files: list[SPath] | None = None
        self._ifo_files: list[SPath] | None = None

        self.has_dvdsrc1 = hasattr(vs.core, "dvdsrc")
        self.has_dvdsrc2 = hasattr(vs.core, "dvdsrc2")
        self.use_dvdsrc = use_dvdsrc

        if self.use_dvdsrc is None:
            if self.has_dvdsrc2:
                self.use_dvdsrc = 2
            elif self.has_dvdsrc1:
                self.use_dvdsrc = 1
            else:
                self.use_dvdsrc = 0

        if isinstance(self.use_dvdsrc, bool):
            self.use_dvdsrc = 2 if self.use_dvdsrc else 0

        if use_dvdsrc == 1:
            assert self.has_dvdsrc1
        if use_dvdsrc == 2:
            assert self.has_dvdsrc2

        self.iso_path = SPath(path).absolute()

        if not self.iso_path.is_dir() and not self.iso_path.is_file():
            raise CustomValueError('"path" needs to point to a .ISO or a dir root of DVD', path, self.__class__)

        fallback_ifostuff = False

        if self.has_dvdsrc2:
            i0bytes = vs.core.dvdsrc2.Ifo(self.iso_path, 0)
            if len(i0bytes) <= 30:
                fallback_ifostuff = True
                print('newer Vapoursynth is required for dvdsrc2 information gathering without mounting')
            else:
                self.ifo0 = IFO0(SectorReadHelper(io.BufferedReader(io.BytesIO(i0bytes))))
                self.vts = []

                for i in range(1, self.ifo0.num_vts + 1):
                    rh = SectorReadHelper(io.BufferedReader(io.BytesIO(vs.core.dvdsrc2.Ifo(self.iso_path, i))))
                    self.vts += [IFOX(rh)]

        if not self.has_dvdsrc2 or fallback_ifostuff:
            for i, a in enumerate(self.ifo_files):
                if i == 0:
                    self.ifo0 = IFO0(a)
                    self.vts = []
                else:
                    self.vts += [IFOX(a)]

        self.json = to_json(self.ifo0, self.vts)

        if not self.has_dvdsrc1:
            debug_print("Does not have dvdsrc cant double check json with libdvdread")
        else:
            dvdsrc_json = json.loads(vs.core.dvdsrc.Json(self.iso_path))
            try:
                del dvdsrc_json["dvdpath"]
                del dvdsrc_json["current_vts"]
                del dvdsrc_json["current_domain"]

                for ifo in dvdsrc_json["ifos"]:
                    del ifo["pgci_ut"]
                    ifo["pgci_ut"] = []
            except KeyError:
                pass

            ja = json.dumps(dvdsrc_json, sort_keys=True)
            jb = json.dumps(self.json, sort_keys=True)

            if ja != jb:
                warnings.warn(f"libdvdread json does not match python implentation\n"
                              f"json a,b have been written to {self.output_folder}")
                open(os.path.join(self.output_folder, "a.json"), "wt").write(ja)
                open(os.path.join(self.output_folder, "b.json"), "wt").write(jb)

        if self.use_dvdsrc == 0:
            self.indexer = indexer if isinstance(indexer, ExternalIndexer) else indexer()

        self.title_count = len(self.ifo0.tt_srpt)

    def get_vts(
        self,
        title_set_nr: int = 1,
        d2v_our_rff: bool = False,
    ) -> vs.VideoNode:
        """
        Gets a full vts.
        only works with dvdsrc2 and d2vsource usese our rff for dvdsrc and d2source rff for d2vsource

        mainly useful for debugging and checking if our rff algorithm is good

        """
        assert self.use_dvdsrc in [0, 2]
        if self.use_dvdsrc == 0:
            vob_input_files = self._get_title_vob_files_for_vts(title_set_nr)
            index_file = self.indexer.index(vob_input_files, output_folder=self.output_folder)[0]

            if d2v_our_rff:
                rawnode = vs.core.dvdsrc2.FullVts(self.iso_path, vts=title_set_nr)
                staff = IsoFileCore._dvdsrc2_extract_data(rawnode)
                return apply_rff_video(self.indexer._source_func(index_file, rff=False),
                                       staff.rff,
                                       staff.tff,
                                       staff.prog,
                                       staff.progseq)
            else:
                return self.indexer._source_func(index_file, rff=True)
        else:
            rawnode = vs.core.dvdsrc2.FullVts(self.iso_path, vts=title_set_nr)
            staff = IsoFileCore._dvdsrc2_extract_data(rawnode)
            return apply_rff_video(rawnode, staff.rff, staff.tff, staff.prog, staff.progseq)

    def get_title(
        self,
        title_nr: int = 1,
        angle_nr: int | None = None,
        rff_mode: int = 0,
    ) -> Title:
        """
        Gets a title.

        :param title_nr:            title nr starting from 1
        :param angle_nr:            starting from 1
        :param rff_mode:            0 apply rff soft telecine (default)
                                    1 calculate per frame durations based on rff
                                    2 set average fps on global clip
        """
        disable_rff = rff_mode >= 1

        tt_srpt = self.ifo0.tt_srpt
        title_idx = title_nr - 1
        if title_idx < 0 or title_idx >= len(tt_srpt):
            raise CustomValueError('"title_nr" out of range', self.__class__)
        tt = tt_srpt[title_idx]

        if tt.nr_of_angles != 1 and angle_nr is None:
            raise CustomValueError('no angle_nr given for multi angle title', self.__class__)

        target_vts = self.vts[tt.title_set_nr - 1]
        target_title = target_vts.vts_ptt_srpt[tt.vts_ttn - 1]

        assert len(target_title) == tt.nr_of_ptts

        for ptt in target_title[1:]:
            if ptt.pgcn != target_title[0].pgcn:
                raise CustomValueError('title is not one program chain (unsupported currently)', self.__class__)

        pgc_i = target_title[0].pgcn - 1
        title_programs = [a.pgn for a in target_title]
        targte_pgc = target_vts.vts_pgci.pgcs[pgc_i]
        pgc_programs = targte_pgc.program_map

        if title_programs[0] != 1 or pgc_programs[0] != 1:
            warnings.warn('WARNING Open Title does not start at the first cell (open issue in github with sample)\n')

        target_programs = [a[1] for a in list(filter(lambda x: (x[0] + 1) in title_programs, enumerate(pgc_programs)))]

        if target_programs != pgc_programs:
            warnings.warn('WARNING Open the program chain does not include all ptts\n')

        vobidcellids_to_take = []
        current_angle = 1
        angle_start_cell_i: int = None

        is_chapter = []
        for cell_i in range(len(targte_pgc.cell_position)):
            cell_position = targte_pgc.cell_position[cell_i]
            cell_playback = targte_pgc.cell_playback[cell_i]

            block_mode = cell_playback.block_mode

            if block_mode == 1:  # BLOCK_MODE_FIRST_CELL
                current_angle = 1
                angle_start_cell_i = cell_i
            elif block_mode == 2 or block_mode == 3:  # BLOCK_MODE_IN_BLOCK and BLOCK_MODE_LAST_CELL
                current_angle += 1

            if block_mode == 0:
                take_cell = True
                angle_start_cell_i = cell_i
            else:
                take_cell = current_angle == angle_nr

            if take_cell:
                vobidcellids_to_take += [(cell_position.vob_id_nr, cell_position.cell_nr)]
                is_chapter += [(angle_start_cell_i + 1) in target_programs]

        assert len(is_chapter) == len(vobidcellids_to_take)

        # should set rnode, vobids and rff, dvdsrc_ranges
        if self.use_dvdsrc == 1:
            dvdsrc_ranges = []
            if not hasattr(vs.core, "dvdsrc"):
                raise CustomValueError('For dvdsrc only features dvdsrc plugin needs to be installed', __file__)
            try:
                import pydvdsrc
            except ImportError:
                raise CustomValueError('For dvdsrc only features pydvdsrc python file needs to be installed', __file__)

            sectors = pydvdsrc.get_sectors_from_vobids(target_vts, vobidcellids_to_take)
            rawnode = vs.core.dvdsrc.FullM2V(self.iso_path, vts=tt.title_set_nr, domain=1, sectors=sectors)
            exa = pydvdsrc.DVDSRCM2vInfoExtracter(rawnode)
            rff = exa.rff

            if not disable_rff:
                rnode = apply_rff_video(rawnode, exa.rff, exa.tff, exa.prog, exa.prog_seq)
                vobids = apply_rff_array(exa.vobid, exa.rff, exa.tff, exa.prog_seq)
            else:
                rnode = rawnode
                vobids = exa.vobid
        elif self.use_dvdsrc == 2:
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

            rawnode = vs.core.dvdsrc2.FullVts(self.iso_path, vts=tt.title_set_nr, ranges=idxx)
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
                    'This usually means outdated/unpatched D2Vwitch', self.__class__)

            frameranges = []
            for a in vobidcellids_to_take:
                frameranges += dvddd[a]

            fflags, vobids, progseq = self._d2v_collect_all_frameflags(tt.title_set_nr)

            index_file = self.indexer.index(vob_input_files, output_folder=self.output_folder)[0]
            node = self.indexer._source_func(index_file, rff=False)
            # node = self.indexer.source(vob_input_files, output_folder=self.output_folder, rff=False)
            assert len(node) == len(fflags)

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

                def apply_timecode(n: int, f: vs.VideoFrame, timecodes, absolutetime) -> vs.VideoFrame:
                    fout = f.copy()
                    fout.props["_DurationNum"] = timecodes[n].numerator
                    fout.props["_DurationDen"] = timecodes[n].denominator
                    fout.props["_AbsoluteTime"] = absolutetime[n]
                    return fout
                rnode = vs.core.std.ModifyFrame(rnode, [rnode], partial(apply_timecode,
                                                                        timecodes=timecodes,
                                                                        absolutetime=absolutetime))
            else:
                rffcnt = 0
                for a in rff:
                    if a:
                        rffcnt += 1

                asd = (rffcnt * 3 + 2 * (len(rff) - rffcnt)) / len(rff)

                fcc = len(rnode) * 5
                new_fps = Fraction(fpsnum * fcc * 2, int(fcc * fpsden * asd),)

                rnode = vs.core.std.AssumeFPS(rnode, fpsnum=new_fps.numerator, fpsden=new_fps.denominator)

                # timecodes = [float(1.0 / rnode.fps) for _ in range(len(rnode))]
                # durationcodes = timecodes
                # absolutetime = absolute_time_from_timecode(timecodes)
                # stiff do timecodes like rff1 because it would break otherwise
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
                print(adjusted)
                print(dvnavchapters)
            else:
                framelen = fpsden / fpsnum
                for i in range(len(adjusted)):
                    # tolerance no idea why so big
                    # on hard telecine ntcs it matches up almost perfectly
                    # but on ~24p pal rffd it does not lol
                    if abs(adjusted[i] - dvnavchapters[i]) > framelen * 20:
                        warnings.warn(
                            f'dvdnavchapters length do not match our chapters {len(adjusted)} {len(dvnavchapters)}'
                            ' (open an issue in github)')
                        print(adjusted)
                        print(dvnavchapters)
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
        for i, a in enumerate(targte_pgc.audio_control):
            if a.available:
                audo = target_vts.vtsi_mat.vts_audio_attr[i]

                if audo.audio_format == AUDIO_FORMAT_AC3:
                    aformat = "ac3"
                elif audo.audio_format == AUDIO_FORMAT_LPCM:
                    aformat = "lpcm"
                else:
                    aformat = "unk"

                audios += [f'{aformat}({audo.language})']
            else:
                audios += ["none"]

        return Title(
            rnode, output_chapters, changes, self, title_nr, tt.title_set_nr,
            vobidcellids_to_take, dvdsrc_ranges, absolutetime, durationcodes,
            audios, patched_end_chapter
        )

    def _d2v_collect_all_frameflags(self, title_set_nr: int) -> Sequence[int]:
        files = self._get_title_vob_files_for_vts(title_set_nr)
        index_file = self.indexer.index(files, output_folder=self.output_folder)[0]
        index_info = self.indexer.get_info(index_file)

        frameflagslst = []
        vobidlst = []
        progseqlst = []
        for iframe in index_info.frame_data:
            vobcell = (iframe.vob, iframe.cell)

            progseq = int(((iframe.info & 0b1000000000) != 0))

            for a in iframe.frameflags:
                if a != 0xFF:
                    frameflagslst += [a]
                    vobidlst += [vobcell]
                    progseqlst += [progseq]

        return frameflagslst, vobidlst, progseqlst

    def _d2v_vobid_frameset(self, title_set_nr: int) -> dict:
        a = self._d2v_collect_all_frameflags(title_set_nr)
        vobid = a[1]

        vobidset = dict()
        for i, a in enumerate(vobid):
            if a not in vobidset:
                vobidset[a] = [[i, i - 1]]
            latest = vobidset[a][-1]
            if latest[1] + 1 == i:
                latest[1] += 1
            else:
                vobidset[a] += [[i, i]]

        return vobidset

    def _get_title_vob_files_for_vts(self, vts: int) -> Sequence[SPath]:
        f1 = self.vob_files
        f1 = list(filter(lambda x: (("VTS_{:02}_".format(vts)) in str(x)), f1))
        f1 = list(filter(lambda x: (not str(x).upper().endswith("0.VOB")), f1))
        return f1

    def _mount_folder_path(self) -> SPath:
        if self.force_root:
            return self.iso_path

        if self.iso_path.name.upper() == self._subfolder:
            self.iso_path = self.iso_path.parent

        return self.iso_path / self._subfolder

    def _dvdsrc2_extract_data(rawnode: vs.VideoNode) -> AllNeddedDvdFrameData:
        dd = bytes(rawnode.get_frame(0).props["InfoFrame"][0])
        assert len(dd) == len(rawnode) * 4

        vobids = []
        tff = []
        rff = []
        prog = []
        progseq = []
        for i in range(0, len(rawnode)):
            sb = dd[i * 4 + 0]
            tff += [(sb & (1 << 0)) >> 0]
            rff += [(sb & (1 << 1)) >> 1]
            prog += [(sb & (1 << 2)) >> 2]
            progseq += [(sb & (1 << 3)) >> 3]

            vobids += [((dd[i * 4 + 1] << 8) + dd[i * 4 + 2], dd[i * 4 + 3])]
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


def get_sectorranges_for_vobcellpair(current_vts: IFOX, pair_id: tuple[int, int]) -> list[tuple[int, int]]:
    ranges = []
    for e in current_vts.vts_c_adt.cell_adr_table:
        if e.vob_id == pair_id[0] and e.cell_id == pair_id[1]:
            ranges += [(e.start_sector, e.last_sector)]
    return ranges
