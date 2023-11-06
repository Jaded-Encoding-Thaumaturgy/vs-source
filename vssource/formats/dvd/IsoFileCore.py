from __future__ import annotations

import datetime
import json
import os
import warnings
from abc import abstractmethod
from fractions import Fraction
from typing import Sequence, cast

from vstools import CustomValueError, Region, SPath, SPathLike, get_prop, remap_frames, vs

from ...dataclasses import AllNeddedDvdFrameData, D2VIndexFrameData
from ...indexers import D2VWitch, DGIndex, ExternalIndexer
from ...rff import apply_rff_array, apply_rff_video, cut_array_on_ranges
from ...utils import debug_print
from .parsedvd import (
    AUDIO_FORMAT_AC3, AUDIO_FORMAT_LPCM, BLOCK_MODE_FIRST_CELL, BLOCK_MODE_IN_BLOCK, BLOCK_MODE_LAST_CELL, IFO0, IFOX,
    SectorReadHelper, to_json
)
from .title import Title
from .utils import absolute_time_from_timecode, double_check_dvdnav, get_sectorranges_for_vobcellpair

__all__ = [
    'IsoFileCore'
]


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

        # this check seems stupid
        if not (self.iso_path.is_dir() or self.iso_path.is_file()):
            raise CustomValueError('"path" needs to point to a .ISO or a dir root of DVD', str(path), self.__class__)

        ifo0: IFO0 | None = None
        ifos: Sequence[SPathLike | bytes] = []
        if self.use_dvdsrc:
            def _getifo(i: int) -> bytes:
                return cast(bytes, vs.core.dvdsrc2.Ifo(self.iso_path, i))

            _ifo0b = _getifo(0)

            if len(_ifo0b) <= 30:
                warnings.warn('Newer VapourSynth is required for dvdsrc2 information gathering without mounting!')
            else:
                ifo0 = IFO0(SectorReadHelper(_ifo0b))
                ifos = [_getifo(i) for i in range(1, ifo0.num_vts + 1)]

        if not ifo0:
            _ifo0p, *ifos = self.ifo_files
            ifo0 = IFO0(SectorReadHelper(_ifo0p))

        self.ifo0 = ifo0
        self.vts = [IFOX(SectorReadHelper(ifo)) for ifo in ifos]

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
            rawnode = vs.core.dvdsrc2.FullVts(self.iso_path, vts=title_set_nr)
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
            ptt_to_take_for_pgc = len([
                ppt for ppt in target_title[i:] if target_title[i].pgcn == ppt.pgcn
            ])

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

            rawnode = vs.core.dvdsrc2.FullVts(self.iso_path, vts=tt.title_set_nr, ranges=idxx)
            staff = IsoFileCore._dvdsrc2_extract_data(rawnode)

            if not disable_rff:
                rnode = apply_rff_video(rawnode, staff.rff, staff.tff, staff.prog, staff.progseq)
                _vobids = apply_rff_array(staff.vobids, staff.rff, staff.tff, staff.progseq)
            else:
                rnode = rawnode
                _vobids = staff.vobids
            vobids = [(vid, vid) for vid in _vobids]
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

            frameranges = [x for y in [dvddd[a] for a in vobidcellids_to_take] for x in y]

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
                tff = [int((a & 2) >> 1) for a in fflags]
                prog = [int((a & 0b01000000) != 0) for a in fflags]

                rnode = apply_rff_video(node, rff, tff, prog, progseq)
                vobids = apply_rff_array(vobids, rff, tff, progseq)
            else:
                rnode = node

        region = Region.from_framerate(rnode.fps)
        rfps = region.framerate

        if not disable_rff:
            rnode = vs.core.std.AssumeFPS(rnode, fpsnum=rfps.numerator, fpsden=rfps.denominator)
            durationcodes = [Fraction(rfps.denominator, rfps.numerator)] * len(rnode)
            absolutetime = [a * (rfps.denominator / rfps.numerator) for a in range(len(rnode))]
        else:
            if rff_mode == 1:
                durationcodes = timecodes = [Fraction(rfps.denominator * (a + 2), rfps.numerator * 2) for a in rff]
                absolutetime = absolute_time_from_timecode(timecodes)

                def _apply_timecodes(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                    f = f.copy()

                    f.props._DurationNum = timecodes[n].numerator
                    f.props._DurationDen = timecodes[n].denominator
                    f.props._AbsoluteTime = absolutetime[n]

                    return f

                rnode = rnode.std.ModifyFrame(rnode, _apply_timecodes)
            else:
                rffcnt = len([a for a in rff if a])

                asd = (rffcnt * 3 + 2 * (len(rff) - rffcnt)) / len(rff)

                fcc = len(rnode) * 5
                new_fps = Fraction(rfps.numerator * fcc * 2, int(fcc * rfps.denominator * asd),)

                rnode = vs.core.std.AssumeFPS(rnode, fpsnum=new_fps.numerator, fpsden=new_fps.denominator)

                durationcodes = timecodes = [Fraction(rfps.denominator * (a + 2), rfps.numerator * 2) for a in rff]
                absolutetime = absolute_time_from_timecode(timecodes)

        changes = [
            *(i for i, (pvob, nvob) in enumerate(zip(vobids[:-1], vobids[1:]), start=1) if nvob != pvob), len(rnode) - 1
        ]

        assert len(changes) == len(is_chapter)

        last_chapter_i = next((i for i, c in reversed(list(enumerate(is_chapter))) if c), 0)

        output_chapters = list[int]()
        for i in range(len(is_chapter)):
            if not is_chapter[i]:
                continue

            for j in range(i + 1, len(is_chapter)):
                if is_chapter[j]:
                    output_chapters.append(changes[j - 1])
                    break
            else:
                output_chapters.append(changes[last_chapter_i])

        dvnavchapters = double_check_dvdnav(self.iso_path, title_nr)

        if dvnavchapters is not None:  # and (rff_mode == 0 or rff_mode == 2):
            # ???????
            if rfps.denominator == 1001:
                dvnavchapters = [a * 1.001 for a in dvnavchapters]

            adjusted = [absolutetime[i] for i in output_chapters]  # [1:len(output_chapters)-1] ]
            if len(adjusted) != len(dvnavchapters):
                warnings.warn(f'dvdnavchapters length do not match our chapters {len(adjusted)} {len(dvnavchapters)}'
                              ' (open an issue in github)')
                print(adjusted, "\n\n\n", dvnavchapters)
            else:
                framelen = rfps.denominator / rfps.numerator
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

        durationcodesf = list(map(float, durationcodes))

        return Title(
            rnode, output_chapters, changes, self, title_nr, tt.title_set_nr,
            vobidcellids_to_take, dvdsrc_ranges, absolutetime, durationcodesf,
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

    def _d2v_vobid_frameset(self, title_set_nr: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
        _, vobids, _ = self._d2v_collect_all_frameflags(title_set_nr)

        vobidset = dict[tuple[int, int], list[tuple[int, int]]]()
        for i, a in enumerate(vobids):
            if a not in vobidset:
                vobidset[a] = [(i, i - 1)]

            last = vobidset[a][-1]

            if last[1] + 1 == i:
                vobidset[a][-1] = (last[0], last[1] + 1)
                continue

            vobidset[a] += [(i, i)]

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
