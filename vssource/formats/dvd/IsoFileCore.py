from __future__ import annotations
from dataclasses import dataclass
import os
import datetime
from abc import abstractmethod
from fractions import Fraction
from typing import List, Sequence, Tuple

from vstools import CustomValueError, SPath, vs, set_output

from ...indexers import D2VWitch, DGIndex, ExternalIndexer
import json
from functools import partial
from ...rff import apply_rff_array, apply_rff_video, cut_array_on_ranges, cut_node_on_ranges
from ...a52 import a52_syncinfo
from .parsedvd.ifo import IFO0, IFOX
import jsondiff
import hashlib


__all__ = [
    'IsoFileCore', 'Title'
]


# d2vwitch needs this patch applied
# https://gist.github.com/jsaowji/ead18b4f1b90381d558eddaf0336164b

def assert_dvdsrc():
    if not hasattr(vs.core, "dvdsrc"):
        raise CustomValueError('For dvdsrc only features dvdsrc plugin needs to be installed', __file__)
    try:
        import pydvdsrc
    except ImportError:
        raise CustomValueError('For dvdsrc only features pydvdsrc python file needs to be installed', __file__)


# https://gist.github.com/jsaowji/2bbf9c776a3226d1272e93bb245f7538
def double_check_dvdnav(iso: str, title: int):
    try:
        import subprocess
        ap = subprocess.check_output(["dvdsrc_dvdnav_title_ptt_test", iso, str(title)])
        lns = ap.splitlines()
        flts = [float(a) for a in lns]

        return flts
    except FileNotFoundError:
        return None


def absolute_time_from_timecode(timecodes):
    absolutetime = []
    for i, a in enumerate(timecodes):
        if i == 0:
            absolutetime += [0.0]
        else:
            absolutetime += [absolutetime[i - 1] + float(a)]
    return absolutetime


@dataclass
class Title:
    node: vs.VideoNode
    chapters: List[int]
    _core: IsoFileCore
    _title: int
    _vts: int
    _vobidcellids_to_take: List[Tuple[int, int]]
    _absolute_time: List[float]
    _audios: List[str]
    _splits: List[int] | None
    _patched_end_chapter: int | None

    def preview_output_split(self, audio: int | None = None):
        set_output(self.video(), f"title {self._title}")
        for i, s in enumerate(self.split_video()):
            set_output(s, f"split {i+1}")

        if audio is not None:
            set_output(list(self.split_audio(audio)), "audio")

    def video(self) -> vs.VideoNode:
        return self.node

    def audio(self, i: int = 0) -> vs.AudioNode:
        assert_dvdsrc()
        import pydvdsrc

        asd = self._audios[i]

        target_vts = self._core.json["ifos"][self._vts]
        sectors = pydvdsrc.get_sectors_from_vobids(target_vts, self._vobidcellids_to_take)

        if asd.startswith("ac3"):
            return vs.core.dvdsrc.FullAC3(self._core.iso_path, self._vts, 1, sectors, i)
        elif asd.startswith("lpcm"):
            return vs.core.dvdsrc.FullLPCM(self._core.iso_path, self._vts, 1, sectors, i)
        else:
            raise CustomValueError('invalid audio at index', self.__class__)

    def split_video(self) -> Tuple[vs.VideoNode, ...]:
        return self._split(self.node, self._cut_fz_v)

    def split_audio(self, i: int = 0) -> Tuple[vs.AudioNode, ...]:
        return self._split(self.audio(i), self._cut_fz_a)

    def split_ac3(self, i: int = 0) -> Tuple[Tuple[str, float]]:
        m = hashlib.sha256()
        m.update(str(self._core.iso_path).encode("utf-8"))
        m.update(str(self._vobidcellids_to_take).encode("utf-8"))
        m.update(str(self._vts).encode("utf-8"))
        m.update(str(i).encode("utf-8"))
        nn = m.hexdigest()
        nn = os.path.join(self._core.output_folder, f"{nn}.ac3")
        if not os.path.exists(nn):
            self.dump_ac3(nn, i)
        bb = open(nn, "rb")

        sr0 = None
        buffer = bytearray()

        ac3_sample_per_frame = 6 * 256
        current_frame = 0

        split_times = []
        for a in self._splits:
            split_times += [self._absolute_time[self.chapters[a - 1]]]

        split_samples = None
        current_split = 0
        file_template = "{}.ac3"

        last_frame_bytes = bytes()
        files = []
        sample_offsets = [0]

        files += [os.path.join(self._core.output_folder, file_template.format(current_split))]
        current_file = open(files[0], "wb")

        while True:
            byte = bb.read(8192)
            buffer += byte
            while True:
                ret = a52_syncinfo(buffer)
                if sr0 is None:
                    sr0 = ret.sample_rate
                    split_samples = [a * sr0 for a in split_times]
                else:
                    assert ret.sample_rate == sr0

                if len(buffer) <= ret.data_size:
                    break
                else:
                    current_frame_bytes = bytes(buffer[0:ret.data_size])

                    sample_start = current_frame * ac3_sample_per_frame
                    sample_end = sample_start + ac3_sample_per_frame
                    current_file.write(current_frame_bytes)

                    if current_split < len(split_samples) and sample_end >= split_samples[current_split]:
                        current_split += 1
                        fp = os.path.join(self._core.output_folder, file_template.format(current_split))
                        current_file = open(fp, "wb")
                        current_file.write(last_frame_bytes)
                        current_file.write(current_frame_bytes)
                        sample_offsets += [round(ac3_sample_per_frame + (sample_end
                                                 - split_samples[current_split - 1]))]
                        files += [fp]

                    buffer = buffer[ret.data_size:]
                    last_frame_bytes = current_frame_bytes
                    current_frame += 1

            if len(byte) < 8192:
                break
        assert len(sample_offsets) == len(files)
        time_offsets = [a / sr0 for a in sample_offsets]

        return [(files[i], time_offsets[i]) for i in range(len(files))]

    def dump_ac3(self, a: str, audio_i: int = 0):
        if not self._audios[audio_i].startswith("ac3"):
            raise CustomValueError(f"autio at {audio_i} is not ac3", __class__)

        if self._core.has_dvdsrc:
            assert_dvdsrc()
            import pydvdsrc

            wrt = open(a, "wb")

            target_vts = self._core.json["ifos"][self._vts]
            sectors = pydvdsrc.get_sectors_from_vobids(target_vts, self._vobidcellids_to_take)

            nd = vs.core.dvdsrc.RawAc3(self._core.iso_path, self._vts, 1, sectors, audio_i)
            for f in nd.frames():
                wrt.write(bytes(f[0]))
            wrt.close()
        else:
            output_file = open(a, "wb")
            files = self._core._get_title_vob_files_for_vts(self._vts)
            # sizes = [os.stat(f).st_size for f in files]

            ranges = []
            for c in self._vobidcellids_to_take:
                ranges += get_sectorranges_for_vobcellpair(self._core.json["ifos"][self._vts], c)

            files = [open(f, "rb") for f in files]
            sector = 0
            file_i = 0

            buffer = bytearray()

            dvd_sector_size = 2048

            while True:
                if file_i >= len(files):
                    break
                result = files[file_i].read(dvd_sector_size)

                if len(result) != dvd_sector_size:
                    assert len(result) == 0
                    file_i += 1
                    continue
                else:
                    sectorin = False
                    for r in ranges:
                        if sector >= r[0] and sector <= r[1]:
                            sectorin = True
                            break

                    if sectorin:
                        buffer += result
                    sector += 1

                start_code = 4
                start_code_plus_len = start_code + 2
                pack_len = 10
                while True:
                    if len(buffer) < start_code:
                        break
                    assert buffer[0] == 0
                    assert buffer[1] == 0
                    assert buffer[2] == 1
                    st = buffer[3]
                    if st == 0xBA:
                        if len(buffer) < start_code + pack_len:
                            break
                        buffer = buffer[start_code + pack_len:]
                    else:
                        assert st in [0xBB, 0xBE, 0xE0, 0xBF, 0xBD]

                        if len(buffer) < start_code_plus_len:
                            break

                        leny = (buffer[4] << 8) + buffer[5]

                        if len(buffer) < start_code_plus_len + leny:
                            break
                        if st == 0xBD:
                            buf = buffer[6:6 + leny]
                            hdr_data_len = buf[2]
                            inner_data = buf[3 + hdr_data_len:]
                            inner_id = inner_data[0]
                            if inner_id >= 0x80 and inner_id <= 0x87:
                                idx = inner_id - 0x80
                                if idx == audio_i:
                                    output_file.write(inner_data[1 + 3:])
                        buffer = buffer[start_code_plus_len + leny:]

    def update_splits(self, splits: List[int]):
        self._splits = splits
        self._sanitize_splits(self._splits)

    def _sanitize_splits(self, splits):
        #        assert len(splits) >= 1
        assert isinstance(splits, list)
        lasta = -1
        for a in splits:
            assert isinstance(a, int)
            assert a > lasta
            assert a <= len(self.chapters)
            lasta = a
        return len(splits) + 1

    def _split(self, a, b) -> Tuple[vs.VideoNode, ...]:
        out = []
        last = 0
        for s in self._splits:
            index = s - 1
            out += [b(a, last, index)]
            last = index
        out += [b(a, last, len(self.chapters) - 1)]

        return tuple(out)

    # starting 0
    # end inclusive
    #  0 0 -> chapter 0
    def _cut_fz_v(self, vnode: vs.VideoNode, f: int, t: int) -> vs.VideoNode:
        if f < 0 or f >= len(self.chapters):
            raise CustomValueError('invalid from', self.__class__)
        if t < -1 or t >= len(self.chapters):
            raise CustomValueError('invalid to', self.__class__)
        f = self.chapters[f]
        t = self.chapters[t]
        return vnode[f:t + 1]

    def _cut_fz_a(self, anode: vs.AudioNode, f: int, t: int) -> vs.AudioNode:
        f = self.chapters[f]
        t = self.chapters[t]

        f = self._absolute_time[f]
        t = self._absolute_time[t]

        f *= anode.sample_rate
        t *= anode.sample_rate

        f = max(f, anode.num_channels - 1)
        t = max(t, anode.num_channels - 1)

        f = round(f)
        t = round(t)

        return anode[f:t + 1]

    def __repr__(self) -> str:
        chapters = self.chapters + [len(self.node) - 1]
        chapter_legnths = [self._absolute_time[chapters[i + 1]] - self._absolute_time[chapters[i]]
                           for i in range(len(self.chapters))]

        chapter_legnths = [str(datetime.timedelta(seconds=x)) for x in chapter_legnths]
        timestrings = [str(datetime.timedelta(seconds=self._absolute_time[x])) for x in self.chapters]

        to_print = "Chapters: (fz)\n"
        for i in range(len(timestrings)):
            to_print += "{:02} {:015} {:015} {}".format(i, timestrings[i], chapter_legnths[i], self.chapters[i])

            if i == 0:
                to_print += " (faked)"

            if self._patched_end_chapter is not None and i == len(timestrings) - 1:
                delta = self.chapters[i] - self._patched_end_chapter
                to_print += f" (originally {self._patched_end_chapter} delta {delta})"
            to_print += "\n"

        to_print += "Audios: (fz)\n"
        for i, a in enumerate(self._audios):
            to_print += "{} {}\n".format(i, a)

        return to_print.strip()


class IsoFileCore:
    _subfolder = "VIDEO_TS"

    def __init__(
        self, path: SPath | str,
        use_dvdsrc=None,
        indexer: ExternalIndexer | type[ExternalIndexer] = None,
    ):
        '''
        Only external indexer supported D2VWitch and DGIndex

        indexer only used if use_dvdsrc == False

        '''
        self.force_root = False
        self.output_folder = "/tmp" if os.name != "nt" else "C:/tmp"

        if indexer is None:
            indexer = DGIndex() if os.name == "nt" else D2VWitch()

        self._mount_path: SPath | None = None
        self._vob_files: list[SPath] | None = None
        self._ifo_files: list[SPath] | None = None

        self.has_dvdsrc = hasattr(vs.core, "dvdsrc")

        if use_dvdsrc is None:
            use_dvdsrc = self.has_dvdsrc

        if not self.has_dvdsrc and use_dvdsrc:
            use_dvdsrc = False
            print("Requested dvdsrc but not installed")

        self.use_dvdsrc = use_dvdsrc
        self.iso_path = SPath(path).absolute()

        if not self.iso_path.is_dir() and not self.iso_path.is_file():
            raise CustomValueError('"path" needs to point to a .ISO or a dir root of DVD', path, self.__class__)

        if self.use_dvdsrc:
            self.json = json.loads(vs.core.dvdsrc.Json(self.iso_path))
        else:
            self.json = {"ifos": []}
            for i, a in enumerate(self.ifo_files):
                if i == 0:
                    self.json["ifos"] += [IFO0(a).crnt]
                else:
                    self.json["ifos"] += [IFOX(a).crnt]
            if not self.has_dvdsrc:
                print("Does not have dvdsrc cant double check json with libdvdread")
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
                    print(f"libdvdread json does not match python json a,b have been written to {self.output_folder}")
                    open(os.path.join(self.output_folder, "a.json"), "wt").write(ja)
                    open(os.path.join(self.output_folder, "b.json"), "wt").write(jb)

            self.indexer = indexer if isinstance(indexer, ExternalIndexer) else indexer()

        self.title_count = len(self.json["ifos"][0]["tt_srpt"])

    def get_title(
        self,
        title_nr: int = 1,
        splits: List[int] | None = None,
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

        tt_srpt = self.json["ifos"][0]["tt_srpt"]
        title_idx = title_nr - 1
        if title_idx < 0 or title_idx >= len(tt_srpt):
            raise CustomValueError('"title_nr" out of range', self.__class__)
        tt = tt_srpt[title_idx]

        if tt["nr_of_angles"] != 1 and angle_nr is None:
            raise CustomValueError('no angle_nr given for multi angle title', self.__class__)

        title_set_nr = tt["title_set_nr"]
        vts_ttn = tt["vts_ttn"]

        target_vts = self.json["ifos"][title_set_nr]
        target_title = target_vts["vts_ptt_srpt"][vts_ttn - 1]

        assert len(target_title) == tt["nr_of_ptts"]

        for ptt in target_title[1:]:
            if ptt["pgcn"] != target_title[0]["pgcn"]:
                raise CustomValueError('title is not one program chain (unsupported currently)', self.__class__)

        pgc_i = target_title[0]["pgcn"] - 1
        title_programs = [a["pgn"] for a in target_title]
        targte_pgc = target_vts["vts_pgcit"][pgc_i]
        pgc_programs = targte_pgc["program_map"]

        if title_programs[0] != 1 or pgc_programs[0] != 1:
            print("Title does not start at the first cell")

        target_programs = [a[1] for a in list(filter(lambda x: (x[0] + 1) in title_programs, enumerate(pgc_programs)))]

        if target_programs != pgc_programs:
            print("The program chain does not include all ptt's")

        vobidcellids_to_take = []
        current_angle = 1
        angle_start_cell_i: int = None

        is_chapter = []
        for cell_i in range(len(targte_pgc["cell_position"])):
            cell_position = targte_pgc["cell_position"][cell_i]
            cell_playback = targte_pgc["cell_playback"][cell_i]

            block_mode = cell_playback["block_mode"]

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
                vobidcellids_to_take += [(cell_position["vob_id_nr"], cell_position["cell_nr"])]
                is_chapter += [(angle_start_cell_i + 1) in target_programs]

        assert len(is_chapter) == len(vobidcellids_to_take)

        # should set rnode, vobids and rff
        if self.use_dvdsrc:
            assert_dvdsrc()
            import pydvdsrc

            sectors = pydvdsrc.get_sectors_from_vobids(target_vts, vobidcellids_to_take)
            rawnode = vs.core.dvdsrc.FullM2V(self.iso_path, vts=title_set_nr, domain=1, sectors=sectors)
            exa = pydvdsrc.DVDSRCM2vInfoExtracter(rawnode)
            rff = exa.rff

            if not disable_rff:
                rnode = apply_rff_video(rawnode, exa.rff, exa.tff)
                vobids = apply_rff_array(exa.rff, exa.vobid)
            else:
                rnode = rawnode
                vobids = exa.vobid
        else:
            vob_input_files = self._get_title_vob_files_for_vts(title_set_nr)
            dvddd = self._d2v_vobid_frameset(title_set_nr)

            if len(dvddd.keys()) == 1 and (0, 0) in dvddd.keys():
                raise CustomValueError(
                    'Youre indexer created a d2v file with only zeros for vobid cellid; This usually means outdated/unpatched D2Vwitch', self.__class__)

            frameranges = []
            for a in vobidcellids_to_take:
                frameranges += dvddd[a]

            fflags, vobids = self._d2v_collect_all_frameflags(title_set_nr)

            index_file = self.indexer.index(vob_input_files, output_folder=self.output_folder)[0]
            node = self.indexer._source_func(index_file, rff=False)
            # node = self.indexer.source(vob_input_files, output_folder=self.output_folder, rff=False)
            assert len(node) == len(fflags)

            fflags = cut_array_on_ranges(fflags, frameranges)
            vobids = cut_array_on_ranges(vobids, frameranges)
            node = cut_node_on_ranges(node, frameranges)

            rff = [(a & 1) for a in fflags]

            if not disable_rff:
                tff = [(a & 2) >> 1 for a in fflags]
                rnode = apply_rff_video(node, rff, tff)
                vobids = apply_rff_array(rff, vobids)
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
            absolutetime = [a * (fpsden / fpsnum) for a in range(len(rnode))]
        else:
            if rff_mode == 1:
                timecodes = [Fraction(fpsden * (a + 2), fpsnum * 2) for a in rff]
                absolutetime = absolute_time_from_timecode(timecodes)

                def apply_timecode(n, f, timecodes, absolutetime):
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

                fcc = len(rnode)
                new_fps = Fraction(fpsnum * fcc * 2, int(fcc * fpsden * asd),)

                rnode = vs.core.std.AssumeFPS(rnode, fpsnum=new_fps.numerator, fpsden=new_fps.denominator)

                timecodes = [1.0 / rnode.fps for _ in range(len(rnode))]
                absolutetime = absolute_time_from_timecode(timecodes)

        changes = []

        for a in range(1, len(vobids)):
            if vobids[a] != vobids[a - 1]:
                changes += [a]

        changes += [len(rnode) - 1]
        assert len(changes) == len(is_chapter)

        output_chapters = []
        for i in range(len(is_chapter)):
            a = is_chapter[i]

            if not a:
                continue

            b = -1

            for j in range(i + 1, len(is_chapter)):
                b = is_chapter[j]
                if b:
                    break
            output_chapters += [changes[-1] if b == -1 else changes[j - 1]]

        dvnavchapters = double_check_dvdnav(self.iso_path, title_nr)

        if dvnavchapters is not None and (rff_mode == 0 or rff_mode == 2):
            # ???????
            if fpsden == 1001:
                dvnavchapters = [a * 1.001 for a in dvnavchapters]

            adjusted = [absolutetime[i] for i in output_chapters]  # [1:len(output_chapters)-1] ]
            if len(adjusted) != len(dvnavchapters):
                print("DVDNAVCHAPTER LENGTH DO NOT MATCH OUR chapters", len(adjusted), len(dvnavchapters))
                print(adjusted)
                print(dvnavchapters)
            else:
                framelen = fpsden / fpsnum
                for i in range(len(adjusted)):
                    # tolerance no idea why so big
                    # on hard telecine ntcs it matches up almost perfectly
                    # but on ~24p pal rffd it does not lol
                    if abs(adjusted[i] - dvnavchapters[i]) > framelen * 20:
                        print("DVDNAV DONT MATCH OUR CHAPTER {} {}".format(adjusted[i], dvnavchapters[i]))
                        print(adjusted)
                        print(dvnavchapters)
                        break
        else:
            print("Skipping sanity check with dvdnav")

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
        for i, a in enumerate(targte_pgc["audio_control"]):
            if a["available"]:
                audo = target_vts["vtsi_mat"]["vts_audio_attr"][i]
                if audo["audio_format"] == 0:
                    format = "ac3"
                elif audo["audio_format"] == 4:
                    format = "lpcm"
                else:
                    format = "unk"
                format += "("
                format += audo["language"]
                format += ")"

                audios += [format]
            else:
                audios += ["none"]

        title = Title(rnode, output_chapters, self, title_nr, title_set_nr,
                      vobidcellids_to_take, absolutetime, audios, splits, patched_end_chapter)
        if splits is not None:
            title._sanitize_splits(splits)
        return title

    def _d2v_collect_all_frameflags(self, title_set_nr: int) -> Sequence[int]:
        files = self._get_title_vob_files_for_vts(title_set_nr)
        index_file = self.indexer.index(files, output_folder=self.output_folder)[0]
        index_info = self.indexer.get_info(index_file)

        lst = []
        lst2 = []
        for iframe in index_info.frame_data:
            v = (iframe.vob, iframe.cell)

            for a in iframe.frameflags:
                if a != 0xFF:
                    lst += [a]
                    lst2 += [v]

        return lst, lst2

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

    def __repr__(self) -> str:
        to_print = f"Path: {self.iso_path}\n"
        to_print += f"Mount: {self._mount_path}\n"
        to_print += f"Titles: {self.title_count}"

        return to_print.strip()

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


def get_sectors_from_vobids(target_vts: dict, vobidcellids_to_take: List[Tuple[int, int]]) -> List[int]:
    sectors = []
    for a in vobidcellids_to_take:
        for srange in get_sectorranges_for_vobcellpair(target_vts, a):
            sectors += list(range(srange[0], srange[1] + 1))
    return sectors


def get_sectorranges_for_vobcellpair(current_vts: dict, pair_id: Tuple[int, int]) -> List[Tuple[int, int]]:
    ranges = []
    for e in current_vts["vts_c_adt"]:
        if e["vob_id"] == pair_id[0] and e["cell_id"] == pair_id[1]:
            ranges += [(e["start_sector"], e["last_sector"])]
    return ranges
