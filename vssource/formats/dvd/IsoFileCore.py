from __future__ import annotations
from dataclasses import dataclass
import os
import datetime
from abc import abstractmethod
from fractions import Fraction
from typing import List, Sequence, Tuple

from vstools import CustomValueError, SPath, vs

from ...indexers import D2VWitch, DGIndex, ExternalIndexer
import json
from functools import partial


from .parsedvd.ifo import IFO0, IFOX
import jsondiff

# d2vwitch needs this patch applied
# https://gist.github.com/jsaowji/ead18b4f1b90381d558eddaf0336164b


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

# TODO: from pydvdsrc
def _get_sectors_from_vobids(target_vts: dict, vobidcellids_to_take: List[Tuple[int, int]]):
    sectors = []
    for a in vobidcellids_to_take:
        for srange in _get_sectorranges_for_vobcellpair(target_vts, a):
            sectors += list(range(srange[0], srange[1] + 1))
    return sectors

# TODO: from pydvdsrc
def _get_sectorranges_for_vobcellpair(current_vts: dict, pair_id: Tuple[int, int]):
    ranges = []
    for e in current_vts["vts_c_adt"]:
        if e["cell_id"] == pair_id[1] and e["vob_id"] == pair_id[0]:
            ranges += [(e["start_sector"], e["last_sector"])]
    return ranges


@dataclass
class Title:
    node: vs.VideoNode
    chapters: List[int]
    _core: IsoFileCore
    _vts: int
    _vobidcellids_to_take: List[Tuple[int, int]]
    _absolute_time: List[float]
    _audios: List[str]

    def __repr__(self) -> str:
        chapters = self.chapters + [len(self.node) - 1]
        chapter_legnths = [self._absolute_time[chapters[i + 1]] - self._absolute_time[chapters[i]]
                           for i in range(len(self.chapters))]

        chapter_legnths = [str(datetime.timedelta(seconds=x)) for x in chapter_legnths]
        timestrings = [str(datetime.timedelta(seconds=self._absolute_time[x])) for x in self.chapters]

        to_print = "Chapters: (fz)\n"
        for i in range(len(timestrings)):
            to_print += "{:02} {:015} {:015} {}\n".format(i, timestrings[i], chapter_legnths[i], self.chapters[i])
        to_print += "Audios: (fz)\n"
        for i, a in enumerate(self._audios):
            to_print += "{} {}\n".format(i, a)

        return to_print.strip()

    def audio_fz(self, i: int = 0) -> vs.AudioNode:
        asd = self._audios[i]

        target_vts = self._core.json["ifos"][self._vts]
        sectors = _get_sectors_from_vobids(target_vts, self._vobidcellids_to_take)

        if asd.startswith("ac3"):
            return vs.core.dvdsrc.FullAC3(self._core.iso_path, self._vts, 1, sectors, i)
        elif asd.startswith("lpcm"):
            return vs.core.dvdsrc.FullLPCM(self._core.iso_path, self._vts, 1, sectors, i)
        else:
            raise CustomValueError('invalid audio at index', self.__class__)

    def dump_ac3(self, a: str, i: int = 0):
        wrt = open(a, "wb")

        target_vts = self._core.json["ifos"][self._vts]
        sectors = _get_sectors_from_vobids(target_vts, self._vobidcellids_to_take)

        nd = vs.core.dvdsrc.RawAc3(self._core.iso_path, self._vts, 1, sectors, i)
        for f in nd.frames():
            wrt.write(bytes(f[0]))
        wrt.close()

    def set_output(self, n: int):
        self.node.set_output(n)


class IsoFileCore:
    _subfolder = "VIDEO_TS"

    def __init__(
        self, path: SPath | str,
        use_dvdsrc = True,
        indexer: ExternalIndexer | type[ExternalIndexer] = None,
    ):
        '''
        Only external indexer supported D2VWitch and DGIndex

        indexer only uesd if use_dvdsrc == False
        '''
        if indexer is None:
            indexer = DGIndex() if os.name == "nt" else D2VWitch()
        force_root: bool = False

        self._mount_path: SPath | None = None
        self._vob_files: list[SPath] | None = None
        self._ifo_files: list[SPath] | None = None



        self.has_dvdsrc = hasattr(vs.core,"dvdsrc")
        
        if not self.has_dvdsrc and use_dvdsrc:
            use_dvdsrc = False
            print("Requested dvdsrc but not installed")

        self.use_dvdsrc = use_dvdsrc
        self.iso_path = SPath(path).absolute()

        if self.use_dvdsrc:
            self.json = json.loads(vs.core.dvdsrc.Json(self.iso_path))
        else:
            self.json = { "ifos": []}
            for i,a in enumerate(self.ifo_files):
                if i == 0:
                    self.json["ifos"] += [ IFO0(a).crnt ]
                else:
                    self.json["ifos"] += [ IFOX(a).crnt ]
            if not self.has_dvdsrc:
                print("Does not have dvdsrc cant double check json with libdvdread")
            else:
                dvdsrc_json = json.loads(vs.core.dvdsrc.Json(self.iso_path))
                del dvdsrc_json["dvdpath"]
                del dvdsrc_json["current_vts"]
                
                del dvdsrc_json["current_domain"]
                if json.dumps(dvdsrc_json,sort_keys=True) != json.dumps(self.json,sort_keys=True):
                    print("libdvdread json does not match our json")
                    #dff = jsondiff.diff(dvdsrc_json, self.json)
                    #print(dff)

                

        
        #json.dump(self.json, open("/tmp/asd.json", "wt"))

        if not self.iso_path.is_dir() and not self.iso_path.is_file():
            raise CustomValueError('"path" needs to point to a .ISO or a dir root of DVD', self.__class__)

        self.indexer = indexer if isinstance(indexer, ExternalIndexer) else indexer()
        self.force_root = force_root

        self.title_count = len(self.json["ifos"][0]["tt_srpt"])

        self.output_folder = "/tmp" if os.name != "nt" else "C:/tmp" 

    def get_title(
        self,
        title_nr: int | None = 1,
        angle_nr: int | None = None,
        rff_mode: int = 0,
    ) -> Title:
        """
    Gets a title.

    :param title_nr:    title nr starting from 1
    :param angle_nr:    starting from 1
    :param rff_mode:    0 apply rff soft telecine (default)
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
                raise CustomValueError('title is not composed of one program chain (unsupported)', self.__class__)

        pgc_i = target_title[0]["pgcn"] - 1
        title_programs = [a["pgn"] for a in target_title]
        targte_pgc = target_vts["vts_pgcit"][pgc_i]
        pgc_programs = targte_pgc["program_map"]

        if title_programs[0] != 1 or pgc_programs[0] != 1:
            print("TITLE DOES NOT START AT FIRST CELL OR FIRST CEL NOT PROGRAM WHY? GIVE SAMPLE")

        target_programs = [a[1] for a in list(filter(lambda x: (x[0] + 1) in title_programs, enumerate(pgc_programs)))]

        if target_programs != pgc_programs:
            print("PLEASE GIVE SAMPLE PGC PROGRAMS DOES NOT MATCH TITLE I HAVE NO IDEA WHY ONE SHOULD DO THAT")

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


        if self.use_dvdsrc:
            import pydvdsrc

            sectors = _get_sectors_from_vobids(target_vts, vobidcellids_to_take)
            rawnode = vs.core.dvdsrc.FullM2V(self.iso_path, vts=title_set_nr, domain=1, sectors=sectors)
            exa = pydvdsrc.DVDSRCM2vInfoExtracter(rawnode)
            rff = exa.rff

            if not disable_rff:
                rnode = pydvdsrc.apply_rff_video(rawnode, exa.rff, exa.tff)
                vobids = pydvdsrc.apply_rff_array(exa.rff, exa.vobid)
            else:
                rnode = rawnode
                vobids = exa.vobid
        else:
            import pydvdsrc

            vob_input_files = self._get_title_vob_files_for_vts(title_set_nr)
            dvddd = self._d2v_vobid_frameset(title_set_nr)

            frameranges = []
            for a in vobidcellids_to_take:
                frameranges += dvddd[a]

            fflags, vobids = self._d2v_collect_all_frameflags(title_set_nr)

            index_file = self.indexer.index(vob_input_files, output_folder=self.output_folder)[0]
            node = self.indexer._source_func(index_file, rff=False)
            # node = self.indexer.source(vob_input_files, output_folder=self.output_folder, rff=False)
            assert len(node) == len(fflags)

            fflags = pydvdsrc.cut_array_on_ranges(fflags, frameranges)
            vobids = pydvdsrc.cut_array_on_ranges(vobids, frameranges)
            node = pydvdsrc.cut_node_on_ranges(node, frameranges)

            rff = [(a & 1) for a in fflags]

            if not disable_rff:
                tff = [(a & 2) >> 1 for a in fflags]
                rnode = pydvdsrc.apply_rff_video(node, rff, tff)
                vobids = pydvdsrc.apply_rff_array(rff, vobids)
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

                timecodes = [1.0 / rnode.fps for a in range(len(rnode))]
                absolutetime = absolute_time_from_timecode(timecodes)

        # I have no idea what logic thise code acually follows just tweaked it till it matched dvdnav
        # only touch on mismatch with dvdnav
        changes = []

        # changes += [0]

        for a in range(1, len(vobids)):
            if vobids[a] != vobids[a - 1]:
                changes += [a]

        changes += [len(rnode) - 1]
        # assert len(changes) == len(is_chapter)+1

        output_chapters = []
        cntt = 0
        for i, a in enumerate(is_chapter):
            if i == 0:
                continue
            if a:
                cntt += 1
                output_chapters += [changes[i - 1]]

        if is_chapter[-1]:
            output_chapters += [changes[-1]]
        else:  # end gets added anyway ????
            output_chapters += [changes[-2]]
        # Changes allowed again from here one

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

        return Title(rnode, output_chapters, self, title_set_nr, vobidcellids_to_take, absolutetime, audios)

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
