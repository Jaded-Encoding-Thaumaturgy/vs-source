from __future__ import annotations

import logging
import datetime
import vapoursynth as vs
from abc import abstractmethod
from fractions import Fraction
from pyparsedvd import vts_ifo
from itertools import accumulate
from typing import List, Tuple, cast, Dict

from .utils.types import Range
from .utils.spathlib import SPath
from .DVDIndexers import D2VWitch, DVDIndexer
from .dataclasses import D2VIndexFileInfo, DGIndexFileInfo, IFOFileInfo, IndexFileType

core = vs.core


class IsoFileCore:
    _subfolder = "VIDEO_TS"

    def __init__(
        self, path: SPath, indexer: DVDIndexer = D2VWitch(), safe_indices: bool = False, force_root: bool = False
    ):
        self.iso_path = SPath(path).absolute()

        if not self.iso_path.is_dir() and not self.iso_path.is_file():
            raise ValueError("IsoFile: path needs to point to a .ISO or a dir root of DVD")

        self.indexer = indexer
        self.safe_indices = safe_indices
        self.force_root = force_root

        # Here so it resets between reloads in vspreview
        self._idx_path: SPath | None = None
        self._mount_path: SPath | None = None
        self._clip: vs.VideoNode | None = None
        self._vob_files: List[SPath] | None = None
        self._index_info: Dict[int, IndexFileType] = {}
        self._ifo_info: Dict[int, IFOFileInfo] = {}
        # Split Clips, Split Chapters, Joined Clip, Joined Chapters
        self._processed_titles: Tuple[List[vs.VideoNode], List[List[int]], vs.VideoNode, List[int]] | None = None

    def get_idx_info(self, index: int = 0) -> D2VIndexFileInfo | DGIndexFileInfo:
        if index not in self._index_info:
            self._index_info[index] = self.indexer.get_info(self.idx_path, index)

        return self._index_info[index]

    def get_ifo_info(self, mount_path: SPath) -> IFOFileInfo:
        ifo_hash = hash(mount_path)

        if ifo_hash in self._ifo_info:
            return self._ifo_info[ifo_hash]

        ifo_files = [f for f in sorted(mount_path.glob('*.[iI][fF][oO]')) if f.stem != 'VIDEO_TS']

        program_chains = []

        m_ifos = len(ifo_files) > 1

        for ifo_file in ifo_files:
            with open(ifo_file, 'rb') as file:
                curr_pgci = vts_ifo.load_vts_pgci(file)
                program_chains += curr_pgci.program_chains[int(m_ifos):]

        split_chapters: List[List[int]] = []

        fps = Fraction(30000, 1001)

        for prog in program_chains:
            dvd_fps_s = [pb_time.fps for pb_time in prog.playback_times]
            if all(dvd_fps_s[0] == dvd_fps for dvd_fps in dvd_fps_s):
                fps = vts_ifo.FRAMERATE[dvd_fps_s[0]]
            else:
                raise ValueError('IsoFile: No VFR allowed!')

            raw_fps = 30 if fps.numerator == 30000 else 25

            split_chapters.append([0] + [
                pb_time.frames + (pb_time.hours * 3600 + pb_time.minutes * 60 + pb_time.seconds) * raw_fps
                for pb_time in prog.playback_times
            ])

        chapters = [list(accumulate(chapter_frames)) for chapter_frames in split_chapters]

        self._ifo_info[ifo_hash] = IFOFileInfo(chapters, fps, m_ifos)

        return self._ifo_info[ifo_hash]

    def get_title(
        self, clip_index: int | None = None, chapters: Range | List[Range] | None = None
    ) -> vs.VideoNode | List[vs.VideoNode]:
        clip, ranges = self.split_titles[clip_index] if clip_index is not None else self.joined_titles
        rlength = len(ranges)

        start: int | None
        end: int | None

        if isinstance(chapters, int):
            start, end = ranges[0], ranges[-1]

            if chapters == rlength - 1:
                start = ranges[-2]
            elif chapters == 0:
                end = ranges[1]
            elif chapters < 0:
                start = ranges[rlength - 1 + chapters]
                end = ranges[rlength + chapters]
            else:
                start = ranges[chapters]
                end = ranges[chapters + 1]

            return clip[start:end]
        elif isinstance(chapters, tuple):
            start, end = chapters

            if start is None:
                start = 0
            elif start < 0:
                start = rlength - 1 + start

            if end is None:
                end = rlength - 1
            elif end < 0:
                end = rlength - 1 + end
            else:
                end += 1

            return clip[ranges[start]:ranges[end]]
        elif isinstance(chapters, list):
            return [cast(vs.VideoNode, self.get_title(clip_index, rchap)) for rchap in chapters]

        return clip

    def __repr__(self) -> str:
        to_print = f"""{self.__class__.__name__[1:]}:
    Iso path: "{self.iso_path}"
    Mount path: {self.mount_path}
    IFO Info:
        Titles:
        """

        ifo_info = self.get_ifo_info(self.mount_path)

        for i, title in enumerate(ifo_info.chapters):
            timestrings = list(map(lambda x: str(datetime.timedelta(seconds=round(x * ifo_info.fps / 1000))), title))

            tmp = ''
            timestring = ''

            for j, v in enumerate(timestrings, 1):
                tmp += f'{j-1}->{v}, '

                if j % 7 == 0:
                    timestring += f"\n{' ' * 20}{tmp}"
                    tmp = ''

            timestring += tmp

            to_print += f"""    Title: {i}
                Chapters: {timestring}
        """

        to_print += f"""FPS: {ifo_info.fps}
        Multiple IFOs: {ifo_info.is_multiple_IFOs}
        """

        return to_print.strip()

    def _mount_folder_path(self) -> SPath:
        if self.force_root:
            return self.iso_path

        if self.iso_path.name.upper() == self._subfolder:
            self.iso_path = self.iso_path.parent

        return self.iso_path / self._subfolder

    def _split_chapters_clips(
        self, split_chapters: List[List[int]], dvd_menu_length: int
    ) -> Tuple[List[List[int]], List[vs.VideoNode]]:
        durations = list(accumulate([0] + [frame[-1] for frame in split_chapters]))

        # Remove splash screen and DVD Menu
        clip = self.clip[dvd_menu_length:]

        # Trim per title
        clips = [clip[s:e] for s, e in zip(durations[:-1], durations[1:])]

        if dvd_menu_length:
            clips.append(self.clip[:dvd_menu_length])
            split_chapters.append([0, dvd_menu_length])

        return split_chapters, clips

    def _join_chapters_clips(
        self, split_chapters: List[List[int]], split_clips: List[vs.VideoNode]
    ) -> Tuple[List[int], vs.VideoNode]:
        joined_chapters = split_chapters[0]
        joined_clip = split_clips[0]

        if len(split_chapters) > 1:
            for rrange in split_chapters[1:]:
                joined_chapters += [
                    r + joined_chapters[-1] for r in rrange if r != 0
                ]

        if len(split_clips) > 1:
            for cclip in split_clips[1:]:
                joined_clip += cclip

        return joined_chapters, joined_clip

    def _process_titles(self) -> Tuple[List[vs.VideoNode], List[List[int]], vs.VideoNode, List[int]]:
        ifo_info = self.get_ifo_info(self.mount_path)

        idx_info = self.get_idx_info(0)

        if isinstance(self.indexer, D2VWitch):
            dvd_menu_length = idx_info.videos[0].num_frames
        else:
            dvd_menu_length = len(idx_info.frame_data) if idx_info.videos[0].size > 2 << 12 else 0

        _split_chapters, _split_clips = self._split_chapters_clips(ifo_info.chapters, dvd_menu_length)
        _joined_chapters, _joined_clip = self._join_chapters_clips(_split_chapters, _split_clips)

        tot_frames = self.clip.num_frames

        if _joined_chapters[-1] <= tot_frames:
            return _split_clips, _split_chapters, _joined_clip, _joined_chapters

        if not self.safe_indices:
            logging.warning(
                Warning(
                    "\nIsoFile:"
                    "\n\tThe chapters are broken, last few chapters"
                    "\n\tand negative indices will probably give out an error."
                    "\n\tYou can set safe_indices = True and trim down the chapters."
                    "\n"
                )
            )

            return _split_clips, _split_chapters, _joined_clip, _joined_chapters

        offset = 0
        split_chapters: List[List[int]] = [[] for _ in range(len(_split_chapters))]

        for i in range(len(_split_chapters)):
            for j in range(len(_split_chapters[i])):
                if _split_chapters[i][j] + offset < tot_frames:
                    split_chapters[i].append(_split_chapters[i][j])
                else:
                    split_chapters[i].append(
                        tot_frames - dvd_menu_length - len(_split_chapters) + i + 2
                    )

                    for k in range(i + 1, len(_split_chapters) - (int(dvd_menu_length > 0))):
                        split_chapters[k] = [0, 1]

                    if dvd_menu_length:
                        split_chapters[-1] = _split_chapters[-1]

                    break
            else:
                offset += _split_chapters[i][-1]
                continue
            break

        _split_chapters, _split_clips = self._split_chapters_clips(
            split_chapters if dvd_menu_length == 0 else split_chapters[:-1], dvd_menu_length
        )

        _joined_chapters, _joined_clip = self._join_chapters_clips(_split_chapters, _split_clips)

        return _split_clips, _split_chapters, _joined_clip, _joined_chapters

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
    def vob_files(self) -> List[SPath]:
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
    def idx_path(self) -> SPath:
        if self._idx_path is not None:
            return self._idx_path

        self._idx_path = self.indexer.index(self.vob_files, False, False, self.iso_path.get_folder())[0]

        if self._idx_path is None or not self._idx_path.is_file():
            raise FileExistsError("Error while creating the index file!")

        return self._idx_path

    @property
    def clip(self) -> vs.VideoNode:
        if self._clip is not None:
            return self._clip

        idx_info = self.get_idx_info(0)

        indexer_kwargs = {**self.indexer.indexer_kwargs}

        if isinstance(idx_info, DGIndexFileInfo) and idx_info.footer.film == 100:
            indexer_kwargs |= dict(fieldop=2)

        ifo_info = self.get_ifo_info(self.mount_path)

        self._clip = self.indexer.vps_indexer(
            self.idx_path, **indexer_kwargs
        ).std.AssumeFPS(
            None, ifo_info.fps.numerator, ifo_info.fps.denominator
        )

        return self._clip

    @property
    def split_titles(self) -> List[Tuple[vs.VideoNode, List[int]]]:
        if self._processed_titles is None:
            self._processed_titles = self._process_titles()

        return list(zip(*self._processed_titles[:2]))

    @property
    def joined_titles(self) -> Tuple[vs.VideoNode, List[int]]:
        if self._processed_titles is None:
            self._processed_titles = self._process_titles()

        return self._processed_titles[2:]

    @abstractmethod
    def _get_mounted_disc(self) -> SPath | None:
        raise NotImplementedError()

    @abstractmethod
    def _mount(self) -> SPath | None:
        raise NotImplementedError()
