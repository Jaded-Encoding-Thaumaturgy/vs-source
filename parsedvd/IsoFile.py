import json
import atexit
import subprocess
import vapoursynth as vs
from os import name as os_name
from typing import List, Union, Optional, Tuple, Any

from .utils.spathlib import SPath

from .IsoFileCore import IsoFileCore

__all__ = ['IsoFile']

Range = Union[Optional[int], Tuple[Optional[int], Optional[int]]]

core = vs.core


class __WinIsoFile(IsoFileCore):
    def _get_mount_path(self) -> SPath:
        if self.iso_path.is_dir():
            return self._mount_folder_path()

        disc = self.__get_mounted_disc() or self.__mount()

        if not disc:
            raise RuntimeError("IsoFile: Couldn't mount ISO file!")

        return disc / self._subfolder

    def __run_disc_util(self, iso_path: SPath, util: str) -> Optional[SPath]:
        process = subprocess.Popen([
            "PowerShell", fr'{util}-DiskImage -ImagePath "{str(iso_path)}" | Get-Volume | ConvertTo-Json'],
            stdout=subprocess.PIPE
        )

        pbjson, err = process.communicate()

        if err or str(pbjson[:len(util)], 'utf8') == util or pbjson == b'':
            return None
        elif util.lower() == "dismount":
            return SPath('')

        bjson: dict[str, str] = json.loads(str(pbjson, 'utf-8'))

        return SPath(f"{bjson['DriveLetter']}:\\")

    def __get_mounted_disc(self) -> Optional[SPath]:
        return self.__run_disc_util(self.iso_path, 'Get')

    def __mount(self) -> Optional[SPath]:
        if (mount := self.__run_disc_util(self.iso_path, 'Mount')):
            atexit.register(self.__unmount)
        return mount

    def __unmount(self) -> Optional[SPath]:
        return self.__run_disc_util(self.iso_path, 'Dismount')


class __LinuxIsoFile(IsoFileCore):
    loop_path: Optional[SPath] = None
    cur_mount: Optional[SPath] = None

    def _get_mount_path(self) -> SPath:
        if self.iso_path.is_dir():
            return self._mount_folder_path()

        disc = self.__get_mounted_disc() or self.__mount()

        if not disc:
            raise RuntimeError("IsoFile: Couldn't mount ISO file!")

        return disc / self._subfolder

    def __subprun(self, *args: Any) -> str:
        return subprocess.run(list(map(str, args)), capture_output=True, universal_newlines=True).stdout

    def __get_mounted_disc(self) -> Optional[SPath]:
        if not (loop_path := self.__subprun("losetup", "-j", self.iso_path).strip().split(":")[0]):
            return self.cur_mount

        self.loop_path = SPath(loop_path)

        if "MountPoints:" in (device_info := self.__run_disc_util(self.loop_path, ["info", "-b"], True)):
            if cur_mount := device_info.split("MountPoints:")[1].split("\n")[0].strip():
                self.cur_mount = SPath(cur_mount)

        return self.cur_mount

    def __run_disc_util(self, path: SPath, params: List[str], strip: bool = False) -> str:
        output = self.__subprun("udisksctl", *params, str(path))

        return output.strip() if strip else output

    def __mount(self) -> Optional[SPath]:
        if not self.loop_path:
            loop_path = self.__run_disc_util(self.iso_path, ["loop-setup", "-f"], True)

            if "mapped file" not in loop_path.lower():
                raise RuntimeError("IsoFile: Couldn't map the ISO file!")

            loop_splits = loop_path.split(" as ")

            self.loop_path = SPath(loop_splits[-1][:-1])

        if "mounted" not in (cur_mount := self.__run_disc_util(self.loop_path, ["mount", "-b"], True)).lower():
            return None

        mount_splits = cur_mount.split(" at ")

        self.cur_mount = SPath(mount_splits[-1])

        atexit.register(self.__unmount)

        return self.cur_mount

    def __unmount(self) -> bool:
        if not self.loop_path:
            return True
        self.__run_disc_util(self.loop_path, ["unmount", "-b", ])
        return bool(self.__run_disc_util(self.loop_path, ["loop-delete", "-b", ]))


IsoFile = __WinIsoFile if os_name == 'nt' else __LinuxIsoFile
