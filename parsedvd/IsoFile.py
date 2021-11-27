import json
import atexit
import subprocess
import vapoursynth as vs
from pathlib import Path
from os import name as os_name
from typing import List, Union, Optional, Tuple, Any


from .IsoFileCore import IsoFileCore

Range = Union[Optional[int], Tuple[Optional[int], Optional[int]]]

core = vs.core


class __WinIsoFile(IsoFileCore):
    def _get_mount_path(self) -> Path:
        if self.iso_path.is_dir():
            return self._mount_folder_path()

        disc = self.__get_mounted_disc() or self.__mount()

        if not disc:
            raise RuntimeError("IsoFile: Couldn't mount ISO file!")

        return disc / self._subfolder

    def __run_disc_util(self, iso_path: Path, util: str) -> Optional[Path]:
        process = subprocess.Popen([
            "PowerShell", fr'{util}-DiskImage -ImagePath "{str(iso_path)}" | Get-Volume | ConvertTo-Json'],
            stdout=subprocess.PIPE
        )

        pbjson, err = process.communicate()

        if err or str(pbjson[:len(util)], 'utf8') == util:
            raise RuntimeError("IsoFile: Couldn't mount ISO file!")
        elif pbjson == b'':
            return None
        elif util.lower() == "dismount":
            return Path("")

        bjson: dict[str, str] = json.loads(str(pbjson, 'utf-8'))

        return Path(f"{bjson['DriveLetter']}:\\")

    def __get_mounted_disc(self) -> Optional[Path]:
        return self.__run_disc_util(self.iso_path, 'Get')

    def __mount(self) -> Optional[Path]:
        mount = self.__run_disc_util(self.iso_path, 'Mount')
        if mount:
            atexit.register(self.__unmount)
        return mount

    def __unmount(self) -> Optional[Path]:
        return self.__run_disc_util(self.iso_path, 'Dismount')


class __LinuxIsoFile(IsoFileCore):
    loop_path: Path = Path("")
    cur_mount: Path = Path("")

    def _get_mount_path(self) -> Path:
        if self.iso_path.is_dir():
            return self._mount_folder_path()

        disc = self.__get_mounted_disc()
        if disc == Path(""):
            disc = self.__mount()

        return disc / self._subfolder

    def __subprun(self, *args: Any) -> str:
        return subprocess.run(list(map(str, args)), capture_output=True, universal_newlines=True).stdout

    def __get_mounted_disc(self) -> Path:
        loop_path = self.__subprun("losetup", "-j", self.iso_path).strip().split(":")[0]

        if not loop_path:
            return self.cur_mount

        self.loop_path = Path(loop_path)

        device_info = self.__run_disc_util(self.loop_path, ["info", "-b"], True)

        if "MountPoints:" in device_info:
            cur_mount = device_info.split("MountPoints: ")[1].split("\n")[0].strip()

            if cur_mount:
                self.cur_mount = Path(cur_mount)

        return self.cur_mount

    def __run_disc_util(self, path: Path, params: List[str], strip: bool = False) -> str:
        output = self.__subprun("udisksctl", *params, str(path))

        return output.strip() if strip else output

    def __mount(self) -> Path:
        if self.loop_path == Path(""):
            loop_path = self.__run_disc_util(self.iso_path, ["loop-setup", "-f"], True)

            if "mapped file" not in loop_path.lower():
                raise RuntimeError("IsoFile: Couldn't map the ISO file!")

            self.loop_path = Path(loop_path.split(" as ")[-1][:-1])

        cur_mount = self.__run_disc_util(self.loop_path, ["mount", "-b"], True)

        if "mounted" not in cur_mount.lower():
            raise RuntimeError("IsoFile: Couldn't mount ISO file!")

        self.cur_mount = Path(cur_mount.split(" at ")[-1])

        atexit.register(self.__unmount)

        return self.cur_mount

    def __unmount(self) -> bool:
        self.__run_disc_util(self.loop_path, ["unmount", "-b", ])
        return bool(self.__run_disc_util(self.loop_path, ["loop-delete", "-b", ]))


IsoFile = __WinIsoFile if os_name == 'nt' else __LinuxIsoFile
