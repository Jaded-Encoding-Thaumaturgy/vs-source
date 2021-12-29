import vapoursynth as vs
from typing import Callable, List, Union, Optional

from ..utils.spathlib import SPath
from .DGIndexNV import DGIndexNV


core = vs.core


class DGIndex(DGIndexNV):
    """Built-in dgindex indexer"""

    def __init__(
        self, path: Union[SPath, str] = 'dgindex',
        vps_indexer: Optional[Callable[..., vs.VideoNode]] = None, ext: str = 'dgi'
    ) -> None:
        super().__init__(path, vps_indexer or core.d2v.Source, ext)
        print(RuntimeWarning("\n\tDGIndex is bugged, it will probably not work on your system/version.\n"))

    def get_cmd(
        self, files: List[SPath], output: SPath,
        idct_algo: int = 5, field_op: int = 2, yuv_to_rgb: int = 1
    ) -> List[str]:
        return list(map(str, [
            self._check_bin_path(), "-AIF", '[' + ','.join([f'"{str(path)}"' for path in files]) + ']',
            "-IA", str(idct_algo), "-FO", field_op, "-YR", yuv_to_rgb,
            "-OM", "0", "-HIDE", "-EXIT", "-O", output
        ]))
