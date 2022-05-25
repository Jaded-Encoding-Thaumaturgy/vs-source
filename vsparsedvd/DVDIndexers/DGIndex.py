from __future__ import annotations

import logging
import vapoursynth as vs
from typing import Any, List

from .DGIndexNV import DGIndexNV
from ..utils.spathlib import SPath


core = vs.core


class DGIndex(DGIndexNV):
    """Built-in dgindex indexer"""

    def __init__(self, **kwargs: Any) -> None:
        if 'bin_path' not in kwargs:
            kwargs['bin_path'] = 'dgindex'
        if 'vps_indexer' not in kwargs:
            kwargs['vps_indexer'] = core.d2v.Source
        if 'ext' not in kwargs:
            kwargs['ext'] = 'dgi'
        super().__init__(**kwargs)
        logging.warning(RuntimeWarning("\n\tDGIndex is bugged, it will probably not work on your system/version.\n"))

    def get_cmd(
        self, files: List[SPath], output: SPath,
        idct_algo: int = 5, field_op: int = 2, yuv_to_rgb: int = 1
    ) -> List[str]:
        return list(map(str, [
            self._get_bin_path(), "-AIF", '[' + ','.join([f'"{str(path)}"' for path in files]) + ']',
            "-IA", str(idct_algo), "-FO", field_op, "-YR", yuv_to_rgb,
            "-OM", "0", "-HIDE", "-EXIT", "-O", output
        ]))
