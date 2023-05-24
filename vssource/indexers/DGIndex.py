from __future__ import annotations

from vstools import SPath, core

from .DGIndexNV import DGIndexNV

__all__ = [
    'DGIndex'
]


class DGIndex(DGIndexNV):
    _bin_path = 'dgindex'
    _ext = 'dgi'
    _source_func = core.lazy.d2v.Source  # type: ignore

    def get_cmd(
        self, files: list[SPath], output: SPath,
        idct_algo: int = 5, field_op: int = 2, yuv_to_rgb: int = 1
    ) -> list[str]:
        return list(map(str, [
            self._get_bin_path(), "-AIF", '[' + ','.join([f'"{str(path)}"' for path in files]) + ']',
            "-IA", str(idct_algo), "-FO", field_op, "-YR", yuv_to_rgb,
            "-OM", "0", "-HIDE", "-EXIT", "-O", output
        ]))
