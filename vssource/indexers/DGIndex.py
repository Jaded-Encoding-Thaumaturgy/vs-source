from __future__ import annotations

from vstools import SPath, core

from .D2VWitch import D2VWitch

__all__ = [
    'DGIndex'
]


class DGIndex(D2VWitch):
    _bin_path = 'dgindex'
    _ext = 'd2v'
    _source_func = core.lazy.d2v.Source  # type: ignore

    def get_cmd(
        self, files: list[SPath], output: SPath,
        idct_algo: int = 5, field_op: int = 2, yuv_to_rgb: int = 1
    ) -> list[str]:
        return list(map(str, [
            self._get_bin_path(), "-IF=[" + ','.join([f'{str(path)}' for path in files]) + ']',
            "-IA=" + str(idct_algo), "-FO=" + str(field_op), "-YR=" + str(yuv_to_rgb),
            "-OM=0", "-OF=[" + str(output).replace(".d2v", "") + "]", "-HIDE", "-EXIT"
        ]))
