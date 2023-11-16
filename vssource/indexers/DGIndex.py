from __future__ import annotations

import os
import subprocess

from vstools import SPath, core

from .D2VWitch import D2VWitch

__all__ = [
    'DGIndex'
]


class DGIndex(D2VWitch):
    _bin_path = 'dgindex'
    _ext = 'd2v'
    _source_func = core.lazy.d2v.Source

    def get_cmd(
        self, files: list[SPath], output: SPath,
        idct_algo: int = 5, field_op: int = 2, yuv_to_rgb: int = 1
    ) -> list[str]:
        is_linux = os.name != 'nt'

        if is_linux:
            output = SPath(f'Z:\\{str(output)[1:]}')
            paths = list(subprocess.check_output(['winepath', '-w', f]).decode('utf-8').strip() for f in files)
        else:
            paths = list(map(str, files))

        for f in paths:
            assert ' ' not in f, 'DGIndex only supports paths without spaces in them!'

        return list(map(str, [
            self._get_bin_path(),
            '-IF=[' + ','.join([f'{p}' for p in paths]) + ']',
            f'-IA={idct_algo}', f'-FO={field_op}', f'-YR={yuv_to_rgb}',
            '-OM=0', f'-OF=[{output.with_suffix("")}]', '-HIDE', '-EXIT'
        ]))
