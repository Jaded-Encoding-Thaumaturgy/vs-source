from __future__ import annotations

import re
from typing import Sequence

from vstools import SPath

__all__ = [
    'opt_int', 'opt_ints',

    'get_all_vobs'
]


def opt_int(val: str | int | None) -> int | None:
    return int(val) if val is not None else None


def opt_ints(vals: Sequence[str | int | None]) -> Sequence[int | None]:
    return [opt_int(x) for x in vals]


def get_all_vobs(*files: SPath) -> list[SPath]:
    found_files = list[SPath]()

    for file in list(files):
        if matches := re.search(r"VTS_([0-9]{2})_([0-9])\.VOB", file.name, re.IGNORECASE):
            found_files += file.get_folder().glob(
                f'[vV][tT][sS]_[{matches[1][0]}-9][{matches[1][1]}-9]_[{matches[2]}-9].[vV][oO][bB]'
            )

    return found_files
