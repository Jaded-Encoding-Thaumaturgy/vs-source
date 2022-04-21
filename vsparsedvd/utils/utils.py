from __future__ import annotations

from typing import Sequence


def opt_int(val: str | int | None) -> int | None:
    return int(val) if val is not None else None


def opt_ints(vals: Sequence[str | int | None]) -> Sequence[int | None]:
    return [opt_int(x) for x in vals]
