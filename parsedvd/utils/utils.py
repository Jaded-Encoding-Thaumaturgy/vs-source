from typing import Union, Optional, List, Sequence


def opt_int(val: Optional[Union[str, int]]) -> Optional[int]:
    return int(val) if val is not None else None


def opt_ints(vals: Sequence[Optional[Union[str, int]]]) -> List[Optional[int]]:
    return [opt_int(x) for x in vals]
