from __future__ import annotations

from vstools import core

from .base import Indexer

__all__ = [
    'BestSource',

    'IMWRI',

    'LSMAS'
]


class BestSource(Indexer):
    _source_func = core.lazy.bs.VideoSource  # type: ignore


class IMWRI(Indexer):
    _source_func = core.lazy.imwri.Read


class LSMAS(Indexer):
    _source_func = core.lazy.lsmas.LWLibavSource
