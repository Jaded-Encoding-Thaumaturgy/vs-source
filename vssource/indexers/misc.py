from __future__ import annotations

from vstools import core

from .base import Indexer

__all__ = [
    'BestSource',

    'IMWRI',

    'LSMAS',

    'CarefulSource',

    'FFMS2'
]


class BestSource(Indexer):
    _source_func = core.lazy.bs.VideoSource  # type: ignore


class IMWRI(Indexer):
    _source_func = core.lazy.imwri.Read


class LSMAS(Indexer):
    _source_func = core.lazy.lsmas.LWLibavSource


class CarefulSource(Indexer):
    _source_func = core.lazy.cs.ImageSource


class FFMS2(Indexer):
    _source_func = core.lazy.ffms2.Source
