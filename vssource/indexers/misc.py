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
    """BestSource indexer"""

    _source_func = core.lazy.bs.VideoSource  # type: ignore


class IMWRI(Indexer):
    """ImageMagick Writer-Reader indexer"""

    _source_func = core.lazy.imwri.Read


class LSMAS(Indexer):
    """L-SMASH-Works indexer"""

    _source_func = core.lazy.lsmas.LWLibavSource


class CarefulSource(Indexer):
    """CarefulSource indexer"""

    _source_func = core.lazy.cs.ImageSource


class FFMS2(Indexer):
    """FFmpegSource2 indexer"""

    _source_func = core.lazy.ffms2.Source
