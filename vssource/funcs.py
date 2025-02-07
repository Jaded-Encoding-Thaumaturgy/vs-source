from __future__ import annotations

from functools import partial
from typing import Any, Literal, Generator, Protocol, Sequence, overload

from vstools import (
    ChromaLocationT, ColorRangeT, CustomRuntimeError, FieldBasedT, FileType, FileTypeMismatchError, IndexingType,
    MatrixT, ParsedFile, PrimariesT, SPath, SPathLike, TransferT, check_perms, copy_signature, initialize_clip,
    match_clip, to_arr, vs
)

from .indexers import FFMS2, IMWRI, LSMAS, BestSource, D2VWitch, DGIndex, DGIndexNV, Indexer

__all__ = [
    'parse_video_filepath',
    'source'
]


def parse_video_filepath(
    filepath: SPathLike | Sequence[SPathLike] | Generator[SPath, None, None]
) -> tuple[SPath, ParsedFile]:
    if isinstance(filepath, Generator):
        filepath = list(filepath)

    try:
        filepath = next(iter(Indexer.normalize_filenames(filepath)))
    except StopIteration:
        raise CustomRuntimeError('No files provided!', source, filepath)

    check_perms(filepath, 'r', strict=True, func=source)

    file = FileType.parse(filepath) if filepath.exists() else None

    def _check_file_type(file_type: FileType) -> bool:
        return file_type in (FileType.VIDEO, FileType.IMAGE) or file_type.is_index()

    if not file or not _check_file_type(FileType(file.file_type)):
        for itype in IndexingType:
            if (newpath := filepath.with_suffix(f'{filepath.suffix}{itype.value}')).exists():
                file = FileType.parse(newpath)

    if not file or not _check_file_type(FileType(file.file_type)):
        raise FileTypeMismatchError('The file "{file}" isn\'t a video or image file!', source, file=filepath)

    return filepath, file


class source_func(Protocol):
    @overload
    def __call__(
        self,
        filepath: SPathLike | Sequence[SPathLike],
        bits: int | None = None, *,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        chroma_location: ChromaLocationT | None = None,
        color_range: ColorRangeT | None = None,
        field_based: FieldBasedT | None = None,
        ref: vs.VideoNode | None = None,
        film_thr: float = 99.0,
        name: str | Literal[False] = False,
        **kwargs: Any
    ) -> vs.VideoNode:
        ...

    @overload
    def __call__(
        self,
        bits: int | None = None, *,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        chroma_location: ChromaLocationT | None = None,
        color_range: ColorRangeT | None = None,
        field_based: FieldBasedT | None = None,
        ref: vs.VideoNode | None = None,
        film_thr: float = 99.0,
        name: str | Literal[False] = False,
        **kwargs: Any
    ) -> source_func:
        ...

    @overload
    def __call__(
        self,
        filepath: None,
        bits: int | None = None, *,
        matrix: MatrixT | None = None,
        transfer: TransferT | None = None,
        primaries: PrimariesT | None = None,
        chroma_location: ChromaLocationT | None = None,
        color_range: ColorRangeT | None = None,
        field_based: FieldBasedT | None = None,
        ref: vs.VideoNode | None = None,
        film_thr: float = 99.0,
        name: str | Literal[False] = False,
        **kwargs: Any
    ) -> source_func:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


_source_func: source_func = ...  # type: ignore


@copy_signature(_source_func)
def source(
    filepath: SPathLike | Sequence[SPathLike] | Generator[SPathLike, None, None] | None = None,
    bits: int | None = None, *,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    ref: vs.VideoNode | None = None,
    film_thr: float = 99.0,
    name: str | Literal[False] = False,
    **kwargs: Any
) -> vs.VideoNode | source_func:
    if filepath is None:
        return partial(  # type: ignore
            source, bits=bits if bits is not None else filepath, matrix=matrix, transfer=transfer, primaries=primaries,
            chroma_location=chroma_location, color_range=color_range, field_based=field_based, ref=ref,
            film_thr=film_thr, name=name, **kwargs
        )

    clip = None

    filepath, file = parse_video_filepath(filepath)

    indexer_kwargs = dict[str, Any]()
    props = dict[str, Any]()

    to_skip = to_arr(kwargs.get('_to_skip', []))

    if file.ext is IndexingType.LWI:
        clip = LSMAS.source_func(filepath, **kwargs)
    elif file.ext is IndexingType.DGI:
        idx_info = DGIndexNV().get_info(filepath, 0).footer

        if idx_info.film >= film_thr:
            indexer_kwargs |= dict(fieldop=1)
            props |= dict(DgiFieldOp=1, _FieldBased=0)

        clip = DGIndexNV().source_func(filepath, **(kwargs | indexer_kwargs))

        props |= dict(DgiFieldOp=0, DgiOrder=idx_info.order, DgiFilm=idx_info.film)
    elif file.file_type is FileType.IMAGE:
        clip = IMWRI.source_func(filepath, **kwargs)
    else:

        try:
            try:
                from pymediainfo import MediaInfo  # type: ignore
            except ImportError:
                ...
            else:
                tracks = MediaInfo.parse(filepath, parse_speed=0.25).video_tracks
                if tracks:
                    trackmeta = tracks[0].to_data()

                    video_format = trackmeta.get("format")

                    if video_format is not None:
                        video_fmt = str(video_format).strip().lower()

                        if video_fmt == 'ffv1':
                            raise RuntimeError

                        bitdepth = trackmeta.get('bit_depth')

                        if bitdepth is not None and video_fmt == 'avc' and int(bitdepth) > 8:
                            raise RuntimeError

            indexer, filepath_dgi = FFMS2 if FFMS2().version[0] >= 5 else BestSource(), SPath(filepath)

            clip = indexer.source_func(filepath_dgi, **indexer_kwargs)
        except (RuntimeError, AttributeError, FileNotFoundError):
            indexers = list[type[Indexer]]([D2VWitch, DGIndex])

            # FFMS2 is preferred over LSMAS, but only major version 5>=.
            if FFMS2().version[0] >= 5:
                indexers.insert(0, FFMS2)
            else:
                indexers.insert(0, LSMAS)

            try:
                from vspreview import is_preview

                best_last = is_preview()
            except BaseException:
                best_last = False

            if best_last:
                indexers.append(BestSource)
            else:
                indexers.insert(0, BestSource)

            for indexerr in indexers:
                if indexerr in to_skip:
                    continue

                try:
                    clip = indexerr.source(filepath, bits=bits)
                    break
                except Exception as e:
                    if 'bgr0 is not supported' in str(e):
                        try:
                            clip = indexerr.source(filepath, format='rgb24', bits=bits)
                            break
                        except Exception:
                            ...

    if clip is None:
        raise CustomRuntimeError(f'None of the indexers you have installed work on this file! "{filepath}"')

    props |= dict(IdxFilePath=str(filepath))

    if name:
        props |= dict(Name=name)

    clip = clip.std.SetFrameProps(**props)

    if ref:
        clip = match_clip(clip, ref, length=file.file_type is FileType.IMAGE)

    return initialize_clip(
        clip, bits, matrix=matrix, transfer=transfer,
        primaries=primaries, chroma_location=chroma_location,
        color_range=color_range, field_based=field_based
    )
