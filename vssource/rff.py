from vstools import vs, core, SPath, normalize_list_to_ranges, FrameRange
from typing import List, Any
from functools import partial

__all__ = [
    'apply_rff_array', 'apply_rff_video',
    'cut_array_on_ranges', 'cut_node_on_ranges',
]


def apply_rff_array(rff: List[int], old_array: List[any]) -> List[any]:
    array_double_rate = []

    for a in range(len(rff)):
        if rff[a] == 1:
            array_double_rate += [old_array[a], old_array[a], old_array[a]]
        else:
            array_double_rate += [old_array[a], old_array[a]]

    assert (len(array_double_rate) % 2) == 0

    array_return = []
    for i in range(len(array_double_rate) // 2):
        f1 = array_double_rate[i * 2 + 0]
        f2 = array_double_rate[i * 2 + 1]
        if f1 != f2:
            print("Warning ambigious pattern due to rff {} {}".format(f1, f2))
        array_return += [f1]

    return array_return


def apply_rff_video(node: vs.VideoNode, rff: List[int], tff: List[int]) -> vs.VideoNode:
    assert len(node) == len(rff)
    assert len(rff) == len(tff)

    fields = []
    tfffs = core.std.SeparateFields(core.std.RemoveFrameProps(node, props=["_FieldBased", "_Field"]), tff=True)

    for i in range(len(rff)):
        current_tff = tff[i]
        current_bff = int(not current_tff)

        if current_tff == 1:
            first_field = 2 * i
            second_field = 2 * i + 1
        else:
            first_field = 2 * i + 1
            second_field = 2 * i

        fields += [{"n": first_field, "tf": current_tff}, {"n": second_field, "tf": current_bff}]
        if rff[i] == 1:
            fields += [fields[-2]]

    assert (len(fields) % 2) == 0

    for a in range(len(fields) // 2):
        tf = fields[a * 2]
        bf = fields[a * 2 + 1]

        # should this assert?
        # assert tf["tf"] != bf["tf"]
        if tf["tf"] == bf["tf"]:
            print("invalid field transition @{}".format(a))

    fields = [x["n"] for x in fields]

    final = clip_remap_frames(tfffs, fields)

    final = core.std.RemoveFrameProps(final, props=["_FieldBased", "_Field"])
    woven = core.std.DoubleWeave(final, tff=True)
    woven = core.std.SelectEvery(woven, 2, 0)
    woven = core.std.SetFieldBased(woven, 2)

    return woven


def cut_array_on_ranges(array: List[Any], ranges: List[FrameRange]) -> List[Any]:
    remap_frames = tuple[int, ...]([
        x for y in [
            range(rrange[0], rrange[1] + 1) for rrange in ranges
        ] for x in y
    ])
    newarray = []
    for i in remap_frames:
        newarray += [array[i]]
    return newarray


def cut_node_on_ranges(node: vs.VideoNode, ranges: List[FrameRange]) -> vs.VideoNode:
    remap_frames = tuple[int, ...]([
        x for y in [
            range(rrange[0], rrange[1] + 1) for rrange in ranges
        ] for x in y
    ])
    return clip_remap_frames(node, remap_frames)


def clip_remap_frames(node: vs.VideoNode, remap_frames) -> vs.VideoNode:  # remap_frames: List[int]
    blank = node.std.BlankClip(length=len(remap_frames))

    def noname(n, target_node, targetremap_frames):
        return target_node[targetremap_frames[n]]

    return blank.std.FrameEval(partial(noname, target_node=node, targetremap_frames=remap_frames))
