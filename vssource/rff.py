from vstools import vs, core, FrameRange
from typing import List, Any
from functools import partial
from copy import deepcopy

__all__ = [
    'apply_rff_array', 'apply_rff_video',
    'cut_array_on_ranges', 'cut_node_on_ranges',
]


def apply_rff_array(old_array: List[any],
                    rff: List[int],
                    tff: List[int],
                    prog: List[int],
                    prog_seq: List[int]) -> List[any]:
    array_double_rate = []

    for a in range(len(rff)):
        if prog_seq[a] == 0:
            if rff[a]:
                array_double_rate += [old_array[a], old_array[a], old_array[a]]
            else:
                array_double_rate += [old_array[a], old_array[a]]
        else:
            if rff[a]:
                if tff[a]:
                    array_double_rate += [old_array[a], old_array[a],
                                          old_array[a], old_array[a], old_array[a], old_array[a]]
                else:
                    array_double_rate += [old_array[a], old_array[a], old_array[a], old_array[a]]
            else:
                array_double_rate += [old_array[a], old_array[a]]

    assert (len(array_double_rate) % 2) == 0

    array_return = []
    for i in range(len(array_double_rate) // 2):
        f1 = array_double_rate[i * 2 + 0]
        f2 = array_double_rate[i * 2 + 1]
        if f1 != f2:
            print("Warning ambigious pattern due to rff {} {}".format(f1, f2))
            print("This probably just means telecine across chapter boundary")
        array_return += [f1]

    return array_return


def apply_rff_video(node: vs.VideoNode,
                    rff: List[int],
                    tff: List[int],
                    prog: List[int],
                    prog_seq: List[int]) -> vs.VideoNode:
    assert len(node) == len(rff) == len(tff) == len(prog) == len(prog_seq)

    fields = []
    tfffs = core.std.SeparateFields(core.std.RemoveFrameProps(node, props=["_FieldBased", "_Field"]), tff=True)

    for i, (current_prg_seq, current_prg, current_rff, current_tff) in enumerate(zip(prog_seq, prog, rff, tff)):
        if not current_prg_seq:
            if current_tff:
                first_field = 2 * i
                second_field = 2 * i + 1
            else:
                first_field = 2 * i + 1
                second_field = 2 * i

            fields += [{"n": first_field, "tf": current_tff, "prg": False},
                       {"n": second_field, "tf": not current_tff, "prg": False}]
            if current_rff:
                assert current_prg
                fields += [deepcopy(fields[-2])]
        else:
            assert current_prg

            cnt = 1
            if current_rff:
                cnt += 1 + int(current_tff)

            fields += [{"n": 2 * i, "tf": 1, "prg": True}, {"n": 2 * i + 1, "tf": 0, "prg": True}] * cnt

    # TODO: mark known progressive frames as progressive

    assert (len(fields) % 2) == 0

    for a, (tf, bf) in enumerate(zip(fields[::2], fields[1::2])):
        if tf["tf"] == bf["tf"]:
            bf["tf"] = not bf["tf"]
            print(f"Invalid field transition @{a}")

    for a in range(len(fields) // 2):
        if fields[a * 2]["tf"] == fields[a * 2 + 1]["tf"]:
            print("Could not fix sth for some reason", a)
            assert False

    final = clip_remap_frames(tfffs, [x["n"] for x in fields])

    def set_field(n, f, fields):
        fout = f.copy()
        fld = fields[n]
        if "_FieldBased" in fout.props:
            del fout["_FieldBased"]

        fout.props['_Field'] = fld["tf"]
        return fout

    final = vs.core.std.ModifyFrame(clip=final, clips=final, selector=partial(set_field, fields=fields))

    woven = core.std.DoubleWeave(final)
    woven = core.std.SelectEvery(woven, 2, 0)

    def update_progressive(n, f, fields):
        fout = f.copy()
        tf = fields[n * 2]
        bf = fields[n * 2 + 1]
        if tf["prg"] and bf["prg"]:
            fout.props['_FieldBased'] = 0
        return fout

    woven = vs.core.std.ModifyFrame(clip=woven, clips=woven, selector=partial(update_progressive, fields=fields))

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
