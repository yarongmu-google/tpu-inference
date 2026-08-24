"""Draw a hardware-unit timeline from an emitted instruction dump.

Maps every op to the unit that executes it and prints one row per unit,
so the ISSUE ORDER is visible: '#' = that unit is busy in this column,
' ' = idle. Columns are fixed-size buckets of consecutive ops, so a row
that is solid while others are blank means that phase is single-unit
bound - i.e. NOT overlapped with anything else.

    python tmp/llo_timeline.py <dump> [--cols 200] [--region moe_gmm1]
"""

import collections
import re
import sys

# stores have no SSA result; match them as bare leading ops too
OP_RE = re.compile(r'=\s*(?:")?(llo\.[\w.]+)|^\s+(llo\.[\w.]+)')
TRACE_RE = re.compile(r'trace_start.*?message\s*=\s*"([^"]+)"')

# op-name fragment -> unit. First match wins, so order matters.
UNIT_RULES = [
    ("MXU", ("vmatmul", "vmatprep", "vmatres", "vlatchi", "vdwg", "vmatpush")),
    ("EUP", ("vexp", "vrecip", "vrsqrt", "vtanh", "vlog", "vpow", "vsig")),
    ("XLU", ("vperm", "vsetperm", "vpermres", "vxpose", "vrot", "slane",
             "vcombine", "vbcast", "vslreplicate", "vslaneid")),
    ("LDST", ("vector_load", "vector_store", "vst", "vld", "sld", "saddr")),
    ("DMA", ("enqueue_dma", "dma_done", "sync", "semaphore", "wait")),
    ("SCALAR", ("llo.s",)),
    ("VALU", ()),          # default for everything else vector-ish
]
UNITS = ["MXU", "VALU", "XLU", "EUP", "LDST", "DMA", "SCALAR"]


def unit_of(op: str) -> str:
    for unit, frags in UNIT_RULES:
        if unit == "VALU":
            continue
        for f in frags:
            if f in op:
                return unit
    return "VALU"


def main() -> None:
    path = sys.argv[1]
    cols = 200
    region_filter = None
    for i, a in enumerate(sys.argv):
        if a == "--cols":
            cols = int(sys.argv[i + 1])
        if a == "--region":
            region_filter = sys.argv[i + 1]

    seq: list[tuple[str, str]] = []          # (region, unit) in issue order
    depth: list[str] = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = TRACE_RE.search(line)
            if m:
                depth.append(m.group(1))
                continue
            if "trace_stop" in line:
                if depth:
                    depth.pop()
                continue
            m = OP_RE.search(line)
            if not m:
                continue
            op = m.group(1) or m.group(2)
            region = depth[-1] if depth else "<top>"
            if region_filter and region != region_filter:
                continue
            seq.append((region, unit_of(op)))

    if not seq:
        print("no ops matched")
        return

    bucket = max(1, len(seq) // cols)
    ncol = (len(seq) + bucket - 1) // bucket
    rows = {u: [" "] * ncol for u in UNITS}
    counts = {u: 0 for u in UNITS}
    for i, (_, u) in enumerate(seq):
        counts[u] += 1
        c = i // bucket
        rows[u][c] = "#"

    # region ruler: first letter of each region change
    ruler = [" "] * ncol
    labels = []
    prev = None
    for i, (r, _) in enumerate(seq):
        if r != prev:
            c = i // bucket
            if ruler[c] == " ":
                ruler[c] = str(len(labels) % 10)
                labels.append((len(labels) % 10, r, i))
            prev = r

    total = len(seq)
    print(f"{path}\n{total} ops, {bucket} ops/column, {ncol} columns\n")
    for u in UNITS:
        pct = 100.0 * counts[u] / total
        print(f"{u:>6} |{''.join(rows[u])}| {counts[u]:>7} ({pct:4.1f}%)")
    print(f"{'':>6} |{''.join(ruler)}|")
    print("\nregions (marker, name, first op index):")
    seen = set()
    for mark, name, idx in labels:
        if name in seen:
            continue
        seen.add(name)
        print(f"  {mark}  {name:<24} @{idx}")


if __name__ == "__main__":
    main()
