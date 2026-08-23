"""Summarize a Mosaic dump into a compact op histogram.

The raw pass dumps are hundreds of MB - unpushable and unreadable. What we
actually want is: which ops, how many, and in which named_scope region.
Emits a few KB of text instead.

    python tmp/dump_histogram.py tmp/mosaic_dump/<file> [more files...]
"""

import collections
import re
import sys

# "%12 = tpu.matmul ..." / '"tpu.enqueue_dma"(...)' / "  vmatmul ..."
OP_RE = re.compile(r'"([a-z_][\w.]*\.[\w.]+)"|=\s*([a-z_][\w.]*\.[\w.]+)\s')
TRACE_RE = re.compile(r'trace_start.*?message\s*=\s*"([^"]+)"')


def summarize(path: str) -> None:
    region = "<toplevel>"
    per_region: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    depth: list[str] = []
    total = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            m = TRACE_RE.search(line)
            if m:
                depth.append(m.group(1))
                region = depth[-1]
                continue
            if "trace_stop" in line:
                if depth:
                    depth.pop()
                region = depth[-1] if depth else "<toplevel>"
                continue
            for a, b in OP_RE.findall(line):
                op = a or b
                per_region[region][op] += 1
                total += 1

    print(f"\n===== {path}")
    print(f"total ops: {total}")
    for reg, counter in sorted(per_region.items(),
                               key=lambda kv: -sum(kv[1].values())):
        n = sum(counter.values())
        print(f"\n-- {reg}: {n} ops")
        for op, c in counter.most_common(25):
            print(f"     {c:>8}  {op}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
